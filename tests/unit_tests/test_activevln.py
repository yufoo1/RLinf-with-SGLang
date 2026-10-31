# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU contracts for the official EmbodiInfer ActiveVLN integration."""

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.activevln.policy import ActiveVLNPolicy


class Processor:
    """Small external processor/tokenizer fixture with one image token."""

    pad_token_id = 0

    @property
    def tokenizer(self):
        return self

    def decode(self, tokens, **kwargs):
        return "continue"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, images, **kwargs):
        ids = [3, 93, 94, 90, 4] + ([5, 6] if text == ["long"] else [])
        return SimpleNamespace(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            pixel_values=torch.linspace(-1, 1, 4 * 1176).reshape(4, 1176),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )


@pytest.fixture
def policy():
    official = pytest.importorskip("embodiinfer.policies.activevln.modeling_activevln")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.3":
        pytest.skip("ActiveVLN uses the isolated Transformers 4.51.3 profile")
    torch.manual_seed(4)
    cfg = transformers.Qwen2_5_VLConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_scaling={"type": "default", "mrope_section": [2, 3, 3]},
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_heads": 2,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
        },
        image_token_id=94,
        video_token_id=95,
        vision_start_token_id=93,
        eos_token_id=2,
        pad_token_id=0,
    )
    cfg._attn_implementation = "eager"
    qwen = transformers.Qwen2_5_VLForConditionalGeneration(cfg).eval()
    qwen.requires_grad_(False)
    qwen.model.layers[-1].requires_grad_(True)
    qwen.model.norm.requires_grad_(True)
    processor = Processor()
    core = official.ActiveVLNPolicy(qwen, processor, max_new_tokens=4, max_context=32)
    return ActiveVLNPolicy(
        core,
        processor,
        max_image_tokens=4,
        max_images=2,
        runtime_options={
            "workspace_tokens": 16,
            "query_bucket_size": 4,
            "sampling_seed": 7,
        },
    ).eval()


def observations():
    return {
        "messages": [
            [{"role": "user", "content": "short"}],
            [{"role": "user", "content": "long"}],
        ],
        "images": [[object()], [object()]],
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("do_sample,top_p", [(True, 1.0), (True, 0.9), (False, 1.0)])
def test_rollout_replay_and_gradients(policy, dtype, do_sample, top_p):
    policy.to(dtype=dtype)
    actions, result = policy.predict_action_batch(
        observations(),
        do_sample=do_sample,
        temperature=0.7,
        top_p=top_p,
        repetition_penalty=1.05,
    )
    assert actions.shape == (2, 1, 4)
    assert actions.dtype == torch.long and actions.device.type == "cpu"
    replay = policy(forward_inputs=result["forward_inputs"], compute_entropy=True)
    assert replay["logprob_mask"].shape == (2, 4)
    tolerance = 0.04 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(
        replay["logprobs"],
        result["prev_logprobs"],
        atol=tolerance,
        rtol=0,
        check_dtype=False,
    )
    mask = replay["logprob_mask"]
    loss = -(replay["logprobs"] * mask).sum() - 0.01 * replay["entropy"].sum()
    loss.backward()
    assert policy.core.qwen.model.layers[0].self_attn.q_proj.weight.grad is None
    gradient = policy.core.qwen.model.layers[-1].self_attn.q_proj.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    assert all(
        parameter.grad is None for parameter in policy.core.qwen.visual.parameters()
    )


def test_eos_masks_padding_but_keeps_terminal_token(policy):
    with torch.no_grad():
        policy.core.qwen.lm_head.weight.zero_()
    policy.core.eos_token_ids = (0,)
    _, result = policy.predict_action_batch(observations(), do_sample=False)
    replay = policy(forward_inputs=result["forward_inputs"])
    assert replay["logprob_mask"].tolist() == [[True, False, False, False]] * 2
    assert (replay["logprobs"][:, 1:] == 0).all()
    assert (replay["logprobs"][:, 0] < 0).all()


def test_optimizer_refit_and_runtime_rebuild(policy):
    actor = copy.deepcopy(policy)
    _, result = policy.predict_action_batch(observations(), do_sample=False)
    batch = result["forward_inputs"]
    before = actor(forward_inputs=batch)["logprobs"].detach()
    optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad], lr=0.01
    )
    (-actor(forward_inputs=batch)["logprobs"].sum()).backward()
    optimizer.step()
    policy.load_state_dict(actor.state_dict())
    assert policy._runtime is None
    actual = policy(forward_inputs=batch)["logprobs"]
    assert not torch.equal(actual, before)
    torch.testing.assert_close(
        actual, actor(forward_inputs=batch)["logprobs"], atol=0, rtol=0
    )
    policy.predict_action_batch(observations(), do_sample=False)
    assert policy._runtime is not None
    policy.to("cpu")
    assert policy._runtime is None


def test_runtime_rebuild_preserves_sampling_stream(policy):
    uninterrupted = copy.deepcopy(policy)
    policy.predict_action_batch(observations())
    uninterrupted.predict_action_batch(observations())
    policy.clear_rollout_runtime()
    actions, result = policy.predict_action_batch(observations())
    expected, reference = uninterrupted.predict_action_batch(observations())
    assert torch.equal(actions, expected)
    assert torch.equal(result["prev_logprobs"], reference["prev_logprobs"])


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"top_k": 2}, "top_k"),
        ({"temperature": 0}, "temperature"),
        ({"max_new_tokens": 5}, "max_new_tokens"),
    ],
)
def test_invalid_sampling_fails_before_inference(policy, kwargs, message):
    with pytest.raises(ValueError, match=message):
        policy.predict_action_batch(observations(), **kwargs)
    assert policy._runtime is None


def test_history_overflow_fails_before_inference(policy):
    policy.max_context = 8
    with pytest.raises(ValueError, match="max_context"):
        policy.predict_action_batch(observations())
    assert policy._runtime is None


@pytest.fixture
def recipe(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "examples/embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(directory))
    with initialize_config_dir(
        config_dir=str(directory / "config"), version_base="1.1"
    ):
        return compose(config_name="activevln_r2r_grpo_embodiinfer")


def test_recipe_validates(recipe):
    from rlinf.config import validate_embodied_cfg

    validate_embodied_cfg(recipe)
    assert recipe.rollout.generation_backend == "embodiinfer"
    assert recipe.algorithm.recompute_logprobs


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("actor.model.is_lora", True, "is_lora"),
        ("actor.fsdp_config.strategy", "fsdp2", "FSDP1"),
        ("actor.fsdp_config.use_orig_params", False, "use_orig_params"),
        ("algorithm.adv_type", "gae", "GRPO"),
        ("algorithm.recompute_logprobs", False, "recompute_logprobs"),
        ("env.train.group_size", 1, "group_size"),
        ("rollout.generation_backend", "huggingface", "embodiinfer"),
        ("rollout.enable_cuda_graph", True, "CUDA Graph"),
        ("rollout.sampling_params.top_k", 4, "top_k"),
        ("rollout.model.embodiinfer.resident_kv", True, "resident_kv"),
    ],
)
def test_invalid_recipe_fails_early(recipe, key, value, message):
    from rlinf.config import validate_embodied_cfg

    OmegaConf.update(recipe, key, value, force_add=True)
    with pytest.raises(ValueError, match=message):
        validate_embodied_cfg(recipe)


@pytest.mark.parametrize("count,trainable_layers", [(0, 0), (1, 1), (-1, 2)])
def test_checkpoint_builder_freezes_parameters(
    policy, recipe, monkeypatch, count, trainable_layers
):
    monkeypatch.setattr("embodiinfer.policies.make_policy", lambda *a, **k: policy.core)
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained", lambda *a, **k: policy.processor
    )
    recipe.actor.model.activevln.train_last_n_layers = count
    built = ActiveVLNPolicy.from_config(recipe.actor.model, torch.float32)
    assert (
        sum(
            any(p.requires_grad for p in layer.parameters())
            for layer in built.core.qwen.model.layers
        )
        == trainable_layers
    )
    assert built.core.qwen.model.norm.weight.requires_grad
    assert not any(p.requires_grad for p in built.core.qwen.visual.parameters())
    assert not built.core.qwen.get_input_embeddings().weight.requires_grad


def test_worker_patch_sync_invalidates_runtime(policy):
    from rlinf.hybrid_engines.weight_syncer import PatchWeightSyncer, WeightPatch
    from rlinf.scheduler import AcceleratorType
    from rlinf.workers.rollout.embodiinfer import EmbodiInferRolloutWorker

    actor = copy.deepcopy(policy)
    _, result = policy.predict_action_batch(observations(), do_sample=False)
    receiver = PatchWeightSyncer(
        snapshot_device="cpu", transport_device="cpu", compression_algorithm="none"
    )
    worker = object.__new__(EmbodiInferRolloutWorker)
    worker.hf_model = policy
    worker.weight_syncer = receiver
    worker.actor_group_name = "actor"
    worker.actor_weight_src_rank = 0
    worker._group_name = "rollout"
    worker._weight_sync_rollout_ranks = [0]
    worker._sync_weight_comm_options = None
    worker.finished_episodes = 0
    worker._timer_metrics = {}
    worker._accelerator_type = AcceleratorType.NO_ACCEL

    async def run():
        to_rollout, to_actor = asyncio.Queue(), asyncio.Queue()
        # Replace only process transport; run the real worker and patch syncer.
        worker.broadcast = lambda *a, **k: SimpleNamespace(async_wait=to_rollout.get)
        await receiver.init_receiver(policy.state_dict(), to_rollout.get, to_actor.put)
        key = "core.qwen.model.layers.1.self_attn.q_proj.weight"
        with torch.no_grad():
            actor.state_dict()[key][0, 0] += 0.3
        payload = WeightPatch(
            version=torch.tensor(1),
            ordinals=torch.tensor([receiver.ordered_keys.index(key)]),
            nnz_per_tensor=torch.tensor([1]),
            rows=torch.tensor([0]),
            cols=torch.tensor([0]),
            values=actor.state_dict()[key][0, :1].clone().view(torch.uint8),
        )
        await to_rollout.put(payload)
        await worker.sync_model_from_actor()

    asyncio.run(run())
    assert worker.version == 1
    assert policy._runtime is None
    batch = result["forward_inputs"]
    torch.testing.assert_close(
        policy(forward_inputs=batch)["logprobs"],
        actor(forward_inputs=batch)["logprobs"],
        atol=0,
        rtol=0,
    )
