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

"""Exercise the official server boundary without installing Habitat in workers."""

import sys
from dataclasses import make_dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.sim.activevln.activevln_env import ActiveVLNEnv


@pytest.fixture
def env(monkeypatch):
    image = Image.new("RGB", (28, 28))
    observation = {
        "obs_str": "View: <image> Next action?",
        "multi_modal_data": {"<image>": [image]},
    }

    class Client:
        def __init__(self, *args, **kwargs):
            self.closed = []
            self.calls = []
            self.turn = 0

        def create_environments_batch(self, configs):
            self.configs = configs

        def reset_batch(self, seeds):
            self.turn = 0
            return {key: ([observation], {}) for key in seeds}

        def get_system_prompts_batch(self, ids):
            return dict.fromkeys(ids, "Navigate to the kitchen.")

        def step_batch(self, actions):
            self.calls.append(actions)
            self.turn += 1
            return {
                key: (
                    observation,
                    float(index == 0),
                    index == 0,
                    {"task_success": index == 0},
                )
                for index, key in enumerate(actions)
            }

        def close_batch(self, ids):
            self.closed.append(list(ids))

    client_module = ModuleType("vlnce_server.client")
    client_module.BatchEnvClient = Client
    config_module = ModuleType("vlnce_server.env_config")
    config_module.VLNCEEnvConfig = make_dataclass(
        "VLNCEEnvConfig",
        [
            "episode_id",
            "data_source",
            "action_space",
            "prompt_format",
            "max_actions_per_step",
            "step_budget",
            "turn_budget",
            "reward_type",
            "success_reward_base",
            "save_as_video",
        ],
    )
    monkeypatch.setitem(sys.modules, "vlnce_server", ModuleType("vlnce_server"))
    monkeypatch.setitem(sys.modules, "vlnce_server.client", client_module)
    monkeypatch.setitem(sys.modules, "vlnce_server.env_config", config_module)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: SimpleNamespace(
            batch_decode=lambda rows, **kwargs: ["move forward 25cm" for row in rows]
        ),
    )
    cfg = OmegaConf.create(
        {
            "group_size": 2,
            "episode_ids": [11, 22, 33],
            "auto_reset": False,
            "server_url": "http://test",
            "tokenizer_path": "unused",
            "data_source": "r2r",
            "action_space": "r2r",
            "step_budget": 500,
            "max_episode_steps": 2,
            "reward_coef": 1.0,
            "seed": 42,
        }
    )
    return ActiveVLNEnv(cfg, 4, 0, 1, worker_info=None)


def test_episode_grouping_history_terminal_mask_and_reset(env):
    first, _ = env.reset()
    assert [x["env_config"]["episode_id"] for x in env.client.configs.values()] == [
        11,
        11,
        22,
        22,
    ]
    assert len(set(env._ids)) == 4
    actions = torch.zeros(4, 1, 8, dtype=torch.long)
    observations, rewards, terminated, truncated, infos = env.chunk_step(actions)
    assert rewards[:, 0].tolist() == [1, 0, 0, 0]
    assert terminated[:, 0].tolist() == [True, False, False, False]
    assert not truncated.any()
    assert infos[0]["episode"]["success_once"].tolist() == [True, False, False, False]
    assert len(first["messages"][0]) == 2
    assert len(observations[0]["messages"][0]) == 4
    assert len(observations[0]["images"][0]) == 2
    _, rewards, terminated, truncated, _ = env.chunk_step(actions)
    assert env._ids[0] not in env.client.calls[-1]
    assert rewards[0] == 0
    assert (terminated | truncated).all()
    calls = len(env.client.calls)
    _, rewards, done, _, _ = env.chunk_step(actions)
    assert done.all() and not rewards.any()
    assert len(env.client.calls) == calls
    env.update_reset_state_ids()
    env.reset()
    assert [x["env_config"]["episode_id"] for x in env.client.configs.values()] == [
        33,
        33,
        11,
        11,
    ]
    assert not env._done.any()
    env.close()
    env.close()
    assert len(env.client.closed) == 2


def test_reject_partial_reset_and_malformed_server_observation(env):
    with pytest.raises(ValueError, match="complete GRPO groups"):
        env.reset([0])
    with pytest.raises(ValueError, match="placeholders"):
        env._append_observation(
            0, {"obs_str": "<image>", "multi_modal_data": {"<image>": []}}
        )


def test_observations_split_and_merge_through_worker_messages(env):
    from rlinf.data.schema.embodied_types import EnvOutput, PolicyInput

    obs, _ = env.reset()
    serialized = EnvOutput(obs=obs, final_obs=obs).to_dict()
    assert serialized["final_obs"]["messages"] == obs["messages"]
    request = PolicyInput(obs=serialized["obs"])
    shards = request.split([2, 2])
    merged = PolicyInput.merge(shards)
    assert merged.request_sizes == [2, 2]
    assert merged.obs["messages"] == obs["messages"]
    assert merged.obs["task_descriptions"] == obs["task_descriptions"]
    assert len(merged.obs["images"]) == 4


def test_worker_offsets_partition_episode_groups(env):
    env.cfg.episode_ids = list(range(1, 9))
    second_worker = ActiveVLNEnv(env.cfg, 4, 1, 2, worker_info=None)
    second_worker.reset()

    def episodes():
        return [
            entry["env_config"]["episode_id"]
            for entry in second_worker.client.configs.values()
        ]

    assert episodes() == [3, 3, 4, 4]
    second_worker.update_reset_state_ids()
    second_worker.reset()
    assert episodes() == [7, 7, 8, 8]
    second_worker.close()
