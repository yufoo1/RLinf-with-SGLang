Train ActiveVLN with EmbodiInfer
================================================================

Use this recipe to run synchronous R2R GRPO training with RLinf and the official
`BUAA-CI-LAB EmbodiInfer <https://github.com/BUAA-CI-LAB/EmbodiInfer>`_ rollout
runtime. Prepare an ActiveVLN checkpoint and environment server, install the
pinned inference package, then launch the training configuration below.

Overview
--------

RLinf owns the environment history, GRPO update, and weight transport. EmbodiInfer
runs batched vision/text prefill and token decoding. The actor uses EmbodiInfer's
differentiable full-sequence forward to score the sampled response tokens.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      ActiveVLN / Qwen2.5-VL

   .. grid-item-card:: Algorithms
      :text-align: center

      GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      R2R navigation

   .. grid-item-card:: Hardware
      :text-align: center

      CPU contract tests; GPU training pending

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Suite
     - Configuration
     - Environment
   * - R2R
     - ``activevln_r2r_grpo_embodiinfer``
     - Official ActiveVLN HTTP server

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The server supplies the instruction, images, and feedback. Each environment
keeps its complete conversation, and one action chunk is one response containing
up to three navigation actions. GRPO groups start from the same episode ID.

.. list-table::
   :header-rows: 1

   * - Value
     - Representation
   * - Observation
     - Per-environment ``messages`` and ``images`` lists
   * - Action
     - Integer token IDs of shape ``[B, 1, max_new_tokens]``
   * - Reward
     - Server success reward
   * - Prompt
     - Server system prompt and complete conversation
   * - Actor input
     - Sampled tokens, mRoPE positions, image patches/grids, lengths, sampling parameters
   * - Loss mask
     - Generated tokens through EOS or stop; prompt and padding excluded

Every rollout prefills the complete history using current weights. Weight sync
and model offload discard inference buffers; sampling generator state survives
runtime rebuilds. The final two language layers and final norm train by default,
while vision and all other parameters remain frozen. Set
``actor.model.activevln.train_last_n_layers=-1`` to train all language layers
and the final norm. Root FSDP1 with ``use_orig_params=True`` owns the trainable
parameters.

Installation
------------

Use a dedicated RLinf environment: the ActiveVLN profile requires Transformers
4.51.3. From the RLinf checkout, install the trainer and pinned official runtime:

.. code-block:: bash

   python -m pip install -e . -r requirements/embodiinfer.txt

The requirements file pins EmbodiInfer to commit
``d8cc7105e52ae08302019c620cf656a1c69be2b0``. Install the same profile on each
worker node. This integration calls the package directly in the rollout process.

Prepare Matterport3D scenes and VLN-CE datasets and start the server following
`ActiveVLN's instructions <https://github.com/arvillion/ActiveVLN#quick-start>`_.
Keep Habitat in the server's environment. Install the client in RLinf's environment:

.. code-block:: bash

   python -m pip install -e /path/to/ActiveVLN/vlnce_server
   export PYTHONPATH="/path/to/ActiveVLN:${PYTHONPATH:-}"

Workers close their own server environments on reset and when all rows finish.

Download the Model
------------------

Download a local R2R checkpoint, then use the same directory for the policy and
environment tokenizer:

.. code-block:: bash

   hf download Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn \
     --local-dir /path/to/activevln

Run It
------

Set the server address and episode IDs to match the server's loaded dataset,
then start the synchronous training entry point:

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   python examples/embodiment/train_embodied_agent.py \
     --config-name activevln_r2r_grpo_embodiinfer \
     actor.model.model_path=/path/to/activevln \
     env.train.tokenizer_path=/path/to/activevln \
     env.eval.tokenizer_path=/path/to/activevln \
     env.train.server_url=http://127.0.0.1:5001 \
     env.train.episode_ids='[1,2]'

This short recipe runs two training iterations, each collecting navigation turns
and updating the actor.
``max_episode_steps`` counts responses; ``step_budget`` counts simulator actions.
Increase ``max_context``, ``max_image_tokens``, and ``max_images`` for longer
histories. Capacity overflow raises an error before inference.

``rollout.generation_backend=embodiinfer`` selects the worker. Its model options
under ``rollout.model.embodiinfer`` are passed to the official batched runtime.
The default uses eager tensor batching. On a prepared GPU, you can separately
validate ``cuda_graph``, ``fused_ops``, ``split_attention``, and
``grammar_tree_budget`` before enabling them. Their implementations remain in
EmbodiInfer. Full-history rollout requires ``resident_kv=False``,
``gpu_preprocess=False``, and ``tree_decode=False``.

Keep ``algorithm.recompute_logprobs=True`` so the actor recomputes old
probabilities using its own forward before an update. Batched incremental decode
and full-sequence replay can round differently, especially in BF16. Sampling uses
temperature, top-p, and repetition penalty; ``top_k`` must be zero. The recipe
uses top-p 1.0 to retain full distribution support.

Visualization and Results
-------------------------

Watch episode returns and success metrics in TensorBoard under the configured
``runner.logger.log_path``. The initial validation uses randomly initialized
small Qwen models on CPU with FP32 and BF16: rollout/replay probabilities,
response masks, actor gradients, parameter updates, and runtime invalidation.
Run it without a GPU or checkpoint download:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=. \
     python -m pytest tests/unit_tests/test_activevln.py \
       tests/unit_tests/test_activevln_env.py -q

The dedicated CPU CI step installs the pinned profile before these tests.
Real-checkpoint GPU training, FSDP execution, full Habitat navigation, learning
curves, and performance have not been validated for this backend. GPU results
from the earlier SGLang prototype do not validate this integration.
