使用 EmbodiInfer 训练 ActiveVLN
================================================================

本示例使用 RLinf 和官方 `BUAA-CI-LAB EmbodiInfer <https://github.com/BUAA-CI-LAB/EmbodiInfer>`_ rollout runtime，接入 R2R 的同步 GRPO 训练。先准备 ActiveVLN checkpoint 和环境服务，再安装固定版本的推理包，最后启动训练配置。

概述
----

RLinf 管理环境历史、GRPO 更新和权重传输，EmbodiInfer 执行批量视觉编码、文本 prefill 和 token decode。actor 调用 EmbodiInfer 的可微整段前向，重算采样回答的 token logprob。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      ActiveVLN / Qwen2.5-VL

   .. grid-item-card:: 算法
      :text-align: center

      GRPO

   .. grid-item-card:: 任务
      :text-align: center

      R2R 导航

   .. grid-item-card:: 硬件
      :text-align: center

      CPU 接口测试；GPU 训练待验证

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 任务集
     - 配置
     - 环境
   * - R2R
     - ``activevln_r2r_grpo_embodiinfer``
     - 官方 ActiveVLN HTTP 服务

观测与动作
~~~~~~~~~~

服务端返回指令、图像和动作反馈。每个环境保存完整对话，一个 action chunk 对应一条回答，由服务端解析成最多三个导航动作。同一 GRPO 分组从相同 episode ID 开始。

.. list-table::
   :header-rows: 1

   * - 数据
     - 表示
   * - Observation
     - 每个环境的 ``messages`` 和 ``images`` 列表
   * - Action
     - 形状为 ``[B, 1, max_new_tokens]`` 的整数 token ID
   * - Reward
     - 服务端成功奖励
   * - Prompt
     - 服务端系统提示词与完整对话
   * - Actor 输入
     - 采样 token、mRoPE 位置、图像 patches/grid、长度及采样参数
   * - Loss mask
     - 包含 EOS 或 stop 在内的生成 token；排除 prompt 和 padding

每次 rollout 都用当前权重重新 prefill 完整历史。同步权重或卸载模型时会清除推理缓存，重建 runtime 时保留采样随机数状态。默认只训练最后两个语言层和末尾 norm，视觉及其他参数保持冻结。设置 ``actor.model.activevln.train_last_n_layers=-1`` 可训练全部语言层和末尾 norm。训练使用根模块 FSDP1，并开启 ``use_orig_params=True``。

安装
----

ActiveVLN 使用 Transformers 4.51.3，请准备独立的 RLinf 环境。在 RLinf 仓库中安装训练框架和固定版本的官方 runtime：

.. code-block:: bash

   python -m pip install -e . -r requirements/embodiinfer.txt

依赖文件将 EmbodiInfer 固定到 commit ``d8cc7105e52ae08302019c620cf656a1c69be2b0``。所有 worker 节点安装同一版本，rollout 进程直接调用 Python 包。

按 `ActiveVLN 官方说明 <https://github.com/arvillion/ActiveVLN#quick-start>`_ 准备 Matterport3D 场景和 VLN-CE 数据，并启动环境服务。Habitat 保留在服务端环境中；RLinf 环境只需安装 client：

.. code-block:: bash

   python -m pip install -e /path/to/ActiveVLN/vlnce_server
   export PYTHONPATH="/path/to/ActiveVLN:${PYTHONPATH:-}"

worker 在 reset 和整批环境结束时关闭自己创建的服务端环境。

下载模型
--------

下载本地 R2R checkpoint，并将 policy 与环境 tokenizer 指向同一目录：

.. code-block:: bash

   hf download Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn \
     --local-dir /path/to/activevln

运行
----

将服务地址和 episode ID 改成服务端实际加载的数据，然后使用同步训练入口启动：

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   python examples/embodiment/train_embodied_agent.py \
     --config-name activevln_r2r_grpo_embodiinfer \
     actor.model.model_path=/path/to/activevln \
     env.train.tokenizer_path=/path/to/activevln \
     env.eval.tokenizer_path=/path/to/activevln \
     env.train.server_url=http://127.0.0.1:5001 \
     env.train.episode_ids='[1,2]'

这个短程配置采集两轮导航轨迹并更新 actor。``max_episode_steps`` 按回答轮数计数，``step_budget`` 按模拟器动作数计数。历史较长时，需增大 ``max_context``、``max_image_tokens`` 和 ``max_images``；超过容量会在推理前报错。

``rollout.generation_backend=embodiinfer`` 选择对应 worker。``rollout.model.embodiinfer`` 中的模型选项直接传给官方批量 runtime。默认采用 eager tensor batching；准备好 GPU 后，可逐项验证并开启 ``cuda_graph``、``fused_ops``、``split_attention`` 和 ``grammar_tree_budget``。这些优化的实现保留在 EmbodiInfer 中。完整历史模式要求 ``resident_kv=False``、``gpu_preprocess=False`` 和 ``tree_decode=False``。

保留 ``algorithm.recompute_logprobs=True``，使 actor 在更新前使用自己的前向路径重算旧 logprob。批量增量 decode 与整段回放可能产生不同的舍入结果，BF16 下尤其需要验证。采样支持 temperature、top-p 和 repetition penalty，``top_k`` 必须为零。示例使用 top-p 1.0，保留完整分布支持。

可视化与结果
------------

在 ``runner.logger.log_path`` 指定的 TensorBoard 目录中查看 episode return 和成功率。初步验证使用 CPU 上随机初始化的小型 Qwen 模型，覆盖 FP32/BF16 rollout 与回放概率、回答 mask、actor 梯度、参数更新和 runtime 失效。以下测试无需 GPU，也不下载 checkpoint：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=. \
     python -m pytest tests/unit_tests/test_activevln.py \
       tests/unit_tests/test_activevln_env.py -q

专用 CPU CI 步骤会先安装固定版本的依赖。本后端尚未验证真实 checkpoint 的 GPU 训练、FSDP 执行、完整 Habitat 导航、学习曲线及性能；早期 SGLang 原型的 GPU 测试结果不作为本次接入的验证结果。
