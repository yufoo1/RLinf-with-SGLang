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

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiInferRolloutWorker(MultiStepRolloutWorker):
    """Run EmbodiInfer policies with RLinf's embodied channels and weight sync."""

    async def sync_model_from_actor(self):
        """Invalidate inference buffers before bucket or in-place patch updates."""
        self.hf_model.clear_rollout_runtime()
        return await super().sync_model_from_actor()
