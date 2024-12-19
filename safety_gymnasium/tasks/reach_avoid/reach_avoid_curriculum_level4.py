# Copyright 2024 Jacques Cloete. All Rights Reserved.
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
# ==============================================================================
"""Reach-avoid curriculum level 4 (reset on success/failure)."""

from safety_gymnasium.tasks.reach_avoid.reach_avoid_curriculum_level1 import (
    ReachAvoidCurriculumLevel1,
)


class ReachAvoidCurriculumLevel4(ReachAvoidCurriculumLevel1):
    """An agent must reach a goal while avoiding hazards."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.hazards.keepout = 0.2
        self.hazards.size = 0.3
        self.hazards.num = 1
        self.agent.placements = [[-1, -1, 1, 0]]
        self.goal.placements = [[-1, 1, 1, 2]]
        self.hazards.placements = [[-1.5, 0, 1.5, 1.5]]
