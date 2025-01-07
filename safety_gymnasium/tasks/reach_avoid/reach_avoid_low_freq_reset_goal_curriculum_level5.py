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
"""Reach-avoid curriculum level 5 (reset on success), low control frequency."""

from safety_gymnasium.tasks.reach_avoid.reach_avoid_low_freq_curriculum_level5 import (
    ReachAvoidLowFreqCurriculumLevel5,
)


class ReachAvoidLowFreqResetGoalCurriculumLevel5(ReachAvoidLowFreqCurriculumLevel5):
    """An agent must reach a goal while avoiding hazards."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.mechanism_conf.continue_on_violation = True
