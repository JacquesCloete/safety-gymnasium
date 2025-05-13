# Copyright 2025 Jacques Cloete. All Rights Reserved.
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
"""RGBZoneEnv level 3."""

from safety_gymnasium.assets.geoms import RGBZones
from safety_gymnasium.tasks.rgb_zone_env.rgb_zone_env_base_task import RGBZoneEnvBaseTask


class RGBZoneEnvLevel3(RGBZoneEnvBaseTask):
    """Four zones."""

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)

        self._add_geoms(
            RGBZones(
                zones_id=0,
                size=self.zone_size,
                num=1,
                rgb_lower_bounds=[0, 0, 0],
                rgb_upper_bounds=[1, 1, 1],
            ),
            RGBZones(
                zones_id=1,
                size=self.zone_size,
                num=1,
                rgb_lower_bounds=[0, 0, 0],
                rgb_upper_bounds=[1, 1, 1],
            ),
            RGBZones(
                zones_id=2,
                size=self.zone_size,
                num=1,
                rgb_lower_bounds=[0, 0, 0],
                rgb_upper_bounds=[1, 1, 1],
            ),
            RGBZones(
                zones_id=3,
                size=self.zone_size,
                num=1,
                rgb_lower_bounds=[0, 0, 0],
                rgb_upper_bounds=[1, 1, 1],
            ),
        )
