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
"""RGBZoneEnv base task."""

import numpy as np

from safety_gymnasium.assets.geoms import ZoneEnvWalls
from safety_gymnasium.assets.geoms.rgb_zones import RGBZones
from safety_gymnasium.bases.base_task import BaseTask


class RGBZoneEnvBaseTask(BaseTask):
    """Base task for RGBZoneEnv tasks."""

    def __init__(self, config, zone_size: float, walls=True, low_freq=True) -> None:
        super().__init__(config=config)
        self.zone_size = zone_size
        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]
        self.lidar_conf.num_bins = 16
        self.lidar_conf.max_dist = None
        self.lidar_conf.exp_gain = 0.5
        self.lidar_conf.alias = True
        self.cost_conf.constrain_indicator = False
        self.observation_flatten = False  # observation is a dict
        self.fast_rebuild = True  # only change geom positions and colors
        if walls:
            self._add_geoms(ZoneEnvWalls())
        if low_freq:
            # To match the standard setup, we need self.num_steps * self.sim_conf.frameskip_binom_n = 10000
            self.num_steps = 100  # lower episode length (to compensate for lower control frequency)
            self.sim_conf.frameskip_binom_n = 100  # lower control frequency

    def reset(self):
        """
        Reset the task. This is called by the Builder.
        Randomize colors of RGBZones before the world is built.
        """
        # Randomize colors of all Zone objects before building the world
        # self._geoms is a dict mapping name to geom object, populated by _add_geoms()
        for geom_obj in self._geoms.values():
            if isinstance(geom_obj, RGBZones):
                rgb_seed = self.random_generator.random_generator.randint(np.iinfo(np.int32).max)
                geom_obj.randomize_color(seed=rgb_seed)

        super().reset()  # Calls Underlying.reset(), which calls _build(), then _build_world_config()

    def get_task_specific_info(self) -> dict:
        info = {}
        for geom_obj in self._geoms.values():
            if isinstance(geom_obj, RGBZones):
                info[geom_obj.name] = {}
                info[geom_obj.name]['color'] = geom_obj.color.copy()
                info[geom_obj.name]['xy'] = geom_obj.pos.copy()
        info['agent'] = {}
        if hasattr(self.agent, 'pos'):
            info['agent']['xy'] = self.agent.pos.copy()
        if hasattr(self.agent, 'mat'):
            rotation_matrix = self.agent.mat
            info['agent']['yaw'] = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return info

    def calculate_reward(self):
        return 0

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        return False
