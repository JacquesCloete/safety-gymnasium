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

import numpy as np

from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Geom


class RGBZones(Geom):  # pylint: disable=too-many-instance-attributes
    """
    Colored zones with RGB colors that are randomized on reset
    within specified bounds. Each zone type (identified by zones_id)
    gets a unique simulation group for lidar.
    """

    def __init__(
        self,
        zones_id: int,  # Unique identifier for this type of zone
        size: float = 0.4,  # radius of zones
        num: int = 1,  # Number of physical instances of this zone type
        rgb_lower_bounds: list = [0.0, 0.0, 0.0],  # [R_low, G_low, B_low], values 0-1
        rgb_upper_bounds: list = [1.0, 1.0, 1.0],  # [R_high, G_high, B_high], values 0-1
        locations=None,  # Fixed locations to override placements
        keepout: float = 0.55,  # Radius of keepout for zone placement
        render_alpha: float = 0.5,  # Alpha for visualization (0-1)
        is_lidar_observed: bool = True,  # Whether the zone is observed by lidar
        is_constrained: bool = True,  # Whether the zone is constrained
    ):
        if not (
            len(rgb_lower_bounds) == 3
            and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in rgb_lower_bounds)
        ):
            raise ValueError("rgb_lower_bounds must be a list of 3 floats between 0 and 1.")
        if not (
            len(rgb_upper_bounds) == 3
            and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in rgb_upper_bounds)
        ):
            raise ValueError("rgb_upper_bounds must be a list of 3 floats between 0 and 1.")
        if not all(low <= high for low, high in zip(rgb_lower_bounds, rgb_upper_bounds)):
            raise ValueError(
                "rgb_lower_bounds must be less than or equal to rgb_upper_bounds for each RGB component."
            )
        self.id = zones_id
        self.name = f"rgb_{self.id}_zones"
        self.num = num
        self.size: float = size
        self.placements: list = None  # Placements list for zones (defaults to full extents)
        self.locations: list = locations if locations else []
        self.keepout: float = keepout
        self.render_alpha: float = render_alpha

        self.rgb_lower_bounds = np.array(rgb_lower_bounds, dtype=float)
        self.rgb_upper_bounds = np.array(rgb_upper_bounds, dtype=float)

        self.color: np.ndarray = np.array([0.0, 0.0, 0.0, self.render_alpha])
        # Note: color will be set in the reset method

        self.group: int = self.calculate_group()
        self.is_lidar_observed: bool = is_lidar_observed
        self.is_constrained: bool = is_constrained

    def calculate_group(self) -> int:
        max_predefined_group = max(GROUP.values())
        return max_predefined_group + self.id + 1

    def randomize_color(self, seed: int | None = None) -> None:
        """
        Sets the zone's color. Randomizes RGB values within the defined bounds.
        Updates self.color (R,G,B,A).
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
            r = rng.uniform(self.rgb_lower_bounds[0], self.rgb_upper_bounds[0])
            g = rng.uniform(self.rgb_lower_bounds[1], self.rgb_upper_bounds[1])
            b = rng.uniform(self.rgb_lower_bounds[2], self.rgb_upper_bounds[2])
        else:
            r = np.random.uniform(self.rgb_lower_bounds[0], self.rgb_upper_bounds[0])
            g = np.random.uniform(self.rgb_lower_bounds[1], self.rgb_upper_bounds[1])
            b = np.random.uniform(self.rgb_lower_bounds[2], self.rgb_upper_bounds[2])

        self.color = np.array([r, g, b, self.render_alpha])

        # Note: If the simulation is already running and this method is called,
        # the visual color of existing geoms in MuJoCo won't update automatically.
        # The environment needs to find these geoms and update sim.model.geom_rgba.
        # This is handled in the reset method of the environment.

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, 2e-2],  # self.hazards_size / 2 + 1e-2],
            'rot': rot,
            'geoms': [
                {
                    'name': self.name,
                    'size': [self.size, 1e-2],  # self.hazards_size / 2],
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': self.group,
                    'rgba': self.color.copy(),
                },
            ],
        }

    def cal_cost(self):
        cost = {f'cost_{self.name}': 0}
        for h_pos in self.pos:
            h_dist = self.agent.dist_xy(h_pos)
            if h_dist <= self.size:
                cost[f'cost_{self.name}'] = 1
        return cost

    @property
    def pos(self):
        """Helper to get the zones positions from layout."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'{self.name[:-1]}{i}').xpos.copy() for i in range(self.num)]
