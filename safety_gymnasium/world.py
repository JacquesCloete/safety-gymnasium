# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""World."""

from __future__ import annotations

import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

import mujoco
import numpy as np
import xmltodict
import yaml

import safety_gymnasium
from safety_gymnasium.utils.common_utils import build_xml_from_dict, convert, rot2quat
from safety_gymnasium.utils.task_utils import get_body_xvelp


# Default location to look for xmls folder:
BASE_DIR = os.path.dirname(safety_gymnasium.__file__)


@dataclass
class Engine:
    """Physical engine."""

    # pylint: disable=no-member
    model: mujoco.MjModel = None
    data: mujoco.MjData = None

    def update(self, model, data):
        """Set engine."""
        self.model = model
        self.data = data


class World:  # pylint: disable=too-many-instance-attributes
    """This class starts mujoco simulation.

    And contains some apis for interacting with mujoco."""

    # Default configuration (this should not be nested since it gets copied)
    # *NOTE:* Changes to this configuration should also be reflected in `Builder` configuration
    DEFAULT: ClassVar[dict[str, Any]] = {
        'agent_base': 'assets/xmls/car.xml',  # Which agent XML to use as the base
        'agent_xy': np.zeros(2),  # agent XY location
        'agent_rot': 0,  # agent rotation about Z axis
        'floor_size': [3.5, 3.5, 0.1],  # Used for displaying the floor
        # FreeGeoms -- this is processed and added by the Builder class
        'free_geoms': {},  # map from name -> object dict
        # Geoms -- similar to objects, but they are immovable and fixed in the scene.
        'geoms': {},  # map from name -> geom dict
        # Mocaps -- mocap objects which are used to control other objects
        'mocaps': {},
        'floor_type': 'mat',
        'task_name': None,
    }

    def __init__(self, agent, obstacles, config=None) -> None:
        """config - JSON string or dict of configuration.  See self.parse()"""
        if config:
            self.parse(config)  # Parse configuration

        self.first_reset = True

        self._agent = agent  # pylint: disable=no-member
        self._obstacles = obstacles

        self.agent_base_path = None
        self.agent_base_xml = None
        self.xml = None
        self.xml_string = None

        self.engine = Engine()
        self.bind_engine()

    def parse(self, config):
        """Parse a config dict - see self.DEFAULT for description."""
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    def bind_engine(self):
        """Send the new engine instance to the agent and obstacles."""
        self._agent.set_engine(self.engine)
        for obstacle in self._obstacles:
            obstacle.set_engine(self.engine)

    def build(self):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        """Build a world, including generating XML and moving objects."""
        # Read in the base XML (contains agent, camera, floor, etc)
        self.agent_base_path = os.path.join(BASE_DIR, self.agent_base)  # pylint: disable=no-member
        with open(self.agent_base_path, encoding='utf-8') as f:  # pylint: disable=invalid-name
            self.agent_base_xml = f.read()
        self.xml = xmltodict.parse(self.agent_base_xml)  # Nested OrderedDict objects
        if self.task_name in ['FormulaOne']:  # pylint: disable=no-member
            self.xml['mujoco']['option']['@integrator'] = 'RK4'
            self.xml['mujoco']['option']['@timestep'] = '0.004'

        if 'compiler' not in self.xml['mujoco']:
            compiler = xmltodict.parse(
                f"""<compiler
                angle="radian"
                meshdir="{BASE_DIR}/assets/meshes"
                texturedir="{BASE_DIR}/assets/textures"
                />""",
            )
            self.xml['mujoco']['compiler'] = compiler['compiler']
        else:
            self.xml['mujoco']['compiler'].update(
                {
                    '@angle': 'radian',
                    '@meshdir': os.path.join(BASE_DIR, 'assets', 'meshes'),
                    '@texturedir': os.path.join(BASE_DIR, 'assets', 'textures'),
                },
            )

        # Convenience accessor for xml dictionary
        worldbody = self.xml['mujoco']['worldbody']

        # Move agent position to starting position
        worldbody['body']['@pos'] = convert(
            # pylint: disable-next=no-member
            np.r_[self.agent_xy, self._agent.z_height],
        )
        worldbody['body']['@quat'] = convert(rot2quat(self.agent_rot))  # pylint: disable=no-member

        # We need this because xmltodict skips over single-item lists in the tree
        worldbody['body'] = [worldbody['body']]
        if 'geom' in worldbody:
            worldbody['geom'] = [worldbody['geom']]
        else:
            worldbody['geom'] = []
        # Add equality section if missing
        if 'equality' not in self.xml['mujoco']:
            self.xml['mujoco']['equality'] = OrderedDict()
        equality = self.xml['mujoco']['equality']
        if 'weld' not in equality:
            equality['weld'] = []

        # Add asset section if missing
        if 'asset' not in self.xml['mujoco']:
            self.xml['mujoco']['asset'] = {}
        if 'texture' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['texture'] = []
        if 'material' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['material'] = []
        if 'mesh' not in self.xml['mujoco']['asset']:
            self.xml['mujoco']['asset']['mesh'] = []
        material = self.xml['mujoco']['asset']['material']
        texture = self.xml['mujoco']['asset']['texture']
        mesh = self.xml['mujoco']['asset']['mesh']

        # load all assets config from .yaml file
        with open(os.path.join(BASE_DIR, 'configs/assets.yaml'), encoding='utf-8') as file:
            assets_config = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506

        texture.append(assets_config['textures']['skybox'])

        if self.floor_type == 'mat':  # pylint: disable=no-member
            texture.append(assets_config['textures']['matplane'])
            material.append(assets_config['materials']['matplane'])
        elif self.floor_type == 'village':  # pylint: disable=no-member
            texture.append(assets_config['textures']['village_floor'])
            material.append(assets_config['materials']['village_floor'])
        elif self.floor_type == 'mud':  # pylint: disable=no-member
            texture.append(assets_config['textures']['mud_floor'])
            material.append(assets_config['materials']['mud_floor'])
        elif self.floor_type == 'none':  # pylint: disable=no-member
            self.floor_size = [1e-9, 1e-9, 0.1]  # pylint: disable=attribute-defined-outside-init
        else:
            raise NotImplementedError

        selected_textures = {}
        selected_materials = {}
        selected_meshes = {}
        for config in (
            # pylint: disable=no-member
            list(self.geoms.values())
            + list(self.free_geoms.values())
            + list(self.mocaps.values())
            # pylint: enable=no-member
        ):
            if 'type' not in config:
                for geom in config['geoms']:
                    if geom['type'] != 'mesh':
                        continue
                    mesh_name = geom['mesh']
                    if mesh_name in assets_config['textures']:
                        selected_textures[mesh_name] = assets_config['textures'][mesh_name]
                        selected_materials[mesh_name] = assets_config['materials'][mesh_name]
                    selected_meshes[mesh_name] = assets_config['meshes'][mesh_name]
            elif config['type'] == 'mesh':
                mesh_name = config['mesh']
                if mesh_name in assets_config['textures']:
                    selected_textures[mesh_name] = assets_config['textures'][mesh_name]
                    selected_materials[mesh_name] = assets_config['materials'][mesh_name]
                selected_meshes[mesh_name] = assets_config['meshes'][mesh_name]
        texture += selected_textures.values()
        material += selected_materials.values()
        mesh += selected_meshes.values()

        # Add light to the XML dictionary
        light = xmltodict.parse(
            """<b>
            <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true"
                exponent="1" pos="0 0 0.5" specular="0 0 0" castshadow="false"/>
            </b>""",
        )
        worldbody['light'] = light['b']['light']

        # Add floor to the XML dictionary if missing
        if not any(g.get('@name') == 'floor' for g in worldbody['geom']):
            floor = xmltodict.parse(
                """
                <geom name="floor" type="plane" condim="6"/>
                """,
            )
            worldbody['geom'].append(floor['geom'])

        # Make sure floor renders the same for every world
        for g in worldbody['geom']:  # pylint: disable=invalid-name
            if g['@name'] == 'floor':
                g.update(
                    {
                        '@size': convert(self.floor_size),  # pylint: disable=no-member
                        '@rgba': '1 1 1 1',
                    },
                )
                if self.floor_type == 'mat':  # pylint: disable=no-member
                    g.update({'@material': 'matplane'})
                elif self.floor_type == 'village':  # pylint: disable=no-member
                    g.update({'@material': 'village_floor'})
                elif self.floor_type == 'mud':  # pylint: disable=no-member
                    g.update({'@material': 'mud_floor'})
                elif self.floor_type == 'none':  # pylint: disable=no-member
                    pass
                else:
                    raise NotImplementedError
        # Add cameras to the XML dictionary
        cameras = xmltodict.parse(
            """<b>
            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
            <camera name="fixedfar++" pos="0 -10 10" zaxis="0 -1 1"/>
            <camera name="topdownnear" pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="45"/>
            <camera name="topdownfar" pos="0 0 6" xyaxes="1 0 0 0 1 0" fovy="45"/>
            <camera name="topdownfar++" pos="0 0 9" xyaxes="1 0 0 0 1 0" fovy="45"/>
            </b>""",
        )
        worldbody['camera'] = cameras['b']['camera']

        # Build and add a tracking camera (logic needed to ensure orientation correct)
        theta = self.agent_rot + np.pi  # pylint: disable=no-member
        xyaxes = {
            'x1': np.cos(theta),
            'x2': -np.sin(theta),
            'x3': 0,
            'y1': np.sin(theta),
            'y2': np.cos(theta),
            'y3': 1,
        }
        pos = {
            'xp': 0 * np.cos(theta) + (-2) * np.sin(theta),
            'yp': 0 * (-np.sin(theta)) + (-2) * np.cos(theta),
            'zp': 2,
        }
        track_camera = xmltodict.parse(
            """<b>
            <camera name="track" mode="track" pos="{xp} {yp} {zp}"
                xyaxes="{x1} {x2} {x3} {y1} {y2} {y3}"/>
            </b>""".format(
                **pos,
                **xyaxes,
            ),
        )
        if 'camera' in worldbody['body'][0]:
            if isinstance(worldbody['body'][0]['camera'], list):
                worldbody['body'][0]['camera'] = worldbody['body'][0]['camera'] + [
                    track_camera['b']['camera'],
                ]
            else:
                worldbody['body'][0]['camera'] = [
                    worldbody['body'][0]['camera'],
                    track_camera['b']['camera'],
                ]
        else:
            worldbody['body'][0]['camera'] = [
                track_camera['b']['camera'],
            ]

        # Add free_geoms to the XML dictionary
        for name, object in self.free_geoms.items():  # pylint: disable=redefined-builtin, no-member
            assert object['name'] == name, f'Inconsistent {name} {object}'
            object = object.copy()  # don't modify original object
            object['freejoint'] = object['name']
            if name == 'push_box':
                object['quat'] = rot2quat(object.pop('rot'))
                dim = object['geoms'][0]['size'][0]
                object['geoms'][0]['dim'] = dim
                object['geoms'][0]['width'] = dim / 2
                object['geoms'][0]['x'] = dim
                object['geoms'][0]['y'] = dim
                # pylint: disable-next=consider-using-f-string
                collision_xml = """
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                        <geom name="col1" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} {y} 0"/>
                        <geom name="col2" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} {y} 0"/>
                        <geom name="col3" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} -{y} 0"/>
                        <geom name="col4" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} -{y} 0"/>
                        """.format(
                    **{k: convert(v) for k, v in object['geoms'][0].items()},
                )
                if len(object['geoms']) == 2:
                    # pylint: disable-next=consider-using-f-string
                    visual_xml = """
                        <geom name="{name}" type="mesh" mesh="{mesh}" material="{material}" pos="{pos}"
                        rgba="1 1 1 1" group="{group}" contype="{contype}" conaffinity="{conaffinity}" density="{density}"
                        euler="{euler}"/>
                    """.format(
                        **{k: convert(v) for k, v in object['geoms'][1].items()},
                    )
                else:
                    visual_xml = """"""
                body = xmltodict.parse(
                    # pylint: disable-next=consider-using-f-string
                    f"""
                    <body name="{object['name']}" pos="{convert(object['pos'])}" quat="{convert(object['quat'])}">
                        {collision_xml}
                        {visual_xml}
                    </body>
                """,
                )
            else:
                if object['geoms'][0]['type'] == 'mesh':
                    object['geoms'][0]['condim'] = 6
                object['quat'] = rot2quat(object.pop('rot'))
                body = build_xml_from_dict(object)
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])
        # Add mocaps to the XML dictionary
        for name, mocap in self.mocaps.items():  # pylint: disable=no-member
            # Mocap names are suffixed with 'mocap'
            assert mocap['name'] == name, f'Inconsistent {name}'
            assert (
                name.replace('mocap', 'obj') in self.free_geoms  # pylint: disable=no-member
            ), f'missing object for {name}'  # pylint: disable=no-member
            # Add the object to the world
            mocap = mocap.copy()  # don't modify original object
            mocap['quat'] = rot2quat(mocap.pop('rot'))
            mocap['mocap'] = 'true'
            mocap['geoms'][0]['contype'] = 0
            mocap['geoms'][0]['conaffinity'] = 0
            mocap['geoms'][0]['pos'] = mocap.pop('pos')
            body = build_xml_from_dict(mocap)
            worldbody['body'].append(body['body'])
            # Add weld to equality list
            mocap['body1'] = name
            mocap['body2'] = name.replace('mocap', 'obj')
            weld = xmltodict.parse(
                # pylint: disable-next=consider-using-f-string
                """
                <weld name="{name}" body1="{body1}" body2="{body2}" solref=".02 1.5"/>
            """.format(
                    **{k: convert(v) for k, v in mocap.items()},
                ),
            )
            equality['weld'].append(weld['weld'])
        # Add geoms to XML dictionary
        for name, geom in self.geoms.items():  # pylint: disable=no-member
            assert geom['name'] == name, f'Inconsistent {name} {geom}'
            geom = geom.copy()  # don't modify original object
            for item in geom['geoms']:
                if 'contype' not in item:
                    item['contype'] = item.get('contype', 1)
                if 'conaffinity' not in item:
                    item['conaffinity'] = item.get('conaffinity', 1)
            if 'rot' in geom:
                geom['quat'] = rot2quat(geom.pop('rot'))
            body = build_xml_from_dict(geom)
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])

        # Instantiate simulator
        # print(xmltodict.unparse(self.xml, pretty=True))
        self.xml_string = xmltodict.unparse(self.xml)
        model = mujoco.MjModel.from_xml_string(self.xml_string)  # pylint: disable=no-member
        data = mujoco.MjData(model)  # pylint: disable=no-member

        # Recompute simulation intrinsics from new position
        mujoco.mj_forward(model, data)  # pylint: disable=no-member
        self.engine.update(model, data)

        print_debug = False
        if print_debug:
            # DEBUG PRINT: Actual positions after build
            print("\n[DEBUG World.build()] Actual positions after build & mj_forward:")
            agent_main_body_name = self.xml['mujoco']['worldbody']['body'][0].get('@name')
            if agent_main_body_name:  # Check if agent_main_body_name was determined
                try:
                    # data.body().xpos gives the world frame position of the body's origin.
                    # This should be correct for both free joint and slide/hinge agents as implemented.
                    actual_agent_pos = self.data.body(agent_main_body_name).xpos.copy()
                    actual_agent_quat = self.data.body(agent_main_body_name).xquat.copy()
                    # MuJoCo quaternions are (w, x, y, z)
                    # Yaw (rotation around Z-axis) can be calculated from quaternion
                    # Using the formula: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
                    w, x, y, z = actual_agent_quat
                    actual_agent_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
                    print(
                        f"  Agent '{agent_main_body_name}' actual pos (body xpos): {actual_agent_pos}, rot: {actual_agent_yaw}"
                    )
                except Exception as e:
                    print(
                        f"  Agent '{agent_main_body_name}': Error getting actual xpos/xquat for debug: {e}"
                    )
            else:
                print("  Agent: Main body name not determined for agent position debug print.")

            # For zones and other static geoms, iterate self.geoms which contains model body names as keys
            # self.geoms is populated from world_config_dict['geoms'] via self.parse()
            if hasattr(self, 'geoms') and isinstance(self.geoms, dict):
                for (
                    model_body_name_key
                ) in self.geoms.keys():  # These are the names of bodies in the model
                    try:
                        # We expect mj_name2id to find the body since model_body_name_key is a key from self.geoms,
                        # which was used in the main loop of fast_rebuild to update model.body_pos.
                        body_id_debug = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_BODY, model_body_name_key
                        )
                        if body_id_debug != -1:
                            actual_pos = self.model.body_pos[body_id_debug].copy()

                            print(f"  Geom '{model_body_name_key}' actual pos: {actual_pos}")
                        else:
                            # This case should ideally not occur if model_body_name_key is valid from self.geoms
                            print(
                                f"  Static Geom body '{model_body_name_key}' (key from self.geoms) NOT FOUND in model by mj_name2id. This is unexpected."
                            )
                    except Exception as e:
                        print(
                            f"  Error getting actual pos for static geom body '{model_body_name_key}': {e}"
                        )
            else:
                print(
                    "  self.geoms (dict of static geoms) not found or not a dict in World object for debug print."
                )

            print("[DEBUG World.build()] End of actual positions print.")
            # End DEBUG PRINT

    def rebuild(self, config=None, state=True, fast_rebuild=False):
        """Build a new sim from a model if the model changed."""
        if state:
            old_state = self.get_state()

        if config:
            self.parse(config)
        if fast_rebuild:
            self.fast_rebuild()
        else:
            self.build()
        if state:
            self.set_state(old_state)
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def fast_rebuild(self):
        """Rebuild the sim by modifying the existing model and data parameters
        based on the current self.config. This avoids recompiling the XML.
        Assumes self.config has been updated via self.parse() if a new config was provided.
        """
        if self.model is None or self.data is None:
            raise RuntimeError(
                'Fast rebuild called before model/data initialized. Perform a full build first.',
            )

        print_debug = False

        # Reset kinematic state to a clean slate
        # model.qpos0 reflects the initial state from the last full XML compilation.
        self.data.qpos[:] = np.copy(self.model.qpos0)
        self.data.qvel[:] = 0.0
        if self.model.na > 0:  # Number of actuators
            self.data.act[:] = 0.0

        # 1. Update Agent's initial pose from the current configuration
        agent_placed_by_fast_rebuild = False
        agent_main_body_name = None

        # Try to get the agent's main body name from the parsed XML.
        # self.xml is populated by the build() method.
        if (
            self.xml
            and 'mujoco' in self.xml
            and 'worldbody' in self.xml['mujoco']
            and isinstance(self.xml['mujoco']['worldbody'].get('body'), list)
            and len(self.xml['mujoco']['worldbody']['body']) > 0
        ):
            agent_main_body_name = self.xml['mujoco']['worldbody']['body'][0].get('@name')

        if agent_main_body_name:
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, agent_main_body_name
                )
                if body_id != -1:
                    # Strategy 1: Try to find and use a root FREE joint
                    for j_id in range(self.model.njnt):
                        if (
                            self.model.jnt_bodyid[j_id] == body_id
                            and self.model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_FREE
                        ):

                            qpos_adr = self.model.jnt_qposadr[j_id]
                            new_agent_pos_3d = np.r_[self.agent_xy, self._agent.z_height]
                            new_agent_quat = rot2quat(self.agent_rot)

                            self.data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(
                                new_agent_pos_3d, dtype=np.float64
                            )
                            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(
                                new_agent_quat, dtype=np.float64
                            )
                            agent_placed_by_fast_rebuild = True
                            break  # Found and used the free joint

                    # Strategy 2: If no free joint was used, try specific slide/hinge joints (for Point-like agents)
                    if not agent_placed_by_fast_rebuild:
                        joint_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "x")
                        joint_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "y")
                        joint_z_id = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, "z"
                        )  # Hinge for rotation

                        if (
                            joint_x_id != -1
                            and self.model.jnt_bodyid[joint_x_id] == body_id
                            and self.model.jnt_type[joint_x_id] == mujoco.mjtJoint.mjJNT_SLIDE
                            and joint_y_id != -1
                            and self.model.jnt_bodyid[joint_y_id] == body_id
                            and self.model.jnt_type[joint_y_id] == mujoco.mjtJoint.mjJNT_SLIDE
                            and joint_z_id != -1
                            and self.model.jnt_bodyid[joint_z_id] == body_id
                            and self.model.jnt_type[joint_z_id] == mujoco.mjtJoint.mjJNT_HINGE
                        ):
                            if print_debug:
                                print(
                                    f"\n[DEBUG World.fast_rebuild() Point Agent] Target self.agent_xy: {self.agent_xy}, self.agent_rot: {self.agent_rot}"
                                )
                                print(
                                    f"[DEBUG World.fast_rebuild() Point Agent] Agent body_id: {body_id}, target z_height: {self._agent.z_height}"
                                )

                            # 1. Directly set the agent body's world position and orientation in the model.
                            self.model.body_pos[body_id] = np.r_[
                                self.agent_xy, self._agent.z_height
                            ].astype(np.float64)
                            self.model.body_quat[body_id] = rot2quat(self.agent_rot).astype(
                                np.float64
                            )
                            if print_debug:
                                print(
                                    f"[DEBUG World.fast_rebuild() Point Agent] Set model.body_pos[{body_id}] to: {self.model.body_pos[body_id]}"
                                )
                                print(
                                    f"[DEBUG World.fast_rebuild() Point Agent] Set model.body_quat[{body_id}] to: {self.model.body_quat[body_id]}"
                                )

                            # 2. Zero out the qpos for the agent's local slide/hinge joints,
                            #    as their effect is now directly in the body's pose.
                            qpos_adr_x = self.model.jnt_qposadr[joint_x_id]
                            qpos_adr_y = self.model.jnt_qposadr[joint_y_id]
                            qpos_adr_z_rot = self.model.jnt_qposadr[joint_z_id]

                            self.data.qpos[qpos_adr_x] = 0.0
                            self.data.qpos[qpos_adr_y] = 0.0
                            self.data.qpos[qpos_adr_z_rot] = 0.0

                            if print_debug:
                                print(
                                    f"[DEBUG World.fast_rebuild() Point Agent] Set data.qpos for x,y,z joints ({qpos_adr_x},{qpos_adr_y},{qpos_adr_z_rot}) to 0.0"
                                )

                            agent_placed_by_fast_rebuild = True
            except Exception as e:
                print(f"[DEBUG World.fast_rebuild()] Exception during agent placement: {e}")
                pass

        if not agent_placed_by_fast_rebuild:
            warning_message = (
                "Warning: Agent may not be at the new randomized position during fast_rebuild. "
                "Its pose will be based on the initial state from the last full build (qpos0). "
            )
            if agent_main_body_name:
                warning_message += f"Attempted to find a root free joint or specific x/y/z joints for body '{agent_main_body_name}'. "
            else:
                warning_message += (
                    "Could not determine agent's main body name for specific joint lookup. "
                )
            warning_message += "This is expected if the agent does not use a recognized root joint mechanism (free joint or x/y slide + z hinge)."
            print(warning_message)

        # 2. Update Geoms (fixed bodies)
        # self.geoms is updated by self.parse()
        for name, geom_group_config in self.geoms.items():
            try:
                body_id = self.model.body(name).id

                if 'pos' in geom_group_config:
                    self.model.body_pos[body_id] = np.asarray(
                        geom_group_config['pos'], dtype=np.float64
                    )
                current_rot = geom_group_config.get('rot')
                current_quat = geom_group_config.get('quat')
                if current_rot is not None:
                    self.model.body_quat[body_id] = np.asarray(
                        rot2quat(current_rot), dtype=np.float64
                    )
                elif current_quat is not None:
                    self.model.body_quat[body_id] = np.asarray(current_quat, dtype=np.float64)

                for geom_item_config in geom_group_config.get('geoms', []):
                    geom_name = geom_item_config['name']
                    geom_id = self.model.geom(geom_name).id
                    if 'rgba' in geom_item_config:
                        self.model.geom_rgba[geom_id] = np.asarray(
                            geom_item_config['rgba'], dtype=float
                        )
                    if 'size' in geom_item_config:
                        config_size_arr = np.asarray(
                            geom_item_config['size'], dtype=np.float64
                        ).flatten()
                        geom_type = self.model.geom_type[geom_id]

                        final_size_for_model = np.zeros(3, dtype=np.float64)

                        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                            if config_size_arr.size >= 1:
                                final_size_for_model[0] = config_size_arr[0]  # radius
                        elif (
                            geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE
                            or geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER
                        ):
                            if config_size_arr.size >= 1:
                                final_size_for_model[0] = config_size_arr[0]  # radius
                            if config_size_arr.size >= 2:
                                final_size_for_model[1] = config_size_arr[1]  # half-height
                        elif config_size_arr.size == 3:  # For BOX, ELLIPSOID, PLANE, MESH (scaling)
                            final_size_for_model[:] = config_size_arr[:]
                        elif config_size_arr.size == 1 and (
                            geom_type == mujoco.mjtGeom.mjGEOM_BOX
                            or geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
                            or geom_type == mujoco.mjtGeom.mjGEOM_MESH
                        ):
                            # Assume uniform scaling if one size param provided
                            final_size_for_model[:] = config_size_arr[0]
                        else:  # Fallback: copy available elements, pad with zeros
                            len_to_copy = min(config_size_arr.size, 3)
                            final_size_for_model[:len_to_copy] = config_size_arr[:len_to_copy]

                        self.model.geom_size[geom_id] = final_size_for_model
            except KeyError:
                # Silently ignore if a configured geom is not in the current model
                # (e.g. if base XML changed and fast_rebuild is attempted)
                pass

        # 3. Update FreeGeoms (movable objects with freejoints)
        # self.free_geoms is updated by self.parse()
        for name, free_geom_config in self.free_geoms.items():
            try:
                joint_name = free_geom_config.get('freejoint', name)
                joint_id = self.model.joint(joint_name).id

                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    if 'pos' in free_geom_config:
                        self.data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(
                            free_geom_config['pos'], dtype=np.float64
                        )
                    current_rot = free_geom_config.get('rot')
                    current_quat = free_geom_config.get('quat')
                    if current_rot is not None:
                        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(
                            rot2quat(current_rot), dtype=np.float64
                        )
                    elif current_quat is not None:
                        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray(
                            current_quat, dtype=np.float64
                        )

                for geom_item_config in free_geom_config.get('geoms', []):
                    geom_name = geom_item_config['name']
                    geom_id = self.model.geom(geom_name).id
                    if 'rgba' in geom_item_config:
                        self.model.geom_rgba[geom_id] = np.asarray(
                            geom_item_config['rgba'], dtype=float
                        )
                    if (
                        'size' in geom_item_config
                    ):  # This block replaces the original self.model.geom_size assignment
                        config_size_arr = np.asarray(
                            geom_item_config['size'], dtype=np.float64
                        ).flatten()
                        geom_type = self.model.geom_type[geom_id]

                        final_size_for_model = np.zeros(3, dtype=np.float64)

                        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                            if config_size_arr.size >= 1:
                                final_size_for_model[0] = config_size_arr[0]  # radius
                        elif (
                            geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE
                            or geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER
                        ):
                            # XML: size="radius half-height" (2 params)
                            # mjModel.geom_size: [radius, half-height, 0.0] (3 params)
                            if config_size_arr.size >= 1:
                                final_size_for_model[0] = config_size_arr[0]  # radius
                            if config_size_arr.size >= 2:
                                final_size_for_model[1] = config_size_arr[1]  # half-height
                        elif config_size_arr.size == 3:  # For BOX, ELLIPSOID, PLANE, MESH (scaling)
                            final_size_for_model[:] = config_size_arr[:]
                        elif config_size_arr.size == 1 and (
                            geom_type == mujoco.mjtGeom.mjGEOM_BOX
                            or geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
                            or geom_type == mujoco.mjtGeom.mjGEOM_MESH
                        ):
                            # Assume uniform scaling if one size param provided
                            final_size_for_model[:] = config_size_arr[0]
                        else:  # Fallback: copy available elements, pad with zeros
                            len_to_copy = min(config_size_arr.size, 3)
                            final_size_for_model[:len_to_copy] = config_size_arr[:len_to_copy]

                        self.model.geom_size[geom_id] = final_size_for_model
            except KeyError:
                pass

        # 4. Update Mocaps
        # self.mocaps is updated by self.parse()
        for name, mocap_config in self.mocaps.items():
            try:
                body_id = self.model.body(name).id
                if self.model.body_mocapid[body_id] != -1:
                    mocap_id = self.model.body_mocapid[body_id]
                    if 'pos' in mocap_config:
                        self.data.mocap_pos[mocap_id] = np.asarray(
                            mocap_config['pos'], dtype=np.float64
                        )
                    current_rot = mocap_config.get('rot')
                    current_quat = mocap_config.get('quat')
                    if current_rot is not None:
                        self.data.mocap_quat[mocap_id] = np.asarray(
                            rot2quat(current_rot), dtype=np.float64
                        )
                    elif current_quat is not None:
                        self.data.mocap_quat[mocap_id] = np.asarray(current_quat, dtype=np.float64)

                    for geom_item_config in mocap_config.get('geoms', []):
                        geom_name = geom_item_config['name']
                        geom_id = self.model.geom(geom_name).id
                        if 'rgba' in geom_item_config:
                            self.model.geom_rgba[geom_id] = np.asarray(
                                geom_item_config['rgba'], dtype=float
                            )
                        if 'size' in geom_item_config:
                            config_size_arr = np.asarray(
                                geom_item_config['size'], dtype=np.float64
                            ).flatten()
                            geom_type = self.model.geom_type[geom_id]

                            final_size_for_model = np.zeros(3, dtype=np.float64)

                            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                                if config_size_arr.size >= 1:
                                    final_size_for_model[0] = config_size_arr[0]  # radius
                            elif (
                                geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE
                                or geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER
                            ):
                                if config_size_arr.size >= 1:
                                    final_size_for_model[0] = config_size_arr[0]  # radius
                                if config_size_arr.size >= 2:
                                    final_size_for_model[1] = config_size_arr[1]  # half-height
                            elif (
                                config_size_arr.size == 3
                            ):  # For BOX, ELLIPSOID, PLANE, MESH (scaling)
                                final_size_for_model[:] = config_size_arr[:]
                            elif config_size_arr.size == 1 and (
                                geom_type == mujoco.mjtGeom.mjGEOM_BOX
                                or geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
                                or geom_type == mujoco.mjtGeom.mjGEOM_MESH
                            ):
                                # Assume uniform scaling if one size param provided
                                final_size_for_model[:] = config_size_arr[0]
                            else:  # Fallback: copy available elements, pad with zeros
                                len_to_copy = min(config_size_arr.size, 3)
                                final_size_for_model[:len_to_copy] = config_size_arr[:len_to_copy]

                            self.model.geom_size[geom_id] = final_size_for_model
            except KeyError:
                pass

        # 5. Floor properties (limited update: size only)
        # self.floor_size is updated by self.parse()
        try:
            floor_geom_id = self.model.geom('floor').id
            self.model.geom_size[floor_geom_id] = np.asarray(self.floor_size, dtype=np.float64)
        except KeyError:
            pass

        # Recompute simulation intrinsics from new position and other changes
        mujoco.mj_forward(self.model, self.data)
        self.engine.update(self.model, self.data)

        if print_debug:
            # DEBUG PRINT: Actual positions after fast_rebuild
            print(
                "\n[DEBUG World.fast_rebuild()] Actual positions after fast_rebuild & mj_forward:"
            )
            if agent_main_body_name:  # Check if agent_main_body_name was determined
                try:
                    # data.body().xpos gives the world frame position of the body's origin.
                    # This should be correct for both free joint and slide/hinge agents as implemented.
                    actual_agent_pos = self.data.body(agent_main_body_name).xpos.copy()
                    actual_agent_quat = self.data.body(agent_main_body_name).xquat.copy()
                    # MuJoCo quaternions are (w, x, y, z)
                    # Yaw (rotation around Z-axis) can be calculated from quaternion
                    # Using the formula: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
                    w, x, y, z = actual_agent_quat
                    actual_agent_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
                    print(
                        f"  Agent '{agent_main_body_name}' actual pos (body xpos): {actual_agent_pos}, rot: {actual_agent_yaw}"
                    )
                except Exception as e:
                    print(
                        f"  Agent '{agent_main_body_name}': Error getting actual xpos/xquat for debug: {e}"
                    )
            else:
                print("  Agent: Main body name not determined for agent position debug print.")

            # For zones and other static geoms, iterate self.geoms which contains model body names as keys
            # self.geoms is populated from world_config_dict['geoms'] via self.parse()
            if hasattr(self, 'geoms') and isinstance(self.geoms, dict):
                for (
                    model_body_name_key
                ) in self.geoms.keys():  # These are the names of bodies in the model
                    try:
                        # We expect mj_name2id to find the body since model_body_name_key is a key from self.geoms,
                        # which was used in the main loop of fast_rebuild to update model.body_pos.
                        body_id_debug = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_BODY, model_body_name_key
                        )
                        if body_id_debug != -1:
                            actual_pos = self.model.body_pos[body_id_debug].copy()

                            print(f"  Geom '{model_body_name_key}' actual pos: {actual_pos}")
                        else:
                            # This case should ideally not occur if model_body_name_key is valid from self.geoms
                            print(
                                f"  Static Geom body '{model_body_name_key}' (key from self.geoms) NOT FOUND in model by mj_name2id. This is unexpected."
                            )
                    except Exception as e:
                        print(
                            f"  Error getting actual pos for static geom body '{model_body_name_key}': {e}"
                        )
            else:
                print(
                    "  self.geoms (dict of static geoms) not found or not a dict in World object for debug print."
                )

            print("[DEBUG World.fast_rebuild()] End of actual positions print.")
            # End DEBUG PRINT

    def reset(self, build=True):
        """Reset the world. (sim is accessed through self.sim)"""
        if build:
            self.build()

    def body_com(self, name):
        """Get the center of mass of a named body in the simulator world reference frame."""
        return self.data.body(name).subtree_com.copy()

    def body_pos(self, name):
        """Get the position of a named body in the simulator world reference frame."""
        return self.data.body(name).xpos.copy()

    def body_mat(self, name):
        """Get the rotation matrix of a named body in the simulator world reference frame."""
        return self.data.body(name).xmat.copy().reshape(3, -1)

    def body_vel(self, name):
        """Get the velocity of a named body in the simulator world reference frame."""
        return get_body_xvelp(self.model, self.data, name).copy()

    def get_state(self):
        """Returns a copy of the simulator state."""
        state = {
            'time': np.copy(self.data.time),
            'qpos': np.copy(self.data.qpos),
            'qvel': np.copy(self.data.qvel),
        }
        if self.model.na == 0:
            state['act'] = None
        else:
            state['act'] = np.copy(self.data.act)

        return state

    def set_state(self, value):
        """
        Sets the state from an dict.

        Args:
        - value (dict): the desired state.
        - call_forward: optionally call sim.forward(). Called by default if
            the udd_callback is set.
        """
        self.data.time = value['time']
        self.data.qpos[:] = np.copy(value['qpos'])
        self.data.qvel[:] = np.copy(value['qvel'])
        if self.model.na != 0:
            self.data.act[:] = np.copy(value['act'])

    @property
    def model(self):
        """Access model easily."""
        return self.engine.model

    @property
    def data(self):
        """Access data easily."""
        return self.engine.data
