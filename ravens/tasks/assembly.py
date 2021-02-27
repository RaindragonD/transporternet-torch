# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

#!/usr/bin/env python
"""Insertion Tasks."""

import cv2
import numpy as np
import pybullet as p
from ravens import utils
from ravens.tasks.task import Task

class TwoDimAssembly(Task):
  """Insertion Task - Base Variant."""

  def __init__(self):
    super().__init__()
    self.max_steps = 2

  def reset(self, env):
    super().reset(env)
    block_id = self.add_block(env)
    fixture_id, targ_pose = self.add_fixture(env)
    self.fixture_id = fixture_id
    # self.goals.append(
    #     ([block_id], [2 * np.pi], [[0]], [targ_pose], 'pose', None, 1.))
    # objs, matches, targs, _, _, metric, params, max_reward
    self.goals.append((
                        [(block_id, (0, None))], # objs
                        np.int32([[1]]),         # matches  
                        [targ_pose],             # targs
                        False, True, 'pose', None, 1 # replace, rotations, metric, params, max_reward
                                                     # reorient = (objID)
                      ))

  def add_block(self, env):
    """Add block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'assets/assembly/block_circ.urdf'
    pose = self.get_random_pose(env, size)
    startOrientation = p.getQuaternionFromEuler([np.pi/2,0,0])
    return env.add_object(urdf, (pose[0], startOrientation))

  def add_fixture(self, env):
    """Add fixture to place block."""
    size = (0.1, 0.1, 0.24)
    urdf = 'assets/assembly/main_block_circ.urdf'
    pose = self.get_random_pose(env, size)
    startOrientation = p.getQuaternionFromEuler([np.pi/2*3,0,0])
    fixture_id = env.add_object(urdf, (pose[0], startOrientation), 'rigid')
    return fixture_id, pose

  # def get_random_pose(self, env, obj_size):
  #   """Get random collision-free object pose within workspace bounds."""

  #   # Get erosion size of object in pixels.
  #   max_size = np.sqrt(obj_size[0]**2 + obj_size[1]**2)
  #   erode_size = int(np.round(max_size / self.pix_size))

  #   _, hmap, obj_mask = self.get_true_image(env)

  #   # Randomly sample an object pose within free-space pixels.
  #   free = np.ones(obj_mask.shape, dtype=np.uint8)
  #   for obj_ids in env.obj_ids.values():
  #     for obj_id in obj_ids:
  #       free[obj_mask == obj_id] = 0
  #   free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
  #   free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
  #   if np.sum(free) == 0:
  #     return
  #   pix = utils.sample_distribution(np.float32(free))
  #   pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
  #   pos = (pos[0], pos[1], obj_size[2] / 2)
  #   theta = np.random.rand() * 2 * np.pi
  #   rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
  #   return pos, rot