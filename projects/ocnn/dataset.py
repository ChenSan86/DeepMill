# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import numpy as np
import ocnn
from ocnn.octree import Octree, Points


__all__ = ['Transform', 'CollateBatch']
classes = __all__


class Transform:
  r''' A boilerplate class which transforms an data_ data_2.0 for :obj:`ocnn`.
  The data_ data_2.0 is first converted to :class:`Points`, then randomly transformed
  (if enabled), and converted to an :class:`Octree`.

  Args:
    depth (int): The octree depth.
    full_depth (int): The octree layers with a depth small than
        :attr:`full_depth` are forced to be full.
    distort (bool): If true, performs the data_2.0 augmentation.
    angle (list): A list of 3 float values to generate random rotation angles.
    interval (list): A list of 3 float values to represent the interval of
        rotation angles.
    scale (float): The maximum relative scale factor.
    uniform (bool): If true, performs uniform scaling.
    jittor (float): The maximum jitter values.
    orient_normal (str): Orient point normals along the specified axis, which is
        useful when normals are not oriented.
  '''

  def __init__(self, depth: int, full_depth: int, distort: bool, angle: list,
               interval: list, scale: float, uniform: bool, jitter: float,
               flip: list, orient_normal: str = '', **kwargs):
    super().__init__()

    # for octree building
    self.depth = depth
    self.full_depth = full_depth

    # for data_2.0 augmentation
    self.distort = distort
    self.angle = angle
    self.interval = interval
    self.scale = scale
    self.uniform = uniform
    self.jitter = jitter
    self.flip = flip

    # for other transformations
    self.orient_normal = orient_normal

  def __call__(self, sample: dict, idx: int):
    r''''''

    output = self.preprocess(sample, idx)
    output = self.transform(output, idx)
    output['octree'] = self.points2octree(output['points'])
    return output

  def preprocess(self, sample: dict, idx: int):
    r''' Transforms :attr:`sample` to :class:`Points` and performs some specific
    transformations, like normalization.
    '''

    xyz = torch.from_numpy(sample.pop('points'))
    normals = torch.from_numpy(sample.pop('normals'))
    sample['points'] = Points(xyz, normals)
    return sample

  def transform(self, sample: dict, idx: int):
    r''' Applies the general transformations provided by :obj:`ocnn`.
    '''

    # The augmentations including rotation, scaling, and jittering.
    points = sample['points']
    if self.distort:
      rng_angle, rng_scale, rng_jitter, rnd_flip = self.rnd_parameters()
      points.flip(rnd_flip)
      points.rotate(rng_angle)
      points.translate(rng_jitter)
      points.scale(rng_scale)

    if self.orient_normal:
      points.orient_normal(self.orient_normal)

    # !!! NOTE: Clip the point cloud to [-1, 1] before building the octree
    inbox_mask = points.clip(min=-1, max=1)
    sample.update({'points': points, 'inbox_mask': inbox_mask})
    return sample

  def points2octree(self, points: Points):
    r''' Converts the data_ :attr:`points` to an octree.
    '''

    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def rnd_parameters(self):
    r''' Generates random parameters for data_2.0 augmentation.
    '''

    rnd_angle = [None] * 3
    for i in range(3):
      rot_num = self.angle[i] // self.interval[i]
      rnd = torch.randint(low=-rot_num, high=rot_num+1, size=(1,))
      rnd_angle[i] = rnd * self.interval[i] * (3.14159265 / 180.0)
    rnd_angle = torch.cat(rnd_angle)

    rnd_scale = torch.rand(3) * (2 * self.scale) - self.scale + 1.0
    if self.uniform:
      rnd_scale[1] = rnd_scale[0]
      rnd_scale[2] = rnd_scale[0]

    rnd_flip = ''
    for i, c in enumerate('xyz'):
      if torch.rand([1]) < self.flip[i]:
        rnd_flip = rnd_flip + c

    rnd_jitter = torch.rand(3) * (2 * self.jitter) - self.jitter
    return rnd_angle, rnd_scale, rnd_jitter, rnd_flip


class CollateBatch:
  r''' Merge a list of octrees and points into a batch.
  '''

  def __init__(self, merge_points: bool = False):
    self.merge_points = merge_points

  def __call__(self, batch: list):
    assert type(batch) == list

    '''
    [{'points': <ocnn.octree.points.Points object at 0x0000020AC2F02D70>, 'inbox_mask': tensor([True, True, True,  ..., True, True, True]), 'octree': <ocnn.octree.octree.Octree object at 0x0000020AC317C8E0>, 'label': 0, 'filename': 'models/00180129_2a7cd9e552c6cea9a96c9b19_trimesh_005_collision_detection.ply', 'labels': array([ 0.804703, -0.563992, -0.18538 , -0.563992, -0.628736, -0.535354],
      dtype=float32), 'tool_params': ['-0.185380', '-0.563992', '-0.628736', '-0.535354']}, {'points': <ocnn.octree.points.Points object at 0x0000020AC2EC3D00>, 'inbox_mask': tensor([True, True, True,  ..., True, True, True]), 'octree': <ocnn.octree.octree.Octree object at 0x0000020A96A18E50>, 'label': 0, 'filename': 'models/00182021_3f32e263c256059ceb93f2cd_trimesh_000_collision_detection.ply', 'labels': array([ 0.97788 , -0.091459, -0.188112, -0.091459,  0.62185 , -0.777778],
      dtype=float32), 'tool_params': ['-0.188112', '-0.091459', '0.621850', '-0.777778']}, {'points': <ocnn.octree.points.Points object at 0x0000020AC2F03FD0>, 'inbox_mask': tensor([True, True, True,  ..., True, True, True]), 'octree': <ocnn.octree.octree.Octree object at 0x0000020AC2F02F50>, 'label': 0, 'filename': 'models/00185383_c75913d57b38f133ce422607_trimesh_009_collision_detection.ply', 'labels': array([ 1.,  0.,  0.,  0.,  0., -1.], dtype=float32), 'tool_params': ['0.000000', '0.000000', '0.000000', '-1.000000']}, {'points': <ocnn.octree.points.Points object at 0x0000020AC2F038E0>, 'inbox_mask': tensor([True, True, True,  ..., True, True, True]), 'octree': <ocnn.octree.octree.Octree object at 0x0000020AC2F035B0>, 'label': 0, 'filename': 'models/00182517_88c54095b2b852ca2a297d19_trimesh_001_collision_detection.ply', 'labels': array([ 1.,  0.,  0.,  0.,  0., -1.], dtype=float32), 'tool_params': ['0.000000', '0.000000', '0.000000', '-1.000000']}, {'points': <ocnn.octree.points.Points object at 0x0000020AC2F038B0>, 'inbox_mask': tensor([False, False, False,  ...,  True,  True,  True]), 'octree': <ocnn.octree.octree.Octree object at 0x0000020AC2F03460>, 'label': 0, 'filename': 'models/00184740_58d40608c3b3d0f9b3eeec2e_trimesh_003_collision_detection.ply', 'labels': array([ 0.986939, -0.029145, -0.158438, -0.029145,  0.934967, -0.353535],
      dtype=float32), 'tool_params': ['-0.158438', '-0.029145', '0.934967', '-0.353535']}, {'points': <ocnn.octree.points.Points object at 0x0000020AC2F03670>, 'inbox_mask': tensor([True, True, True,  ..., True, True, True]), 'octree': <ocnn.octree.octree.Octree object at 0x0000020AC2F02BF0>, 'label': 0, 'filename': 'models/00183359_0d8a3d7f49a277dc4b41a15a_trimesh_001_collision_detection.ply', 'labels': array([1., 0., 0., 0., 0., 1.], dtype=float32), 'tool_params': ['0.000000', '0.000000', '0.000000', '1.000000']}, {'points': <ocnn.octree.points.Points object at 0x0000020AC2F03EE0>, 'inbox_mask': tensor([ True,  True,  True,  ...,  True,  True, False]), 'octree': <ocnn.octree.octree.Octree object at 0x0000020AC317C580>, 'label': 0, 'filename': 'models/00182901_576b71a8e4b0cc2e7f45caf5_trimesh_003_collision_detection.ply', 'labels': array([ 1.,  0.,  0.,  0.,  0., -1.], dtype=float32), 'tool_params': ['0.000000', '0.000000', '0.000000', '-1.000000']}, {'points': <ocnn.octree.points.Points object at 0x0000020AC317CBB0>, 'inbox_mask': tensor([True, True, True,  ..., True, True, True]), 'octree': <ocnn.octree.octree.Octree object at 0x0000020AC317D8D0>, 'label': 0, 'filename': 'models/00181071_2c803360484b532e34aed9a1_trimesh_001_collision_detection.ply', 'labels': array([-0.394907,  0.347998, -0.850262,  0.347998,  0.913182,  0.212121],
      dtype=float32), 'tool_params': ['-0.850262', '0.347998', '0.913182', '0.212121']}]'''
    outputs = {}
    for key in batch[0].keys():
      outputs[key] = [b[key] for b in batch]

      # Merge a batch of octrees into one super octree
      if 'octree' in key:
        octree = ocnn.octree.merge_octrees(outputs[key])
        # NOTE: remember to construct the neighbor indices
        octree.construct_all_neigh()
        outputs[key] = octree

      # Merge a batch of points
      if 'points' in key and self.merge_points:
        outputs[key] = ocnn.octree.merge_points(outputs[key])

      # Convert the labels to a Tensor
      if 'label' == key:
        outputs['label'] = torch.tensor(outputs[key])

      if 'label_2' in key:
        outputs['label_2'] = torch.tensor(outputs[key])


      if  'labels' == key:
        arr = np.asarray(outputs[key])
        outputs['labels'] = torch.from_numpy(arr).to(torch.float32)




    return outputs
'''{'points': <ocnn.octree.points.Points object at 0x00000276CD562470>, 'inbox_mask': [tensor([True, True, True,  ..., True, True, True])], 'octree': <ocnn.octree.octree.Octree object at 0x00000276CD562530>, 'label': tensor([0]), 'filename': ['models/00188527_4c51c8d28bbc8e3e63b05da1_trimesh_035_collision_detection.ply'], 'tool_params': [['0.387790', '0.382705', '0.092861', '-0.919192']]}
'''
