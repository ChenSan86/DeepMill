# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch

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
      if 'label' in key:
        outputs['label'] = torch.tensor(outputs[key])

      if 'label_2' in key:
        outputs['label_2'] = torch.tensor(outputs[key])

    # print("\n"*10)
    # print(outputs)
    # print("\n"*10)

    return outputs
'''{'points': <ocnn.octree.points.Points object at 0x00000158DA2AD750>,
 'inbox_mask': [tensor([True, True, True,  ..., True, True, True]),
 tensor([False, False, False,  ...,  True,  True,  True]),
 tensor([True, True, True,  ..., True, True, True]),
 tensor([True, True, True,   ..., True, True, True]), 
tensor([True, True, True,  ..., True, True, True]), 
  tensor([True, True, True,  ..., True, True, True]), 
  tensor([True, True, True,  ..., True, True, True]), 
  tensor([True, True, True,  ..., True, True, True])], 
  'octree': <ocnn.octree.octree.Octre
e object at 0x00000158DA2AE080>, 
'label': tensor([0, 0, 0, 0, 0, 0, 0, 0]),
 'filename': ['models/00185844_14f3d01d62b5237f728896c5_trimesh_076_collision_detection.ply', 
 'models/00180686_ea6367c8a224ad6fdf4fb34b_trimesh_013_collision_detection.ply', 
 'models/00186429_9d497b2fa29b87491f08ef85_trimesh_003_collision_detection.ply', 
'models/00186448_19a4b1099ec6cf2b78a62985_trimesh_000_collision_detection.ply', 
'models/00180129_2a7cd9e552c6cea9a96c9b19_trimesh_005_collision_detection.ply', 
'models/00186294_57720c19e4b0b4b404503992_trimesh_001_collision_detection.ply',
 'models/00186536_35fd82bd6c05f33776aa5852_trimesh_021_collision_detection.ply',
  'models/00183156_e4dfafb620a34b2168d7e04a_trimesh_001_collision_detection.ply'], 
  'tool_params': [['1.859530', '4.404930', '94.149500', '5.689460'],
   ['1.528950', '5.992420', '60.097500', '5.337980'], 
   ['1.022710', '8.981910', '54.232700', '2.376390'], 
   ['1.908020', '4.877120', '19.339000', '0.614663'], 
   ['1.969010', '4.070060', '96.976400', '0.958661'], 
   ['1.838680', '6.335170', '52.698200', '9.206030'], 
   ['1.295500', '0.211525', '42.697500', '0.987759'], 
['1.143460', '10.010300', '53.347800', '4.771190']]}'''
