#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
from torch.autograd import Variable


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    xyz = xyz.transpose(2, 1)
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # （B, npoint）
    distance = torch.ones(B, N).to(device) * 1e10  # （B, N）

    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # batch_size 

    barycenter = torch.sum((xyz), 1)
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist, 1)[1]

    for i in range(npoint):
        print("-------------------------------------------------------")
        print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        print("dist    : ", dist)
        mask = dist < distance
        print("mask %i : %s" % (i, mask))
        distance[mask] = dist[mask]
        print("distance: ", distance)

        farthest = torch.max(distance, -1)[1]

    return centroids


if __name__ == '__main__':
    sim_data = Variable(torch.rand(1, 3, 8))
    print(sim_data)

    centroids = farthest_point_sample(sim_data, 4)

    print("Sampled pts: ", centroids)
