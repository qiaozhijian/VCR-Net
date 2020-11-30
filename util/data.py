#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Part of the code is referred from: https://github.com/charlesq34/pointnet

import glob
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../../dataset')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition='train', args=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../../dataset')

    if args.dataset == 'modelnet40':
        download()
        all_data = []
        all_label = []
        fileTemplate = 'ply_data_%s*.h5'
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', fileTemplate % partition)):
            f = h5py.File(h5_name, mode='r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label
    elif args.dataset == 'kitti':
        all_idx = []
        rotations = []
        translations = []
        DATA_DIR = os.path.join(DATA_DIR, 'kitti_down/h5')
        listTrain = [os.path.join(DATA_DIR, '00.h5'), os.path.join(DATA_DIR, '03.h5'), os.path.join(DATA_DIR, '05.h5'),
                     os.path.join(DATA_DIR, '07.h5'), os.path.join(DATA_DIR, '10.h5')]
        listTest = [os.path.join(DATA_DIR, '02.h5'), os.path.join(DATA_DIR, '04.h5'), os.path.join(DATA_DIR, '06.h5'),
                    os.path.join(DATA_DIR, '08.h5'), os.path.join(DATA_DIR, '09.h5')]
        if partition == 'train':
            for h5_name in listTrain:
                f = h5py.File(h5_name, mode='r')
                idx_train = f['idx_train'][::3].astype('int32')
                rotations_train = f['rotations_train'][::3].astype('float32')
                translations_train = f['translations_train'][::3].astype('float32')
                f.close()
                all_idx.append(idx_train)
                rotations.append(rotations_train)
                translations.append(translations_train)
            all_idx = np.concatenate(all_idx, axis=0)
            rotations = np.concatenate(rotations, axis=0)
            translations = np.concatenate(translations, axis=0)
            return all_idx, rotations, translations
        else:
            for h5_name in listTest:
                f = h5py.File(h5_name, mode='r')
                idx_odo = f['idx_odo'][:].astype('int32')
                rotations_odo = f['rotations_odo'][:].astype('float32')
                translations_odo = f['translations_odo'][:].astype('float32')
                f.close()
                all_idx.append(idx_odo)
                rotations.append(rotations_odo)
                translations.append(translations_odo)
            all_idx = np.concatenate(all_idx, axis=0)
            rotations = np.concatenate(rotations, axis=0)
            translations = np.concatenate(translations, axis=0)
            return all_idx, rotations, translations


def Normalization():
    data = 0


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def getPointCloud(seqN, binNum, binNumNext, R_ab=None, translation_ab=None, num_points=None):
    if seqN < 11:
        path = './data/kitti_down/bin/' + str(
            seqN).zfill(2) + '/velodyne/' + str(binNum).zfill(6) + '.bin'
        pointcloud = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])[:, 0:3];
        points = pointcloud.shape[0]
        supply_idx = points // 6

        if points < num_points:
            point_supply = np.tile(pointcloud[supply_idx, :], (num_points - points, 1))
            pointcloud1 = np.concatenate((pointcloud, point_supply), axis=0)
        else:
            pointcloud1 = pointcloud[:num_points]

        path = './data/kitti_down/bin/' + str(
            seqN).zfill(2) + '/velodyne/' + str(binNumNext).zfill(6) + '.bin'
        pointcloud = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])[:, 0:3];
        points = pointcloud.shape[0]
        if points < num_points:

            point_supply = np.tile(pointcloud1[supply_idx, :], (num_points - points, 1))
            point_supply = np.matmul(R_ab, point_supply.T) + translation_ab.reshape([3, 1])
            pointcloud2 = np.concatenate((pointcloud, point_supply.T), axis=0)
        else:
            pointcloud2 = pointcloud[:num_points]

        return pointcloud1.T, pointcloud2.T
    else:
        seqN = seqN - 11
        path = './data/kitti_down/bin/' + str(
            seqN).zfill(2) + '/velodyne/' + str(binNum).zfill(6) + '.bin'
        pointcloud = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])[:, 0:3];
        points = pointcloud.shape[0]
        supply_idx = points // 6
        if points < num_points:
            point_supply = np.tile(pointcloud[supply_idx, :], (num_points - points, 1))
            pointcloud1 = np.concatenate((pointcloud, point_supply), axis=0)
        else:
            pointcloud1 = pointcloud[:num_points]
        return pointcloud1


class KITTI(Dataset):
    def __init__(self, args, partition='train'):
        self.reserve = args.reserve
        self.num_points = args.num_points
        self.partition = partition
        self.gaussian_noise = args.gaussian_noise
        self.partial = args.partial
        print('Load KITTI Dataset')
        self.all_idx, self.rotations, self.translations = load_data(partition, args)

    def __getitem__(self, item):
        pointcloud = getPointCloud(self.all_idx[item, 0] + 11, self.all_idx[item, 1], self.all_idx[item, 2],
                                   num_points=int(self.num_points / self.reserve) + 1)
        zoom = True
        if zoom:
            pointcloud = pointcloud / 30.0

        if self.partition != 'train':
            np.random.seed(item)

        anglex = (np.random.uniform() - 0.5) * 2 * 5.0 / 180.0 * np.pi
        angley = (np.random.uniform() - 0.5) * 2 * 5.0 / 180.0 * np.pi
        anglez = (np.random.uniform() - 0.5) * 2 * 30.0 / 180.0 * np.pi

        cosx = np.cos(anglex);
        cosy = np.cos(angley);
        cosz = np.cos(anglez)
        sinx = np.sin(anglex);
        siny = np.sin(angley);
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T

        if zoom:
            translation_ab = np.array(
                [np.random.uniform(-5.0, 5.0) / 30.0, np.random.uniform(-5.0, 5.0) / 30.0,
                 np.random.uniform(-1.0, 1.0) / 30.0])
        else:
            translation_ab = np.array(
                [np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0), np.random.uniform(-1.0, 1.0)])

        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = np.random.permutation(pointcloud).T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        if self.partial:
            pointcloud1 = nearest_neighbor(pointcloud1, self.reserve)
        pointcloud1 = pointcloud1[:, :self.num_points]
        pointcloud1 = np.random.permutation(pointcloud1.T).T

        if self.partial:
            pointcloud2 = nearest_neighbor(pointcloud2, self.reserve)
        pointcloud2 = pointcloud2[:, :self.num_points]
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32'), 0

    def __len__(self):
        return self.all_idx.shape[0]


class ModelNet40(Dataset):
    def __init__(self, args, partition='train'):
        self.dataset = args.dataset
        self.num_points = args.num_points
        self.partition = partition
        self.reserve = args.reserve
        self.gaussian_noise = args.gaussian_noise
        self.model = args.model
        self.factor = args.factor
        self.partial = args.partial
        print('Load ModelNet40 Dataset')
        self.data, self.label = load_data(partition, args)
        self.label = self.label.squeeze()
        self.unseen = args.unseen
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        label = 0

        # [num,num_dim]
        pointcloud = self.data[item]

        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)

        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T

        if self.partition == 'train':
            translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                       np.random.uniform(-0.5, 0.5)])
        else:
            translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                       np.random.uniform(-0.5, 0.5)])

        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = ((np.random.permutation(pointcloud))[: self.num_points]).T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])

        euler_ba = -euler_ab[::-1]

        if not self.model == 'lpd':
            pointcloud1 = np.random.permutation(pointcloud1.T).T
            if self.partial:
                pointcloud1 = nearest_neighbor(pointcloud1, self.reserve)
            pointcloud2 = np.random.permutation(pointcloud2.T).T
            if self.partial:
                pointcloud2 = nearest_neighbor(pointcloud2, self.reserve)
        else:
            # [3,num_points]
            pointcloud = np.concatenate((pointcloud1, pointcloud2), axis=0)
            pointcloud = np.random.permutation(pointcloud.T).T
            pointcloud1 = pointcloud[0:3, :]
            pointcloud2 = pointcloud[3:6, :]

        # [3,num_points]
        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32'), label

    def __len__(self):
        return self.data.shape[0]


def nearest_neighbor(dst, reserve):
    dst = dst.T
    num = np.max([dst.shape[0], dst.shape[1]])
    num = int(num * reserve)
    src = dst[-1, :].reshape(1, -1)
    neigh = NearestNeighbors(n_neighbors=num)
    neigh.fit(dst)
    indices = neigh.kneighbors(src, return_distance=False)
    indices = indices.ravel()
    return dst[indices, :].T


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break
