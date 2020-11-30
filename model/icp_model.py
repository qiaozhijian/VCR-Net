#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from util.util import transform_point_cloud, npmat2euler


class ICP(nn.Module):
    def __init__(self, max_iterations=10, tolerance=0.001):
        super(ICP, self).__init__()

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    # [B,dims,num]
    def forward(self, srcInit, dst):
        icp_start = time()
        src = srcInit
        prev_error = 0
        for i in range(self.max_iterations):
            # find the nearest neighbors between the current source and destination points
            mean_error, src_corr = self.nearest_neighbor(src, dst)
            # compute the transformation between the current source and nearest destination points
            rotation_ab, translation_ab = self.best_fit_transform(src, src_corr)
            src = transform_point_cloud(src, rotation_ab, translation_ab)

            if torch.abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        rotation_ab, translation_ab = self.best_fit_transform(srcInit, src)

        rotation_ba = rotation_ab.transpose(2, 1).contiguous()
        translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

        print("icp: ", time() - icp_start)
        return srcInit, src, rotation_ab, translation_ab, rotation_ba, translation_ba

    def nearest_neighbor(self, src, dst):

        batch_size = src.size(0)
        num_points = src.size(2)

        inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), dst)
        xx = torch.sum(src ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(dst ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        pairwise_distance = pairwise_distance - yy

        idx = pairwise_distance.topk(k=1, dim=-1)[1]
        val = pairwise_distance.topk(k=1, dim=-1)[0]

        idx_base = torch.arange(0, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) \
                       .view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        dst = dst.transpose(2, 1).contiguous()
        candidates = dst.view(batch_size * num_points, -1)[idx, :]
        candidates = candidates.view(batch_size, num_points, 1, 3).squeeze(-2)  # (batch_size,num, tgtK, 3)

        return val.mean(), candidates.transpose(2, 1).contiguous()

    def best_fit_transform(self, src, src_corr):

        batch_size = src.size(0)

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())

            R.append(r)
            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def getDateset(batch_size=8, gaussian_noise=False, angle=4, num_points=512, dims=3, tTld=0.5):
    pointcloud1All = []
    pointcloud2All = []
    R_abAll = []
    translation_abAll = []

    for b in range(batch_size):
        pointcloud = np.random.rand(num_points, dims) - 0.5

        if gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)

        np.random.seed(b)
        anglex = np.random.uniform() * angle / 180.0 * np.pi
        angley = np.random.uniform() * angle / 180.0 * np.pi
        anglez = np.random.uniform() * angle / 180.0 * np.pi

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

        translation_ab = np.array([np.random.uniform(-tTld, tTld), np.random.uniform(-tTld, tTld),
                                   np.random.uniform(-tTld, tTld)])

        pointcloud1 = pointcloud.T  # [3,num]

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        pointcloud1All.append(pointcloud1)
        pointcloud2All.append(pointcloud2)
        R_abAll.append(R_ab)
        translation_abAll.append(translation_ab)

    pointcloud1All = np.asarray(pointcloud1All).reshape(batch_size, dims, num_points)
    pointcloud2All = np.asarray(pointcloud2All).reshape(batch_size, dims, num_points)
    R_abAll = np.asarray(R_abAll).reshape(batch_size, 3, 3)
    translation_abAll = np.asarray(translation_abAll).reshape(batch_size, 3)

    # [3,num_points]
    return pointcloud1All.astype('float32'), pointcloud2All.astype('float32'), R_abAll.astype(
        'float32'), translation_abAll.astype('float32')


def test_one_epoch(args, net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    with torch.no_grad():
        for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
                test_loader):
            src = src.cuda()
            target = target.cuda()
            rotation_ab = rotation_ab.cuda()
            translation_ab = translation_ab.cuda()
            rotation_ba = rotation_ba.cuda()
            translation_ba = translation_ba.cuda()

            batch_size = src.size(0)
            num_examples += batch_size
            srcInit, src, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src,
                                                                                                             target)

            if args.use_mFea:
                src = src.transpose(2, 1).split([3, 5], dim=2)[0].transpose(2, 1)
                target = target.transpose(2, 1).split([3, 5], dim=2)[0].transpose(2, 1)

            ## save rotation and translation
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.numpy())

            rotations_ba.append(rotation_ba.detach().cpu().numpy())
            translations_ba.append(translation_ba.detach().cpu().numpy())
            rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
            translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
            eulers_ba.append(euler_ba.numpy())

            transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

            transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

            ###########################
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
            if args.loss == 'pose':
                loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)
            elif args.loss == 'point':
                loss = torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2])
            else:
                lossPose = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                           + F.mse_loss(translation_ab_pred, translation_ab)
                lossPoint = torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2])
                loss = lossPose + 0.1 * lossPoint
            if args.cycle:
                rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
                translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                            translation_ab_pred.view(batch_size, 3, 1)).view(batch_size,
                                                                                                             3)
                                               + translation_ba_pred) ** 2, dim=[0, 1])
                cycle_loss = rotation_loss + translation_loss

                loss = loss + cycle_loss * 0.1

            total_loss += loss.item() * batch_size

            if args.cycle:
                total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

            mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

            mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba


def testICP(args, net, test_loader, boardio, textio):
    test_loss, test_cycle_loss, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)
    test_rmse_ba = np.sqrt(test_mse_ba)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

    test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
    test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)) ** 2)
    test_r_rmse_ba = np.sqrt(test_r_mse_ba)
    test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
    test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
    test_t_rmse_ba = np.sqrt(test_t_mse_ba)
    test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint(
        'EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f,  rot_MAE: %f,trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
        % (-1, test_loss, test_cycle_loss, \
           test_mse_ab, test_rmse_ab, test_mae_ab, \
           test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab, \
           test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    if args.cycle:
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (-1, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                         test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))
    return test_loss


from util.icp import icp

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Only run on GPU!")
        exit(-1)
    pointcloud1, pointcloud2, R_ab, t_ab = getDateset(batch_size=1, angle=45, tTld=0.5, num_points=512)
    net = ICP().cuda()
    src = torch.tensor(pointcloud1).cuda()
    target = torch.tensor(pointcloud2).cuda()
    rotation_ab = torch.tensor(R_ab).cuda()
    translation_ab = torch.tensor(t_ab).cuda()
    batch_size = src.size(0)

    src, src_corr, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)
    identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
    loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) + F.mse_loss(
        translation_ab_pred, translation_ab)
    print(loss)

    src_np = src.detach().cpu().numpy().squeeze()
    target_np = target.detach().cpu().numpy().squeeze()
    T, distances, iterations = icp(src_np.T, target_np.T, tolerance=0.000001)
    print(iterations)
    T1 = np.eye(4);
    T1[:3, :3] = R_ab.squeeze();
    T1[:3, 3] = t_ab
    T = torch.tensor(T, dtype=torch.float32).cuda()
    T1 = torch.tensor(T1, dtype=torch.float32).cuda()

    identity = torch.eye(3).cuda()
    loss = F.mse_loss(torch.matmul(T[:3, :3].transpose(1, 0), T1[:3, :3]), identity) + F.mse_loss(T[:3, 3], T1[:3, 3])
    print(loss)
