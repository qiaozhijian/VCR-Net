#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model.lpdnet_model import LPDNet
from model.transformer import Transformer
from util.util import quat2mat, transform_point_cloud, npmat2euler, get_graph_feature


class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()  # [b,3,num]
        x = get_graph_feature(x)  # [b,6,num,20]

        x = F.relu(self.bn1(self.conv1(x)))  # [b,64,num,20]
        x1 = x.max(dim=-1, keepdim=True)[0]  # [b,64,num,1]

        x = F.relu(self.bn2(self.conv2(x)))  # [b,64,num,20]
        x2 = x.max(dim=-1, keepdim=True)[0]  # [b,64,num,1]

        x = F.relu(self.bn3(self.conv3(x)))  # [b,128,num,20]
        x3 = x.max(dim=-1, keepdim=True)[0]  # [b,128,num,1]

        x = F.relu(self.bn4(self.conv4(x)))  # [b,256,num,20]
        x4 = x.max(dim=-1, keepdim=True)[0]  # [b,256,num,1]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # [b,512,num,1]

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)  # [b,512,num]
        return x


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.use_mFea = args.use_mFea

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]

        if self.use_mFea:
            src = input[2].transpose(2, 1).split([3, 5], dim=2)[0].transpose(2, 1)
            tgt = input[3].transpose(2, 1).split([3, 5], dim=2)[0].transpose(2, 1)
        else:
            src = input[2]
            tgt = input[3]

        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

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
        return R, t.view(batch_size, 3), src, src_corr


class DCP(nn.Module):
    def __init__(self, args):
        super(DCP, self).__init__()
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif args.emb_nn == 'lpdnet':
            self.emb_nn = LPDNet(args)
        else:
            raise Exception('Not implemented')

        if args.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':
            self.pointer = Transformer(args=args)
        else:
            raise Exception("Not implemented")

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')

    def forward(self, *input):
        # [B,3,num]
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        rotation_ab, translation_ab, src, src_corr = self.head(src_embedding, tgt_embedding, src, tgt)
        if self.cycle:
            rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src)
        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, src, src_corr


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
            rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred, src, src_corr = net(src,
                                                                                                              target)

            # save(src,src_corr, rotation_ba, translation_ba, rotation_ab_pred, translation_ab_pred)

            if args.use_mFea:
                src = src.transpose(2, 1).split([3, 5], dim=2)[0].transpose(2, 1)
                target = target.transpose(2, 1).split([3, 5], dim=2)[0].transpose(2, 1)

            ## save rotation and translation
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.numpy())
            ##
            rotations_ba.append(rotation_ba.detach().cpu().numpy())
            translations_ba.append(translation_ba.detach().cpu().numpy())
            rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
            translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
            eulers_ba.append(euler_ba.numpy())

            transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

            # savePlt(args,target,transformed_src,name='corrpose')

            transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

            # if args.
            # visualization_test(args,src, target,transformed_target,folderName='visualization')

            ###########################
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)

            if args.loss == 'pose':
                loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)
            elif args.loss == 'point':
                loss = torch.nn.functional.mse_loss(transformed_src, src_corr)

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


def train_one_epoch(args, net, train_loader, opt):
    net.train()

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

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
            train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred, src, src_corr = net(src, target)

        if args.use_mFea:
            src = src.transpose(2, 1).split([3, 5], dim=2)[0].transpose(2, 1)
            target = target.transpose(2, 1).split([3, 5], dim=2)[0].transpose(2, 1)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
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
            loss = torch.nn.functional.mse_loss(transformed_src, src_corr)

        if args.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                        translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                           + translation_ba_pred) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        loss.backward()
        opt.step()
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


def testDCP(args, net, test_loader, boardio, textio):
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
    textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab,
                     test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    if args.cycle:
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (-1, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                         test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))

    K = 5
    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    loss = np.sum((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2, axis=-1)
    idx = np.argsort(loss, axis=-1, kind='quicksort', order=None)
    idTopKRot = idx[-K:]

    loss = np.sum((test_translations_ab - test_translations_ab_pred) ** 2, axis=-1)
    idx = np.argsort(loss, axis=-1, kind='quicksort', order=None)
    idTopKTrans = idx[-K:]

    # showBad(args, net, test_loader, idTopKRot, idTopKTrans)

    return test_loss


def showBad(args, net, test_loader, idTopKRot, idTopKTrans):
    net.eval()
    idTopKRotPair = np.zeros((idTopKRot.shape[0], 2), dtype=np.int32)
    idTopKTransPair = np.zeros((idTopKTrans.shape[0], 2), dtype=np.int32)
    for i in range(idTopKRot.shape[0]):
        idTopKRotPair[i, 0] = idTopKRot[i] // args.test_batch_size
        idTopKRotPair[i, 1] = idTopKRot[i] % args.test_batch_size
        idTopKTransPair[i, 0] = idTopKTrans[i] // args.test_batch_size
        idTopKTransPair[i, 1] = idTopKTrans[i] % args.test_batch_size

    batch_id = -1
    with torch.no_grad():
        for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
                test_loader):
            src = src.cuda()
            target = target.cuda()
            # rotation_ab = rotation_ab.cuda()
            # translation_ab = translation_ab.cuda()
            batch_id = batch_id + 1
            if (batch_id in idTopKRotPair[:, 0]) or (batch_id in idTopKTransPair[:, 0]):

                rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

                transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

                idR = idTopKRotPair[idTopKRotPair[:, 0] == batch_id, 1]
                idt = idTopKTransPair[idTopKTransPair[:, 0] == batch_id, 1]

                rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
                translation_ab_pred = translation_ab_pred.detach().cpu().numpy()
                euler_ab = euler_ab.detach().cpu().numpy()
                translation_ab = translation_ab.detach().cpu().numpy()

                for id_R in idR:
                    r = Rotation.from_dcm(rotation_ab_pred[id_R, :, :])
                    eulers_ab_pred = r.as_euler('zyx', degrees=True)
                    euler_ab_i = euler_ab[id_R, :]
                    euler_delta = eulers_ab_pred - np.degrees(euler_ab_i)
                    mae = np.mean(np.abs(euler_delta))
                    title = 'euler error: ' + str(euler_delta) + ' mae: ' + str(mae)
                    filename = str(batch_id * args.test_batch_size + id_R) + '-R'
                    # savePlt(args, target[id_R,:,:], transformed_src[id_R,:,:], folderName='bad_cp', title=title,filename=filename)

                for id_t in idt:
                    translation_ab_pred_i = translation_ab_pred[id_t, :]
                    translation_ab_i = translation_ab[id_t, :]
                    translation_delta = translation_ab_pred_i - translation_ab_i
                    mae = np.mean(np.abs(translation_delta))
                    title = 'trans error: ' + str(translation_delta) + ' mae: ' + str(mae)
                    filename = str(batch_id * args.test_batch_size + id_t) + '-T'
                    # savePlt(args, target[id_t,:,:], transformed_src[id_t,:,:], folderName='bad_cp', title=title,filename=filename)


def trainDCP(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=False, threshold=0.000001)

    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    best_test_mse_ba = np.inf
    best_test_rmse_ba = np.inf
    best_test_mae_ba = np.inf

    best_test_r_mse_ba = np.inf
    best_test_r_rmse_ba = np.inf
    best_test_r_mae_ba = np.inf
    best_test_t_mse_ba = np.inf
    best_test_t_rmse_ba = np.inf
    best_test_t_mae_ba = np.inf

    for epoch in range(args.epochs):
        # scheduler.step()
        train_loss, train_cycle_loss, \
        train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, \
        train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
        train_translations_ba_pred, train_eulers_ab, train_eulers_ba = train_one_epoch(args, net, train_loader, opt)

        test_loss, test_cycle_loss, \
        test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
        test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader)

        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)

        train_rmse_ba = np.sqrt(train_mse_ba)
        test_rmse_ba = np.sqrt(test_mse_ba)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        train_rotations_ba_pred_euler = npmat2euler(train_rotations_ba_pred, 'xyz')
        train_r_mse_ba = np.mean((train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)) ** 2)
        train_r_rmse_ba = np.sqrt(train_r_mse_ba)
        train_r_mae_ba = np.mean(np.abs(train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)))
        train_t_mse_ba = np.mean((train_translations_ba - train_translations_ba_pred) ** 2)
        train_t_rmse_ba = np.sqrt(train_t_mse_ba)
        train_t_mae_ba = np.mean(np.abs(train_translations_ba - train_translations_ba_pred))

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

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_cycle_loss = test_cycle_loss

            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab

            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            best_test_mse_ba = test_mse_ba
            best_test_rmse_ba = test_rmse_ba
            best_test_mae_ba = test_mae_ba

            best_test_r_mse_ba = test_r_mse_ba
            best_test_r_rmse_ba = test_r_rmse_ba
            best_test_r_mae_ba = test_r_mae_ba

            best_test_t_mse_ba = test_t_mse_ba
            best_test_t_rmse_ba = test_t_rmse_ba
            best_test_t_mae_ba = test_t_mae_ba

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        scheduler.step(best_test_loss)
        lr = opt.param_groups[0]['lr']
        if lr <= 0.0000011:
            break
        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint(
            'EPOCH:: %d, Loss: %f, Cycle Loss:, %f,lr: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
            'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
            % (epoch, train_loss, train_cycle_loss, lr, train_mse_ab, train_rmse_ab, train_mae_ab, train_r_mse_ab,
               train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))
        if args.cycle:
            textio.cprint('B--------->A')
            textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                          'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                          % (
                              epoch, train_loss, train_mse_ba, train_rmse_ba, train_mae_ba, train_r_mse_ba,
                              train_r_rmse_ba,
                              train_r_mae_ba, train_t_mse_ba, train_t_rmse_ba, train_t_mae_ba))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab, test_r_mse_ab,
                         test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
        if args.cycle:
            textio.cprint('B--------->A')
            textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                          'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                          % (epoch, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                             test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, best_test_loss, best_test_cycle_loss, best_test_mse_ab, best_test_rmse_ab,
                         best_test_mae_ab, best_test_r_mse_ab, best_test_r_rmse_ab,
                         best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))
        if args.cycle:
            textio.cprint('B--------->A')
            textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                          'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                          % (epoch, best_test_loss, best_test_mse_ba, best_test_rmse_ba, best_test_mae_ba,
                             best_test_r_mse_ba, best_test_r_rmse_ba,
                             best_test_r_mae_ba, best_test_t_mse_ba, best_test_t_rmse_ba, best_test_t_mae_ba))

        boardio.add_scalar('A->B/train/loss', train_loss, epoch)
        boardio.add_scalar('A->B/train/MSE', train_mse_ab, epoch)
        boardio.add_scalar('A->B/train/RMSE', train_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/MAE', train_mae_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MSE', train_t_mse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MAE', train_t_mae_ab, epoch)

        boardio.add_scalar('B->A/train/loss', train_loss, epoch)
        boardio.add_scalar('B->A/train/MSE', train_mse_ba, epoch)
        boardio.add_scalar('B->A/train/RMSE', train_rmse_ba, epoch)
        boardio.add_scalar('B->A/train/MAE', train_mae_ba, epoch)
        boardio.add_scalar('B->A/train/rotation/MSE', train_r_mse_ba, epoch)
        boardio.add_scalar('B->A/train/rotation/RMSE', train_r_rmse_ba, epoch)
        boardio.add_scalar('B->A/train/rotation/MAE', train_r_mae_ba, epoch)
        boardio.add_scalar('B->A/train/translation/MSE', train_t_mse_ba, epoch)
        boardio.add_scalar('B->A/train/translation/RMSE', train_t_rmse_ba, epoch)
        boardio.add_scalar('B->A/train/translation/MAE', train_t_mae_ba, epoch)

        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss, epoch)
        boardio.add_scalar('A->B/test/MSE', test_mse_ab, epoch)
        boardio.add_scalar('A->B/test/RMSE', test_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/MAE', test_mae_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MSE', test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)

        boardio.add_scalar('B->A/test/loss', test_loss, epoch)
        boardio.add_scalar('B->A/test/MSE', test_mse_ba, epoch)
        boardio.add_scalar('B->A/test/RMSE', test_rmse_ba, epoch)
        boardio.add_scalar('B->A/test/MAE', test_mae_ba, epoch)
        boardio.add_scalar('B->A/test/rotation/MSE', test_r_mse_ba, epoch)
        boardio.add_scalar('B->A/test/rotation/RMSE', test_r_rmse_ba, epoch)
        boardio.add_scalar('B->A/test/rotation/MAE', test_r_mae_ba, epoch)
        boardio.add_scalar('B->A/test/translation/MSE', test_t_mse_ba, epoch)
        boardio.add_scalar('B->A/test/translation/RMSE', test_t_rmse_ba, epoch)
        boardio.add_scalar('B->A/test/translation/MAE', test_t_mae_ba, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/MSE', best_test_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/RMSE', best_test_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/MAE', best_test_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

        boardio.add_scalar('B->A/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('B->A/best_test/MSE', best_test_mse_ba, epoch)
        boardio.add_scalar('B->A/best_test/RMSE', best_test_rmse_ba, epoch)
        boardio.add_scalar('B->A/best_test/MAE', best_test_mae_ba, epoch)
        boardio.add_scalar('B->A/best_test/rotation/MSE', best_test_r_mse_ba, epoch)
        boardio.add_scalar('B->A/best_test/rotation/RMSE', best_test_r_rmse_ba, epoch)
        boardio.add_scalar('B->A/best_test/rotation/MAE', best_test_r_mae_ba, epoch)
        boardio.add_scalar('B->A/best_test/translation/MSE', best_test_t_mse_ba, epoch)
        boardio.add_scalar('B->A/best_test/translation/RMSE', best_test_t_rmse_ba, epoch)
        boardio.add_scalar('B->A/best_test/translation/MAE', best_test_t_mae_ba, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()
