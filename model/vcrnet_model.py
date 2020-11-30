#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model.icp_model import ICP
from model.lpdnet_model import LPDNet
from model.transformer import Transformer, clones
from util.util import get_graph_feature, transform_point_cloud, npmat2euler, quat2mat


def vcrnetIter(net, src, tgt, iter=1):
    transformed_src = src
    bFirst = True

    for i in range(iter):
        srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(
            transformed_src, tgt)
        transformed_src = transform_point_cloud(transformed_src, rotation_ab_pred, translation_ab_pred)

        if bFirst:
            bFirst = False
            rotation_ab_pred_final = rotation_ab_pred.detach()
            translation_ab_pred_final = translation_ab_pred.detach()
        else:
            rotation_ab_pred_final = torch.matmul(rotation_ab_pred.detach(), rotation_ab_pred_final)
            translation_ab_pred_final = torch.matmul(rotation_ab_pred.detach(),
                                                     translation_ab_pred_final.unsqueeze(2)).squeeze(
                2) + translation_ab_pred.detach()

    rotation_ba_pred_final = rotation_ab_pred_final.transpose(2, 1).contiguous()
    translation_ba_pred_final = -torch.matmul(rotation_ba_pred_final, translation_ab_pred_final.unsqueeze(2)).squeeze(2)

    return srcK, src_corrK, rotation_ab_pred_final, translation_ab_pred_final, rotation_ba_pred_final, translation_ba_pred_final


def vcrnetIcpNet(args, net, src, tgt):
    icpNet = ICP(max_iterations=args.max_iterations).cuda()
    srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, tgt)

    transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

    _, _, rotation_ab_pred_icp, translation_ab_pred_icp, rotation_ba_pred_icp, translation_ba_pred_icp = icpNet(
        transformed_src, tgt)

    rotation_ab_pred = torch.matmul(rotation_ab_pred_icp, rotation_ab_pred)
    translation_ab_pred = torch.matmul(rotation_ab_pred_icp, translation_ab_pred.unsqueeze(2)).squeeze(
        2) + translation_ab_pred_icp

    rotation_ba_pred = rotation_ab_pred.transpose(2, 1).contiguous()
    translation_ba_pred = -torch.matmul(rotation_ba_pred, translation_ab_pred.unsqueeze(2)).squeeze(2)

    return transformed_src, tgt, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred


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


class VcpTopK(nn.Module):
    """Generate the VCP points based on K most similar points

    """

    def __init__(self, args):
        super(VcpTopK, self).__init__()
        self.emb_nn = args.emb_nn
        self.partial = args.partial
        self.overlap2 = args.overlap2

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]

        src = input[2]
        tgt = input[3]

        if self.partial:
            src_overlap, src_embedding_overlap, tgt_overlap, tgt_embedding_overlap, src_remain, tgt_remain = \
                self.selectCom(src, src_embedding, tgt, tgt_embedding, overlap2=self.overlap2)
            src, src_corr = self.getCopair(src_overlap, src_embedding_overlap, tgt_overlap, tgt_embedding_overlap,
                                           self.overlap2)
        else:
            src, src_corr = self.getCopairALL(src, src_embedding, tgt, tgt_embedding)

        return src, src_corr

    def selectCom(self, src, src_emb, tgt, tgt_emb, overlap2=0.75):
        """Select the points that overlap between two point cluods.

        Args:
            src: source point cloud
            src_emb: embedded source
            tgt: target
            tgt_emb: embeded target
            overlap2: ratio of points that overlap in the original point cloud.

        Returns:
            src, src_emb, tgt, tgt_emb after selection (batch_size, 3, num)
        """
        batch_size, n_dims, num_points_src = src.size()
        batch_size, n_dims, num_points_tgt = tgt.size()
        all_index_src = np.arange(batch_size * num_points_src)
        all_index_tgt = np.arange(batch_size * num_points_tgt)
        _, n_fdims, _ = src_emb.size()
        srcK = int(num_points_src * 0.84 * overlap2)
        tgtK = int(num_points_tgt * 0.84 * overlap2)

        inner = -2 * torch.matmul(src_emb.transpose(2, 1).contiguous(), tgt_emb)
        xx = torch.sum(src_emb ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(tgt_emb ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        scores = pairwise_distance - yy

        idx_base_tgt = torch.arange(0, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) \
                           .view(-1, 1, 1) * num_points_tgt

        scoresSoft = torch.softmax(scores, dim=2)  # [b,num,num]
        scoresColSum = torch.sum(scoresSoft, dim=1, keepdim=True)
        idxColSum = scoresColSum.topk(k=tgtK, dim=-1)[1]

        idxColSum = idxColSum + idx_base_tgt
        idxColSum = idxColSum.view(-1)  # (batch_size* tgtK)

        index_remain = np.setdiff1d(all_index_tgt.reshape(-1), idxColSum.detach().cpu().numpy())
        # choose src topK
        tgt = tgt.transpose(2, 1).contiguous()  # (batch_size, num, 3)
        tgt_overlap = tgt.view(batch_size * num_points_tgt, n_dims)[idxColSum, :]
        tgt_overlap = tgt_overlap.view(batch_size, tgtK, n_dims).permute(0, 2, 1)  # (batch_size, 3, num)
        tgt_remain = tgt.view(batch_size * num_points_tgt, n_dims)[index_remain, :]
        tgt_remain = tgt_remain.view(batch_size, -1, n_dims).permute(0, 2, 1)  # (batch_size, 3, num)

        # choose src topK
        tgt_emb = tgt_emb.transpose(2, 1).contiguous()  # (batch_size, num, 3)
        tgt_emb = tgt_emb.view(batch_size * num_points_tgt, n_fdims)[idxColSum, :]
        tgt_emb = tgt_emb.view(batch_size, tgtK, n_fdims).permute(0, 2, 1)  # (batch_size, 3, num)

        idx_base_src = torch.arange(0, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) \
                           .view(-1, 1, 1) * num_points_src
        scoresSoft = torch.softmax(scores, dim=1)  # [b,num,num]
        scoresRowSum = torch.sum(scoresSoft, dim=2, keepdim=True)
        idxRowSum = scoresRowSum.topk(k=srcK, dim=-2)[1]  # (batch_size, srcK, 1)

        idxRowSum = idxRowSum + idx_base_src
        idxRowSum = idxRowSum.view(-1)  # (batch_size* srcK)
        index_remain = np.setdiff1d(all_index_src, idxRowSum.detach().cpu().numpy())
        # choose src topK
        src = src.transpose(2, 1).contiguous()  # (batch_size, num, 3)
        src_overlap = src.view(batch_size * num_points_src, n_dims)[idxRowSum, :]
        src_overlap = src_overlap.view(batch_size, srcK, n_dims).permute(0, 2, 1)  # (batch_size, 3, num)
        src_remain = src.view(batch_size * num_points_src, n_dims)[index_remain, :]
        src_remain = src_remain.view(batch_size, -1, n_dims).permute(0, 2, 1)  # (batch_size, 3, num)

        # choose src topK
        src_emb = src_emb.transpose(2, 1).contiguous()  # (batch_size, num, 3)
        src_emb = src_emb.view(batch_size * num_points_src, n_fdims)[idxRowSum, :]
        src_emb = src_emb.view(batch_size, srcK, n_fdims).permute(0, 2, 1)  # (batch_size, 3, num)

        return src_overlap, src_emb, tgt_overlap, tgt_emb, src_remain, tgt_remain

    def getCopair(self, src, src_emb, tgt, tgt_emb, overlap2):
        """Get virtual corresponding point clouds

        The VCP is acquired by weighted sum of related #tgtK points

        Args:
            src: source point cloud
            src_emb: embedded source
            tgt: target
            tgt_emb: embeded target

        Returns:

        """

        batch_size, n_dims, num_src = src.size()
        batch_size, n_dims, num_tgt = tgt.size()
        _, f_dims, _ = src_emb.size()

        tgtK = 1  #
        srcK = int(num_src * 0.52 * overlap2)

        # 1). in form of softmax weights
        inner = -2 * torch.matmul(src_emb.transpose(2, 1).contiguous(), tgt_emb)
        xx = torch.sum(src_emb ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(tgt_emb ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        pairwise_distance = pairwise_distance - yy
        # for clarification, call it negative pair-wise distance would be better

        pairwise_distance = torch.softmax(pairwise_distance, dim=2)  # [b,num_src,num_tgt]

        idx = pairwise_distance.topk(k=tgtK, dim=-1)[1]
        val = pairwise_distance.topk(k=tgtK, dim=-1)[0]

        # 2). choose tgtK tgt candidates
        idx_base = torch.arange(0, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) \
                       .view(-1, 1, 1) * num_tgt
        idx = idx + idx_base
        idx = idx.view(-1)  # batch_size * num_src * tgtK
        tgt = tgt.transpose(2, 1).contiguous()
        candidates = tgt.view(batch_size * num_tgt, -1)[idx, :]
        candidates = candidates.view(batch_size, num_src, tgtK, n_dims)  # (batch_size,num_src, tgtK, 3)

        # choose tgt topK
        idx_base = torch.arange(0, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) \
                       .view(-1, 1, 1) * num_src
        idx = torch.sum(val, dim=-1, keepdim=True).topk(k=srcK, dim=-2)[1]  # (batch_size, srcK, 1)
        idx = idx + idx_base
        idx = idx.view(-1)  # (batch_size* srcK)
        tgtCandidates = candidates.view(batch_size * num_src, tgtK, n_dims)[idx, :, :]
        tgtCandidates = tgtCandidates.view(batch_size, srcK, tgtK, n_dims).permute(0, 1, 3,
                                                                                   2)  # (batch_size,srcK, 3, tgtK)

        # 3). choose val topK origin val, which is the weight in the sum (batch_size, num_src, tgtK)
        val_sum = torch.sum(val, dim=-1, keepdim=True)
        val = torch.div(val, val_sum)
        valCandidates = val.view(batch_size * num_src, tgtK)[idx, :]
        valCandidates = valCandidates.view(batch_size, srcK, tgtK).unsqueeze(-1)  # (batch_size, srcK, tgtK, 1)

        src_corr = torch.matmul(tgtCandidates, valCandidates).squeeze(-1).permute(0, 2, 1)

        # choose src topK
        src = src.transpose(2, 1).contiguous()  # (batch_size, num, 3)
        src = src.view(batch_size * num_src, n_dims)[idx, :]
        src = src.view(batch_size, srcK, n_dims).permute(0, 2, 1)  # (batch_size, 3, srcK)

        return src, src_corr

    def getCopairALL(self, src, src_emb, tgt, tgt_emb):
        batch_size, n_dims, num_points = src.size()
        # Calculate the distance matrix
        inner = -2 * torch.matmul(src_emb.transpose(2, 1).contiguous(), tgt_emb)
        xx = torch.sum(src_emb ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(tgt_emb ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        pairwise_distance = pairwise_distance - yy

        scores = torch.softmax(pairwise_distance, dim=2)  # [b,num,num]
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        return src, src_corr


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr):

        batch_size = src.size(0)

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        if torch.isnan(H).sum() > 0:
            print('')
            print('')
            print('')
            print('H :nan')
            if torch.isnan(src_centered).sum() > 0:
                print('src_centered :nan')
            if torch.isnan(src_corr_centered).sum() > 0:
                print('src_corr_centered :nan')

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


class VcpByDis(nn.Module):
    def __init__(self, args):
        super(VcpByDis, self).__init__()
        self.emb_nn = args.emb_nn

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]

        src = input[2]
        tgt = input[3]

        d_k = src_embedding.size(1)

        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        return src, src_corr


class VcpAtt(nn.Module):
    def __init__(self, args):
        super(VcpAtt, self).__init__()
        self.emb_dims = args.emb_dims
        self.linears_emb = clones(nn.Linear(self.emb_dims, self.emb_dims), 2)
        self.linears_3d = clones(nn.Linear(3, 3), 2)
        self.attn = None
        self.dropout = None
        self.mask = None

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]

        src = input[2]
        tgt = input[3]

        query = src_embedding.transpose(2, 1).contiguous()
        key = tgt_embedding.transpose(2, 1).contiguous()
        value = tgt

        query = self.linears_emb[0](query)
        key = self.linears_emb[1](key)

        query = query.transpose(2, 1).contiguous()
        key = key.transpose(2, 1).contiguous()

        inner = -2 * torch.matmul(query.transpose(2, 1).contiguous(), key)
        xx = torch.sum(query ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(key ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        pairwise_distance = pairwise_distance - yy

        scores = torch.softmax(pairwise_distance, dim=2)  # [b,num,num]
        src_corr = torch.matmul(value, scores.transpose(2, 1).contiguous())
        return src, src_corr


class VCRNet(nn.Module):
    def __init__(self, args):
        super(VCRNet, self).__init__()
        self.emb_dims = args.emb_dims  # Dimension of embeddings, default = 512
        self.cycle = args.cycle  # Whether to use cycle consistency, default = False
        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif args.emb_nn == 'lpdnet':  # default
            self.emb_nn = LPDNet(args)
        else:
            raise Exception('Not implemented')

        if args.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':  # default
            self.pointer = Transformer(args=args)
        else:
            self.pointer = None

        if args.vcp_nn == 'topK':  # default
            self.head = VcpTopK(args=args)
        elif args.vcp_nn == 'att':
            self.head = VcpAtt(args=args)
        elif args.vcp_nn == 'dist':
            self.head = VcpByDis(args=args)
        else:
            raise Exception("Not implemented")

        self.svd = SVDHead(args=args)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]

        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        if self.pointer is not None:
            src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)
            src_embedding = src_embedding + src_embedding_p
            tgt_embedding = tgt_embedding + tgt_embedding_p

        srcK, src_corrK = self.head(src_embedding, tgt_embedding, src, tgt)

        rotation_ab, translation_ab = self.svd(srcK, src_corrK)

        if self.cycle:
            srcK_ba, src_corrK_ba = self.head(tgt_embedding, src_embedding, tgt, src)
            rotation_ba, translation_ba = self.svd(srcK_ba, src_corrK_ba)
        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

        return srcK, src_corrK, rotation_ab, translation_ab, rotation_ba, translation_ba


def test_one_epoch(args, net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss_VCRNet = 0

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

            if args.iter > 0:
                srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = vcrnetIter(
                    net, src, target, iter=args.iter)
            elif args.iter == 0:
                srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = vcrnetIcpNet(
                    args, net, src, target)
            else:
                raise RuntimeError('args.iter')

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

            # Predicted point cloud
            transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
            # Real point cloud
            transformed_srcK = transform_point_cloud(srcK, rotation_ab, translation_ab)

            # transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
            # from PC_reg_gif.draw import plot3d2
            # plot3d2(transformed_src[1], target[1])
            # plot3d2(transformed_target[1], src[1])

            ###########################
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
            if args.loss == 'pose':
                loss_VCRNet = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                              + F.mse_loss(translation_ab_pred, translation_ab)
            elif args.loss == 'point':
                loss_VCRNet = torch.nn.functional.mse_loss(transformed_srcK, src_corrK)
            else:
                lossPose = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                           + F.mse_loss(translation_ab_pred, translation_ab)
                transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
                lossPoint = torch.nn.functional.mse_loss(transformed_src, target)
                loss_VCRNet = lossPose + 0.1 * lossPoint

            loss_pose = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                        + F.mse_loss(translation_ab_pred, translation_ab)

            total_loss_VCRNet += loss_VCRNet.item() * batch_size

            if args.cycle:
                rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
                translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                            translation_ab_pred.view(batch_size, 3, 1)).view(batch_size,
                                                                                                             3)
                                               + translation_ba_pred) ** 2, dim=[0, 1])
                cycle_loss = rotation_loss + translation_loss

                loss_pose = loss_pose + cycle_loss * 0.1

            total_loss += loss_pose.item() * batch_size

            if args.cycle:
                total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

            mse_ab += torch.mean((transformed_srcK - src_corrK) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ab += torch.mean(torch.abs(transformed_srcK - src_corrK), dim=[0, 1, 2]).item() * batch_size

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
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba, total_loss_VCRNet * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt):
    net.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss_VCRNet = 0

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
        srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

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

        transformed_srcK = transform_point_cloud(srcK, rotation_ab, translation_ab)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        if args.loss == 'pose':
            loss_VCRNet = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                          + F.mse_loss(translation_ab_pred, translation_ab)
        elif args.loss == 'point':
            loss_VCRNet = torch.nn.functional.mse_loss(transformed_srcK, src_corrK)
        else:
            lossPose = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                       + F.mse_loss(translation_ab_pred, translation_ab)
            lossPoint = torch.nn.functional.mse_loss(transformed_src, target)
            loss_VCRNet = lossPose + 0.1 * lossPoint

        loss_VCRNet.backward()
        total_loss_VCRNet += loss_VCRNet.item() * batch_size

        loss_pose = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)
        if args.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                        translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                           + translation_ba_pred) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss_pose = loss_pose + cycle_loss * 0.1

        opt.step()
        total_loss += loss_pose.item() * batch_size

        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

        mse_ab += torch.mean((transformed_srcK - src_corrK) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_srcK - src_corrK), dim=[0, 1, 2]).item() * batch_size

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
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba, total_loss_VCRNet * 1.0 / num_examples


def testVCRNet(args, net, test_loader, boardio, textio):
    test_loss_Pose, test_cycle_loss_Pose, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba, test_loss_VCRNet = test_one_epoch(args, net, test_loader)
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
        'EPOCH:: %d, Loss: %f, test_LossPose: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
        'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
        % (-1, test_loss_VCRNet, test_loss_Pose, test_cycle_loss_Pose, test_mse_ab, test_rmse_ab, test_mae_ab,
           test_r_mse_ab, test_r_rmse_ab,
           test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    if args.cycle:
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, test_LossPose, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (-1, test_loss_VCRNet, test_loss_Pose, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba,
                         test_r_rmse_ba,
                         test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    loss = np.sum((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2, axis=-1)
    idx = np.argsort(loss, axis=-1, kind='quicksort', order=None)

    loss = np.sum((test_translations_ab - test_translations_ab_pred) ** 2, axis=-1)
    idx = np.argsort(loss, axis=-1, kind='quicksort', order=None)

    return test_loss_Pose


def trainVCRNet(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        # print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        # print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.000001)

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
        train_loss_Pose, train_cycle_loss, \
        train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, \
        train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
        train_translations_ba_pred, train_eulers_ab, train_eulers_ba, train_loss_VCRNet = train_one_epoch(args, net,
                                                                                                          train_loader,
                                                                                                          opt)

        test_loss_Pose, test_cycle_loss_Pose, \
        test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
        test_translations_ba_pred, test_eulers_ab, test_eulers_ba, test_loss_VCRNet = test_one_epoch(args, net,
                                                                                                     test_loader)

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

        if best_test_loss >= test_loss_Pose:
            best_test_loss = test_loss_Pose
            best_test_cycle_loss = test_cycle_loss_Pose

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

        # scheduler.step()
        scheduler.step(best_test_loss)
        lr = opt.param_groups[0]['lr']

        if lr <= 0.0000011:
            break

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint(
            'EPOCH:: %d, Loss: %f, LossPose: %f, Cycle Loss:, %f, lr: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
            'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
            % (
                epoch, train_loss_VCRNet, train_loss_Pose, train_cycle_loss, lr, train_mse_ab, train_rmse_ab,
                train_mae_ab,
                train_r_mse_ab,
                train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))
        if args.cycle:
            textio.cprint('B--------->A')
            textio.cprint('EPOCH:: %d, Loss: %f, LossPose: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                          'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                          % (epoch, train_loss_VCRNet, train_loss_Pose, train_mse_ba, train_rmse_ba, train_mae_ba,
                             train_r_mse_ba, train_r_rmse_ba,
                             train_r_mae_ba, train_t_mse_ba, train_t_rmse_ba, train_t_mae_ba))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint(
            'EPOCH:: %d, Loss: %f, LossPose: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
            'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
            % (epoch, test_loss_VCRNet, test_loss_Pose, test_cycle_loss_Pose, test_mse_ab, test_rmse_ab, test_mae_ab,
               test_r_mse_ab,
               test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
        if args.cycle:
            textio.cprint('B--------->A')
            textio.cprint('EPOCH:: %d, Loss: %f, LossPose: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                          'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                          % (
                              epoch, test_loss_VCRNet, test_loss_Pose, test_mse_ba, test_rmse_ba, test_mae_ba,
                              test_r_mse_ba,
                              test_r_rmse_ba,
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

        boardio.add_scalar('A->B/train/loss', train_loss_VCRNet, epoch)
        boardio.add_scalar('A->B/train/lossPose', train_loss_Pose, epoch)

        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss_VCRNet, epoch)
        boardio.add_scalar('A->B/test/lossPose', test_loss_Pose, epoch)
        boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/lr', lr, epoch)
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))

        gc.collect()
