#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from util.util import knn, get_graph_feature, farthest_point_sample


# TranformNet
# input x [B,num_dims,num]
class TranformNet(nn.Module):
    """Transform net in LPD-net reference [11](PointNet)

    Aiming at rotational translation invariance
    """

    def __init__(self, k=3, negative_slope=1e-2):
        super(TranformNet, self).__init__()
        self.negative_slope = negative_slope
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.LeakyReLU(negative_slope=self.negative_slope)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        """forward function for TranformNet

        Args:
            x: [B,num_dims,num]

        Returns:
            Transformation matrix of size numxnum
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        iden = torch.eye(self.k, dtype=torch.float32, device=device).view(1, self.k * self.k).repeat(batchsize, 1)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class LPDNet(nn.Module):
    """Implement for LPDNet using pytorch package.

    """

    def __init__(self, args, negative_slope=0.0):
        super(LPDNet, self).__init__()
        self.negative_slope = negative_slope
        self.k = 20
        self.t3d = args.t3d
        self.tfea = args.tfea
        self.emb_dims = args.emb_dims
        # [b,6,num,20]
        self.convDG1 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=True),
                                     nn.LeakyReLU(negative_slope=self.negative_slope))
        self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                     nn.LeakyReLU(negative_slope=self.negative_slope))
        self.convSN1 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=True),
                                     nn.LeakyReLU(negative_slope=self.negative_slope))

        self.conv1_lpd = nn.Conv1d(3, 64, kernel_size=1, bias=True)
        self.conv2_lpd = nn.Conv1d(64, 64, kernel_size=1, bias=True)
        self.conv3_lpd = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=True)
        if self.t3d:
            self.t_net3d = TranformNet(3)
        if self.tfea:
            self.t_net_fea = TranformNet(64)

    # input x: # [B,num_dims,num]
    # output x: # [b,emb_dims,num]
    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        #
        xInit3d = x
        if self.t3d:
            trans = self.t_net3d(x)
            x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)

        x = F.leaky_relu(self.conv1_lpd(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.conv2_lpd(x), negative_slope=self.negative_slope)

        if self.tfea:
            trans_feat = self.t_net_fea(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        # Serial structure
        # Dynamic Graph cnn for feature space
        x = get_graph_feature(x, k=self.k)  # [b,64*2,num,20]
        x = self.convDG1(x)  # [b,128,num,20]
        x1 = x.max(dim=-1, keepdim=True)[0]  # [b,128,num,1]
        x = self.convDG2(x)  # [b,128,num,20]
        x2 = x.max(dim=-1, keepdim=True)[0]  # [b,128,num,1]

        # Spatial Neighborhood fusion for cartesian space
        idx = knn(xInit3d, k=self.k)
        x = get_graph_feature(x2.squeeze(-1), idx=idx, k=self.k)  # [b,128*2,num,20]
        x = self.convSN1(x)  # [b,256,num,20]
        x3 = x.max(dim=-1, keepdim=True)[0]  # [b,256,num,1]

        x = torch.cat((x1, x2, x3), dim=1).squeeze(-1)  # [b,512,num]
        x = F.leaky_relu(self.conv3_lpd(x), negative_slope=self.negative_slope).view(batch_size, -1,
                                                                                     num_points)  # [b,emb_dims,num]
        return x


class LPD(nn.Module):
    def __init__(self, args):
        super(LPD, self).__init__()
        self.emb_dims = args.emb_dims
        self.num_points = args.num_points
        self.negative_slope = 0.2
        self.emb_nn = LPDNet(args, negative_slope=self.negative_slope)
        self.cycle = args.cycle

    def forward(self, *input):
        # [B,3,num]
        src = input[0]
        tgt = input[1]
        batch_size = src.size(0)
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        loss = self.getLoss(src, src_embedding, tgt_embedding)
        mse_ab_ = torch.mean((src_embedding - tgt_embedding) ** 2, dim=[0, 1, 2]) * batch_size
        mae_ab_ = torch.mean(torch.abs(src_embedding - tgt_embedding), dim=[0, 1, 2]) * batch_size

        return src_embedding, tgt_embedding, loss, mse_ab_, mae_ab_

    def kfn(self, x, k=20):
        inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)  # [b,num,num]
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [b,1,num] 

        pairwise_distance = xx + inner
        pairwise_distance = pairwise_distance + xx.transpose(2, 1).contiguous()  # [b,num,num]

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, k, neg_k)
        return idx

    # src_embedding_k (batch_size, dims, k)
    # tgt_embedding_k (batch_size, dims, k, 1)
    # topFarTgt (batch_size, dims, k, neg_k)
    def triplet_loss(self, src_embedding_k, tgt_embedding_k, topFarTgt):
        batch_size, dims, k, num_pos = tgt_embedding_k.size()
        batch_size, dims, k, num_neg = topFarTgt.size()
        margin = 1.0
        src_embedding_k = src_embedding_k.unsqueeze(3)
        src_embedding_k_p = src_embedding_k.repeat(1, 1, 1, num_pos)
        src_embedding_k_n = src_embedding_k.repeat(1, 1, 1, num_neg)
        dp_loss = torch.mean(((src_embedding_k_p - tgt_embedding_k) ** 2), dim=[1, 3])
        dn_loss = torch.mean(((src_embedding_k_n - topFarTgt) ** 2), dim=[1, 3])

        loss = torch.max(torch.cuda.FloatTensor([0.0]), 1 - dn_loss / (margin + dp_loss))

        return loss

    # src src_embedding [B,dims,num]
    def getLoss(self, src, src_embedding, tgt_embedding, k=32, neg_k=8):
        batch_size, pt_dims, num_points = src.size()
        _, emb_dims, _ = src_embedding.size()

        sampleIdx = farthest_point_sample(src, npoint=k)  # [B,K]
        sample_PtIdx = sampleIdx.unsqueeze(1).repeat(1, pt_dims, 1)
        sample_EmbIdx = sampleIdx.unsqueeze(1).repeat(1, emb_dims, 1)
        src_k = torch.gather(src, index=sample_PtIdx, dim=2)
        src_embedding_k = torch.gather(src_embedding, index=sample_EmbIdx, dim=2)  # [B,dims,k]
        tgt_embedding_k = torch.gather(tgt_embedding, index=sample_EmbIdx, dim=2)  # [B,dims,k]

        idx = self.kfn(src_k, k=neg_k)

        idx_base = torch.arange(0, batch_size,
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).view(-1, 1, 1) * k

        idx = idx + idx_base  # (batch_size, k, neg_k)

        idx = idx.view(-1)  # (batch_size * k * neg_k)

        topFarTgt = tgt_embedding_k.transpose(2, 1).contiguous().view(batch_size * k, -1)[idx,
                    :]  # (batch_size * k * neg_k,emb_dims)
        topFarTgt = topFarTgt.view(batch_size, k, neg_k, -1).permute(0, 3, 1, 2)

        tgt_embedding_k_p = tgt_embedding_k.unsqueeze(3)
        loss_triplet = self.triplet_loss(src_embedding_k, tgt_embedding_k_p, topFarTgt)

        src_embedding = src_embedding.transpose(2, 1).contiguous()
        tgt_embedding = tgt_embedding.transpose(2, 1).contiguous()
        src_length = torch.norm(src_embedding, dim=-1)
        tgt_length = torch.norm(tgt_embedding, dim=-1)
        identity = torch.empty((batch_size, num_points),
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).fill_(1)
        loss_norm1 = torch.sqrt(F.mse_loss(src_length, identity))
        loss_norm2 = torch.sqrt(F.mse_loss(tgt_length, identity))

        loss = loss_triplet.mean() + (loss_norm1 + loss_norm2) / 2.0 * 0.03

        return loss


def test_one_epoch(args, net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0

    total_loss = 0
    num_examples = 0

    with torch.no_grad():
        for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
                test_loader):
            src = src.cuda()
            target = target.cuda()
            batch_size = src.size(0)
            num_examples += batch_size
            # [b, emb_dims, num]
            src_embedding, tgt_embedding, loss, mse_ab_, mae_ab_ = net(src, target)

            total_loss += loss.sum().item() * batch_size
            mse_ab += mse_ab_.sum().item()
            mae_ab += mae_ab_.sum().item()

    return total_loss * 1.0 / num_examples, mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt):
    net.train()
    mse_ab = 0
    mae_ab = 0

    total_loss = 0
    num_examples = 0

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
            train_loader):
        src = src.cuda()
        target = target.cuda()
        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        src_embedding, tgt_embedding, loss, mse_ab_, mae_ab_ = net(src, target)
        loss = loss.sum()
        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size
        mse_ab += mse_ab_.sum().item()
        mae_ab += mae_ab_.sum().item()

    return total_loss * 1.0 / num_examples, mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples


def testLPD(args, net, test_loader, boardio, textio):
    test_loss, test_mse_ab, test_mae_ab = test_one_epoch(args, net, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)

    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f'
                  % (-1, test_loss, test_mse_ab, test_rmse_ab, test_mae_ab))
    return test_loss


def trainLPD(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    best_test_loss = np.inf

    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    for epoch in range(args.epochs):

        train_loss, train_mse_ab, train_mae_ab = train_one_epoch(args, net, train_loader, opt)

        test_loss, test_mse_ab, test_mae_ab = test_one_epoch(args, net, test_loader)

        scheduler.step()

        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f'
                      % (epoch, train_loss, train_mse_ab, train_rmse_ab, train_mae_ab))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f'
                      % (epoch, test_loss, test_mse_ab, test_rmse_ab, test_mae_ab))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f'
                      % (epoch, best_test_loss, best_test_mse_ab, best_test_rmse_ab, best_test_mae_ab))

        boardio.add_scalar('A->B/train/loss', train_loss, epoch)
        boardio.add_scalar('A->B/train/MSE', train_mse_ab, epoch)
        boardio.add_scalar('A->B/train/RMSE', train_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/MAE', train_mae_ab, epoch)

        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss, epoch)
        boardio.add_scalar('A->B/test/MSE', test_mse_ab, epoch)
        boardio.add_scalar('A->B/test/RMSE', test_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/MAE', test_mae_ab, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/MSE', best_test_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/RMSE', best_test_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/MAE', best_test_mae_ab, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))

        gc.collect()
