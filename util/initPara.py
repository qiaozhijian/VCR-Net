#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import socket
from datetime import datetime

import numpy as np
import pandas as pd
import sympy as sp
import torch
import torch.nn as nn
from sympy import Float as spFloat
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from model.dcp_model import DCP
from model.icp_model import ICP
from model.lpdnet_model import LPD
from model.vcrnet_model import VCRNet
from util.data import ModelNet40, KITTI


def delModule(path):
    state_dict = torch.load(path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[0:6] == 'module':
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def initNet(vcrnet):
    lpdNet = vcrnet.emb_nn
    if hasattr(lpdNet, 'convDG1'):
        print('init lpdNet')
        for m in lpdNet.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=lpdNet.negative_slope, mode="fan_in", nonlinearity="leaky_relu")
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=lpdNet.negative_slope, mode="fan_in", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, 0)

    attHead = vcrnet.head
    if hasattr(attHead, 'linears_emb'):
        print('init attHead')
        for i in range(2):
            nn.init.eye_(attHead.linears_emb[i].weight)
            nn.init.constant_(attHead.linears_emb[i].bias, 0)
        for i in range(2):
            nn.init.eye_(attHead.linears_3d[i].weight)
            nn.init.constant_(attHead.linears_3d[i].bias, 0)


def saveNetAsExcel(net):
    net = list(net.named_parameters())
    net_df = pd.DataFrame(net)
    # create and writer pd.DataFrame to excel
    writer = pd.ExcelWriter('Net.xlsx')
    net_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        """print to cmd and save to a txt file at the same time

        """
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.exp_name = args.model + '-' + args.emb_nn + '-' + datetime.now().strftime("%d-%H-%M-%S")
    if args.eval:
        subDir = 'test/'
    else:
        subDir = 'train/'
    args.exp_name = args.exp_name + '-' + socket.gethostname()[:3]
    if not os.path.exists('checkpoints/' + subDir):
        os.makedirs('checkpoints/' + subDir)
    args.exp_name = subDir + args.exp_name
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

    # when reserve is 0.75
    # if we cut (1-reserve) percent of 2 pcls, the real overlapping of 2 pcls is between 50% and 100% relative to origin whole pcl.
    # so we solve reserve to make the expectation of real overlapping be overlap which we input to args
    # at last overlap2 represents overlapping relative to cut pcl.
    # when overlap is 0.575, reserve is 0.75
    n = sp.Symbol('n')
    a = (n - 3.0 / 2.0 * n ** 2) * (1.0 - 2.0 * n)
    b = 0.5 * (n - 1.0) ** 2 * n - 1.0 / 6.0 * (1.0 - n) ** 3 + 1.0 / 6.0 * (1.0 - 2.0 * n) ** 3
    f = ((a + b) * 2.0 + (1.0 - 2.0 * n) ** 3) / (1.0 - n) ** 2 - args.overlap
    re = np.asarray(sp.solve(f, n))
    for i in range(3):
        if type(re[i]) == spFloat and re[i] <= 0.5 and re[i] >= 0.0:
            args.reserve = 1 - re[i]
            break
    args.overlap2 = args.overlap / args.reserve
    return args


def para():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--overlap', type=float, default=0.75)
    parser.add_argument('--model', type=str, default='vcrnet',
                        choices=['dcp', 'lpd', 'vcrnet', 'icp'])
    parser.add_argument('--gaussian_noise', type=bool, default=False,
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False,
                        help='Wheter to test on unseen category')
    parser.add_argument('--factor', type=float, default=4,
                        help='Divided factor for rotations')
    parser.add_argument('--emb_nn', type=str, default='lpdnet',
                        choices=['pointnet', 'dgcnn', 'lpdnet', 'lpdnetorigin'],
                        help='Embedding nn to use, [pointnet, dgcnn, lpdnet]')
    parser.add_argument('--vcp_nn', type=str, default='topK',
                        choices=['topK', 'att', 'dist'],
                        help='Strategy to generate virtual corresponding points, [topK, att, dist]')
    parser.add_argument('--emb_dims', type=int, default=512,
                        help='Dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Num of points to use')
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--ff_dims', type=int, default=1024,
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--partial', action='store_true', default=False,
                        help='use part of the point cloud')
    parser.add_argument('--t3d', action='store_true', default=False,
                        help='3d tranform ')
    parser.add_argument('--tfea', action='store_true', default=False,
                        help='feature transform')
    parser.add_argument('--loss', type=str, default='point',
                        choices=['pose', 'point'],
                        help='Name of the experiment')
    parser.add_argument('--cycle', type=bool, default=False,
                        help='Whether to use cycle consistency')
    parser.add_argument('--model_path', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--dataset', type=str, default='modelnet40',
                        choices=['modelnet40', 'kitti'],
                        help='dataset to use')
    parser.add_argument('--n_blocks', type=int, default=1,
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout ratio in transformer')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='Name of the experiment')
    parser.add_argument('--pointer', type=str, default='transformer',
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--use_point_loss', action='store_true', default=False,
                        help='Use the L2 distance between the matched points for loss')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    args = _init_(args)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if not torch.cuda.is_available():
        textio.cprint('no cuda detect!')

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(args=args, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
        test_loader = DataLoader(
            ModelNet40(args=args, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=4)
    elif args.dataset == 'kitti':
        train_loader = DataLoader(
            KITTI(args=args, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            KITTI(args=args, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("not implemented for the dataset: {}".format(args.dataset))

    if args.model == 'dcp':
        net = DCP(args).cuda()
    elif args.model == 'lpd':
        net = LPD(args).cuda()
    elif args.model == 'vcrnet':
        net = VCRNet(args).cuda()
        initNet(net)
    elif args.model == 'icp':
        net = ICP(max_iterations=args.max_iterations).cuda()
    else:
        raise Exception('Not implemented for the model: {}'.format(args.model))

    if args.model_path == '':
        model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
    else:
        model_path = args.model_path
        textio.cprint(model_path)
    if not os.path.exists(model_path):
        textio.cprint("can't find pretrained model")
    else:
        textio.cprint("load pretrained model")
        net.load_state_dict(torch.load(model_path), strict=False)

    para = sum([np.prod(list(p.size())) for p in net.parameters()])

    textio.cprint('Model {} : params: {:4f}M'.format(net._get_name(), para * 4 / 1000 / 1000))

    net = nn.DataParallel(net)
    textio.cprint("Let's use {} GPUs!".format(torch.cuda.device_count()))

    return args, net, train_loader, test_loader, boardio, textio
