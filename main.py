#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from model.dcp_model import trainDCP, testDCP
from model.icp_model import testICP
from model.lpdnet_model import trainLPD, testLPD
from model.vcrnet_model import trainVCRNet, testVCRNet
from util.initPara import para


def main():
    args, net, train_loader, test_loader, boardio, textio = para()

    if args.eval:
        if args.model == 'vcrnet':
            testVCRNet(args, net, test_loader, boardio, textio)
        elif args.model == 'dcp':
            testDCP(args, net, test_loader, boardio, textio)
        elif args.model == 'lpd':
            testLPD(args, net, test_loader, boardio, textio)
        elif args.model == 'icp':
            testICP(args, net, test_loader, boardio, textio)
    else:
        if args.model == 'vcrnet':
            trainVCRNet(args, net, train_loader, test_loader, boardio, textio)
        elif args.model == 'lpd':
            trainLPD(args, net, train_loader, test_loader, boardio, textio)
        elif args.model == 'dcp':
            trainDCP(args, net, train_loader, test_loader, boardio, textio)
        elif args.model == 'icp':
            print("icp can't be trained")

    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
