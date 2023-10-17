"""
federated learning with different aggregation strategy on office dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from nets.models import UNet
from utils.loss import MultiClassDiceLoss,FocalLoss_Ori
from torch.utils.data import DataLoader

from utils.LoadData import LoadDatasets
from utils.data_utils import OfficeDataset
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np
import os

def train(model, data_loader, optimizer, device):
    model.train()
    loss_all = 0
    total = 0
    dice_all = 0
    for datas, target in data_loader:
        optimizer.zero_grad()
        data = datas['image']
        data = data.to(device)
        target = datas['label']
        target = target.to(device)
        #没有激活函数
        output = model(data)
        outputssoft = torch.softmax(output, dim=1)
        mDiceLoss, mDice = MultiClassDiceLoss()(outputssoft, target)
        floss = FocalLoss_Ori()(output, target)
        loss = mDiceLoss + floss
        loss_all += loss.item()
        dice_all += mDice.item()

        loss.backward()
        optimizer.step()
    return loss_all / len(data_loader),dice_all/len(data_loader)


def train_prox(args, model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


def val(model, data_loader, device):
    model.eval()
    loss_all = 0
    dice_all = 0
    with torch.no_grad():
        for datas, target in data_loader:
            data = datas['image']
            data = data.to(device)
            target = datas['label']
            target = target.to(device)
            # 没有激活函数
            output = model(data)
            outputssoft = torch.softmax(output, dim=1)
            mDiceLoss, mDice = MultiClassDiceLoss()(outputssoft, target)
            floss = FocalLoss_Ori()(output, target)
            loss = mDiceLoss + floss
            loss_all += loss.item()
            dice_all += mDice.item()
        return loss_all / len(data_loader), dice_all / len(data_loader)



################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models



if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    seed = 500
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--iters', type=int, default=300, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedbn', help='[FedBN | FedAvg | FedProx]')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='/mnt/sda/li/checkpoint/office', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    args = parser.parse_args()

    exp_folder = 'fed_office'

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
    log = args.log
    if log:
        log_path = os.path.join('./logs/office/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)


        logfile = None  # 定义并初始化logfile变量


        logfile = open(os.path.join(log_path, '{}.log'.format(args.mode)), 'a')  # 打开日志文件
        print(logfile)
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))

    # train_loaders, val_loaders, test_loaders = prepare_data(args)
    A_DATA, B_DATA, C_DATA, test_dataset, val_A, val_B, val_C = LoadDatasets()

    A_dataloder = DataLoader(A_DATA, batch_size=args.batch, shuffle=True)
    B_dataloder = DataLoader(B_DATA, batch_size=args.batch, shuffle=True)
    C_dataloder = DataLoader(C_DATA, batch_size=args.batch, shuffle=True)
    train_loaders = [A_dataloder,B_dataloder,C_dataloder]
    test_loaders = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)
    A_valloder = DataLoader(val_A, batch_size=args.batch, shuffle=True)
    B_valloder = DataLoader(val_B, batch_size=args.batch, shuffle=True)
    C_valloder = DataLoader(val_C, batch_size=args.batch, shuffle=True)
    val_loaders = [A_valloder,B_valloder,C_valloder]


    # setup model
    server_model = UNet(1,5).to(device)
    loss_fun = nn.CrossEntropyLoss()
    # name of each datasets
    #datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
    # federated client number
    client_num = 3
    client_weights = [1 / client_num for i in range(client_num)]
    # each local client model
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/office/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_dice = checkpoint['best_epoch'], checkpoint['best_dice']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
        best_epoch = 0
        best_dice = [0. for j in range(client_num)]
        start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers = [optim.Adam(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx, model in enumerate(models):
                if args.mode.lower() == 'fedprox':
                    # skip the first server model(random initialized)
                    if a_iter > 0:
                        train_loss, train_dice = train_prox(args, model, train_loaders[client_idx],
                                                           optimizers[client_idx], device)
                    else:
                        train_loss, train_dice = train(model, train_loaders[client_idx], optimizers[client_idx],
                                                      device)
                else:
                    train_loss,train_dice = train(model, train_loaders[client_idx], optimizers[client_idx],
                                                  device)
                    print(' Site-{:<10s}| Train Loss: {:.4f} | Train Dice: {:.4f}'.format(str(client_idx), train_loss,
                                                                                          train_dice))
        print("============ Aggregation epoch {} ============".format(wi + a_iter * args.wk_iters))
        with torch.no_grad():
            # aggregation
            server_model, models = communication(args, server_model, models, client_weights)
            # Report loss after aggregation
            for client_idx, model in enumerate(models):
                train_loss, train_dice = val(model, train_loaders[client_idx], device)
                print(' Site-{:<10s}| Train Loss: {:.4f} | Train Dice: {:.4f}'.format(str(client_idx), train_loss,
                                                                                     train_dice))
                if args.log:
                    logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Dice: {:.4f}\n'.format(str(client_idx),
                                                                                                   train_loss,
                                                                                                   train_dice))
            # Validation
            val_acc_list = [None for j in range(client_num)]
            for client_idx, model in enumerate(models):
                val_loss, val_dice = val(model, val_loaders[client_idx], device)
                val_acc_list[client_idx] = val_dice
                print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Dice: {:.4f}'.format(str(client_idx), val_loss,
                                                                                   val_dice), flush=True)
                if args.log:
                    logfile.write(
                        ' Site-{:<10s}| Val  Loss: {:.4f} | Val  Dice: {:.4f}\n'.format(str(client_idx), val_loss,
                                                                                       val_dice))
            # Record best
            if np.mean(val_acc_list) > np.mean(best_dice):
                for client_idx in range(client_num):
                    best_dice[client_idx] = val_acc_list[client_idx]
                    best_epoch = a_iter
                    best_changed = True
                    print(' Best site-{:<10s}| Epoch:{} | Val Dice: {:.4f}'.format(str(client_idx), best_epoch,
                                                                                  best_dice[client_idx]))
                    if args.log:
                        logfile.write(
                            ' Best site-{:<10s} | Epoch:{} | Val Dice: {:.4f}\n'.format(str(client_idx), best_epoch,
                                                                                       best_dice[client_idx]))
            i = 0
            if best_changed:
                i=i+1
                print(' Saving the server checkpoint to {}...'.format(SAVE_PATH))
                #logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))
                filename = SAVE_PATH+'/fedbn_'+str(i)+'.pth.tar'
                if args.mode.lower() == 'fedbn':
                    torch.save({
                        # 'model_0': models[0].state_dict(),
                        # 'model_1': models[1].state_dict(),
                        # 'model_2': models[2].state_dict(),
                        'state_dict': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_dice': best_dice,
                        'a_iter': a_iter
                    }, filename)
                    best_changed = False
                    for client_idx in range(3):
                        _, test_dice = val(models[client_idx], val_loaders[client_idx], device)
                        print(' Test site-{:<10s}| Epoch:{} | Test Dice: {:.4f}'.format(str(client_idx), best_epoch, test_dice))
                        if args.log:
                            logfile.write(
                                ' Test site-{:<10s}| Epoch:{} | Test Dice: {:.4f}\n'.format(str(client_idx), best_epoch,
                                                                                           test_dice))
                else:
                    torch.save({
                        'state_dict': server_model.state_dict(),
                        'best_epoch': best_epoch,
                        'best_dice': best_dice,
                        'a_iter': a_iter
                    }, SAVE_PATH)
                    best_changed = False
                    for client_idx in range(4):
                        _, test_dice = val(server_model, val_loaders[client_idx], device)
                        print(' Test site-{:<10s}| Epoch:{} | Test Dice: {:.4f}'.format(str(client_idx), best_epoch, test_dice))
                        if args.log:
                            logfile.write(
                                ' Test site-{:<10s}| Epoch:{} | Test Dice: {:.4f}\n'.format(str(client_idx), best_epoch,
                                                                                           test_dice))
            if log:
                logfile.flush()
    if log:
        logfile.flush()
        logfile.close()
