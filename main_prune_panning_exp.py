# -*- coding: utf-8 -*-

"""
Created on 09/19/2021
main_prune_panning.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from models.model_base import ModelBase
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.base.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import get_network
from pruner.Panning_exp import Panning, GraSP_fetch_data
from utils import mail_log


def init_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='configs/cifar10/resnet32/Panning_98.json')
    parser.add_argument('--config', type=str, default='configs/cifar10/vgg19/Panning_98.json')
    # parser.add_argument('--config', type=str, default='configs/mnist/lenet/Panning_90.json')
    parser.add_argument('--run', type=str, default='exp123')
    parser.add_argument('--epoch', type=str, default='666')
    parser.add_argument('--prune_mode', type=int, default=0)
    parser.add_argument('--prune_mode_pa', type=int, default=0)  # 第二次修剪模式
    parser.add_argument('--prune_conv', type=int, default=0)  # 修剪卷积核标志
    parser.add_argument('--prune_conv_pa', type=int, default=0)
    parser.add_argument('--core_link', type=int, default=0)  # 核链标志
    parser.add_argument('--enlarge', type=int, default=0)  # 扩张标志
    parser.add_argument('--prune_link', type=int, default=0)  # 按核链修剪
    parser.add_argument('--prune_epoch', type=int, default=0)  # 第二次修剪时间
    parser.add_argument('--single_data', type=int, default=0)  # 单次修剪的数据方式
    parser.add_argument('--gtg_mode', type=int, default=5)  # 训练过程求解梯度的模式
    parser.add_argument('--remain', type=float, default=666)
    parser.add_argument('--lr_mode', type=str, default='cosine', help='cosine or preset')
    parser.add_argument('--dp', type=str, default='../Data', help='dataset path')
    args = parser.parse_args()

    runs = None
    if len(args.run) > 0:
        runs = args.run
    config = process_config(args.config, runs)
    if args.epoch != '666':
        config.epoch = args.epoch
        print("set new epoch:{}".format(config.epoch))
    if args.remain != 666:
        config.target_ratio = (100 - args.remain) / 100.0
        print("set new target_ratio:{}".format(config.target_ratio))
    config.prune_mode = args.prune_mode
    config.prune_mode_pa = args.prune_mode_pa
    config.prune_conv = True if args.prune_conv == 1 else False
    config.prune_conv_pa = True if args.prune_conv_pa == 1 else False
    config.core_link = True if args.core_link == 1 else False
    config.enlarge = True if args.enlarge == 1 else False
    config.prune_link = True if args.prune_link == 1 else False
    config.prune_epoch = args.prune_epoch
    config.single_data = args.single_data
    config.lr_mode = args.lr_mode
    config.gtg_mode = args.gtg_mode
    config.dp = args.dp
    config.send_mail_head = (args.config + ' -> ' + args.run + '\n')
    config.send_mail_str = (mail_log.get_words() + '\n')
    config.send_mail_str += "=> 我能在河边钓一整天的🐟 <=\n"

    return config


def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'models/base/%s.py' % config.network.lower())
    path_main = os.path.join(path, 'main_prune_panning.py')
    path_pruner = os.path.join(path, 'pruner/Panning.py')
    logger = get_logger('log', logpath=config.summary_dir + '/',
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    return logger, writer


def print_mask_information(mb, logger):
    ratios = mb.get_ratio_at_each_layer()
    logger.info('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    re_str = '** Mask information of %s. Overall Remaining: %.2f%%\n' % (mb.get_name(), ratios['ratio'])
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
        re_str += '  (%d) %.2f%%\n' % (count, v)
        count += 1
    return re_str


def save_state(net, acc, epoch, loss, config, ckpt_path, is_best=False):
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'loss': loss,
        'args': config
    }
    if not is_best:
        torch.save(state, '%s/pruned_%s_%s%s_%d.t7' % (ckpt_path,
                                                       config.dataset,
                                                       config.network,
                                                       config.depth,
                                                       epoch))
    else:
        torch.save(state, '%s/finetuned_%s_%s%s_best.t7' % (ckpt_path,
                                                            config.dataset,
                                                            config.network,
                                                            config.depth))


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, iteration, lr_mode, gtg_mode=0, num_classes=10,
          masks=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = None
    if lr_mode == 'cosine':
        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_last_lr(), 0, 0, correct, total))
        writer.add_scalar('iter_%d/train/lr' % iteration, lr_scheduler.get_last_lr(), epoch)
    elif 'preset' in lr_mode:
        lr_scheduler(optimizer, epoch)
        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
        writer.add_scalar('iter_%d/train/lr' % iteration, lr_scheduler.get_lr(optimizer), epoch)

    # ------------- debug ----------------
    # 分析梯度项一阶二阶的变化
    # 求二阶不能这样
    # grad_l2 = 0
    # for idx, layer in enumerate(net.modules()):
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         # print(layer.weight.grad.shape)
    #         grad_l2 += layer.weight.grad.pow(2).sum()
    # grad_l2.sqrt()

    # 重新计算损失和建图，便于求一阶二阶导
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)

    inputs_one = []
    targets_one = []

    grad_w = None  # 一阶梯度 g

    # 10个类别，每类样本取10个
    samples_per_class = 10
    if gtg_mode == 0 or gtg_mode == 5:
        inputs, targets = GraSP_fetch_data(loader, num_classes, samples_per_class)  # 按样本数分组，标签相等
        # 不同标签组排列
        _index = []
        for i in range(samples_per_class):
            _index.extend([i + j * samples_per_class for j in range(0, num_classes)])
        inputs = inputs[_index]
        targets = targets[_index]
        # print(targets[:num_classes])
    elif gtg_mode == 1:
        inputs, targets = GraSP_fetch_data(loader, num_classes, samples_per_class)  # 按标签分组，等量样本
    elif gtg_mode == 2:
        inputs, targets = GraSP_fetch_data(loader, num_classes, samples_per_class, 1)  # 随机取100个样本
    elif gtg_mode == 3:
        inputs, targets = GraSP_fetch_data(loader, num_classes, 1, 1)  # 随机取10个样本
    elif gtg_mode == 4:
        inputs, targets = GraSP_fetch_data(loader, 2, 1, 1)  # 随机取2个样本
    else:
        inputs, targets = GraSP_fetch_data(loader, num_classes, samples_per_class)
    N = inputs.shape[0]

    # 考虑二范数的梯度
    if gtg_mode == 5:
        l2_loss = []
        for module in net.modules():
            if type(module) is nn.Conv2d:
                l2_loss.append((module.weight ** 2).sum() / 2.0)
        _l2_reg = 0.0005 * sum(l2_loss)
        print(_l2_reg)
        # _l2_reg = l2_regularization(net, 0.0005)
    else:
        _l2_reg = 0

    # 把数据分为前后两个list
    din = copy.deepcopy(inputs)
    dtarget = copy.deepcopy(targets)
    inputs_one.append(din[:N // 2])
    targets_one.append(dtarget[:N // 2])
    inputs_one.append(din[N // 2:])
    targets_one.append(dtarget[N // 2:])
    inputs = inputs.cuda()
    targets = targets.cuda()

    outputs = net.forward(inputs[:N // 2])
    loss = criterion(outputs, targets[:N // 2]) + _l2_reg
    grad_w_p = autograd.grad(loss, weights, retain_graph=True)
    grad_w = list(grad_w_p)

    outputs = net.forward(inputs[N // 2:])
    loss = criterion(outputs, targets[N // 2:]) + _l2_reg
    grad_w_p = autograd.grad(loss, weights, retain_graph=True)
    for idx in range(len(grad_w)):
        grad_w[idx] += grad_w_p[idx]

    # layer_key
    layer_key = [x for x in masks.keys()]

    # 梯度范数的梯度
    # 生气了，🤬
    all_gral2 = 0  # g0 * g1 and g0 * g2
    all_gl2 = 0  # g1 * g1 and g2 * g2
    all_g1_g2 = 0  # 2 * g1 * g2

    gr_l2 = 0
    grasp_l2 = 0
    last_grad_f = 0  # for debug
    _grad_hg = None  # 二阶梯度 Hg
    _grasp_hg = None  # 二阶梯度 Hg
    lam_q = 1 / len(inputs_one)
    for it in range(len(inputs_one)):
        inputs = inputs_one.pop(0).cuda()
        targets = targets_one.pop(0).cuda()
        outputs = net.forward(inputs)
        loss = criterion(outputs, targets) + _l2_reg

        # torch.autograd.grad() 对指定参数求导
        # .torch.autograd.backward() 动态图所有参数的梯度都计算，叠加到 .grad 属性
        grad_f = autograd.grad(loss, weights, create_graph=True)  # 一阶梯度 g
        # print([torch.mean(g) for g in grad_f])
        gr_l2 = 0
        grasp_l2 = 0
        g1_g2 = 0  # for debug
        g1_g2_list = []  # for debug 层g1g2
        # g1_g2_sim = 0  # for debug
        g1_g2_sim_list = []  # for debug 层相似度
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # grasp_l2 += (grad_w[count].data * grad_f[count]).sum()  # g0 * g1 and g0 * g2
                grasp_l2 += (grad_w[count].data * grad_f[count] * masks[layer_key[count]]).sum()  # g0 * g1 and g0 * g2
                # gr_l2 += (grad_f[count].pow(2).sum()) * lam_q    # g1 * g1 and g2 * g2
                gr_l2 += ((grad_f[count] * masks[layer_key[count]]).pow(2).sum()) * lam_q  # g1 * g1 and g2 * g2
                if it % 2 == 1:
                    layer_g1g2 = (grad_f[count] * last_grad_f[count]).sum()  # g1 * g2
                    # layer_g1g2 = (grad_f[count] * last_grad_f[count] * masks[layer_key[count]]).sum()  # g1 * g2
                    g1_g2 += layer_g1g2
                    g1_g2_list.append(layer_g1g2)
                    # g1_g2_sim += (grad_f[count] * last_grad_f[count]).sum() / (grad_f[count].pow(2).sum().sqrt()*last_grad_f[count].pow(2).sum().sqrt())
                    g1_g2_sim_list.append(layer_g1g2 / ((grad_f[count]).pow(2).sum().sqrt() * (last_grad_f[count]).pow(2).sum().sqrt()))
                    # g1_g2_sim_list.append(layer_g1g2 / (
                    #         (grad_f[count] * masks[layer_key[count]]).pow(2).sum().sqrt() * (
                    #         last_grad_f[count] * masks[layer_key[count]]).pow(2).sum().sqrt()))
                count += 1

        if it % 2 == 0:
            last_grad_f = grad_f
        all_gral2 += grasp_l2
        all_gl2 += gr_l2
        # print(grasp_l2, 2*gr_l2, 2*g1_g2)

        # gr_l2.sqrt()
        # if _grad_hg is None:
        #     _grad_hg = autograd.grad(gr_l2, weights, retain_graph=True)
        # else:
        #     _grad_hg = [_grad_hg[i] + autograd.grad(gr_l2, weights, retain_graph=True)[i] for i in range(len(_grad_hg))]
        #
        # if _grasp_hg is None:
        #     _grasp_hg = autograd.grad(grasp_l2, weights, retain_graph=True)
        # else:
        #     _grasp_hg = [_grasp_hg[i] + autograd.grad(grasp_l2, weights, retain_graph=True)[i] for i in range(len(_grasp_hg))]

    # ghg = 0  # 二阶项
    # for i in range(len(grad_w)):
    #     ghg += (grad_w[i]*_grad_hg[i]).sum()
    # ghg.sqrt()

    # 计算GraSP and GL2 的差值绝对值和
    # _layer_cnt = 0
    # _diff_sum = 0
    # _grasp_mean = 0
    # _grasp_sum = 0
    # for idx, layer in enumerate(net.modules()):
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         x = -layer.weight.data * _grasp_hg[_layer_cnt]  # -theta_q Hg
    #         q = -layer.weight.data * _grad_hg[_layer_cnt]  # -theta_q Hg
    #         # print(torch.mean(x), torch.sum(x))
    #         # print(torch.mean(l), torch.sum(l))
    #         # print(torch.mean(x))
    #         # print(torch.mean(q))
    #         # print('-'*10)
    #         _diff_sum += float(torch.sum(torch.abs(x-q)))
    #         _grasp_mean += float(torch.mean(x))
    #         _grasp_sum += float(torch.sum(x))
    #         _layer_cnt += 1
    # print('-'*20)
    # ------------------------------

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if lr_mode == 'cosine':
            desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (lr_scheduler.get_last_lr(), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        elif 'preset' in lr_mode:
            desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct,
                     total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('iter_%d/train/loss' % iteration, train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('iter_%d/train/acc' % iteration, 100. * correct / total, epoch)

    writer.add_scalar('iter_%d/train/gral2' % iteration, all_gral2, epoch)
    writer.add_scalar('iter_%d/train/gl2' % iteration, all_gl2, epoch)
    writer.add_scalar('iter_%d/train/g1_g2' % iteration, g1_g2, epoch)
    writer.add_scalar('iter_%d/train/g1g2_conv' % iteration, g1_g2_list[15], epoch)
    writer.add_scalar('iter_%d/train/g1g2_line' % iteration, g1_g2_list[16], epoch)
    writer.add_scalar('iter_%d/train/sim_conv' % iteration, g1_g2_sim_list[15], epoch)
    writer.add_scalar('iter_%d/train/sim_line' % iteration, g1_g2_sim_list[16], epoch)
    # writer.add_scalar('iter_%d/train/g1_g2_sim' % iteration, g1_g2_sim, epoch)
    # writer.add_scalar('iter_%d/train/ghg' % iteration, ghg, epoch)
    # writer.add_scalar('iter_%d/train/_diff_sum' % iteration, _diff_sum, epoch)
    # writer.add_scalar('iter_%d/train/_grasp_mean' % iteration, _grasp_mean, epoch)
    # writer.add_scalar('iter_%d/train/_grasp_sum' % iteration, _grasp_sum, epoch)


def test(net, loader, criterion, epoch, writer, iteration):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100. * correct / total

    writer.add_scalar('iter_%d/test/loss' % iteration, test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('iter_%d/test/acc' % iteration, 100. * correct / total, epoch)
    return acc


def train_once(mb, net, trainloader, testloader, writer, config, ckpt_path, learning_rate, weight_decay, num_epochs,
               iteration, ratio, num_classes, logger, last_mask, keep_masks=None, lr_mode='cosine'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = None
    if lr_mode == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif 'preset' in lr_mode:
        if lr_mode == 'preset_01':
            lr_schedule = {0: learning_rate * 0.01,
                           int(num_epochs * 0.5): learning_rate * 0.01,
                           int(num_epochs * 0.75): learning_rate * 0.01}
        else:
            lr_schedule = {0: learning_rate,
                           int(num_epochs * 0.5): learning_rate * 0.1,
                           int(num_epochs * 0.75): learning_rate * 0.01}
        lr_scheduler = PresetLRScheduler(lr_schedule)

    print_inf = ''
    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epochs):

        # 淘金
        if epoch == config.prune_epoch and config.prune_mode_pa > 0:
            masks, _ = Panning(mb.model, ratio, trainloader, 'cuda',
                               num_classes=num_classes,
                               samples_per_class=config.samples_per_class,
                               num_iters=config.get('num_iters', 1),
                               reinit=False,
                               prune_mode=config.prune_mode_pa,
                               prune_conv=config.prune_conv_pa,
                               add_link=config.core_link,
                               delete_link=config.core_link,
                               enlarge=config.enlarge,
                               prune_link=config.prune_link,
                               first_masks=keep_masks
                               )

            # # 与之前98%比较（筛选占比）
            # _all_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in last_mask.values()]))
            # _and_mask_num = 0
            # _la_cnt = 0
            # for m, g in masks.items():
            #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #         and_mask = masks[m] * last_mask[m]
            #         _now_mask_num = torch.sum(torch.cat([torch.flatten(and_mask == 1)]))
            #         _last_mask_num = torch.sum(torch.cat([torch.flatten(last_mask[m] == 1)]))
            #         print(_la_cnt, '-' * 20)
            #         # print(f'_now_mask_num: {_now_mask_num}')
            #         # print(f'_last_mask_num: {_last_mask_num}')
            #         print(f'/: {1 - _now_mask_num / _last_mask_num}')
            #         _and_mask_num += _now_mask_num
            #         _la_cnt += 1
            # print('all', '-' * 20)
            # # print(f'_all_mask_num: {_all_mask_num}')
            # # print(f'_and_mask_num: {_and_mask_num}')
            # print(f'/: {1 - _and_mask_num / _all_mask_num}')

            # print(masks)
            # ========== register mask ==================
            mb.register_mask(masks)
            # ========== print pruning details ============
            logger.info('**[%d] Mask and training setting: ' % iteration)
            print_inf = print_mask_information(mb, logger)

        train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer,
              iteration=iteration, lr_mode=lr_mode, gtg_mode=config.gtg_mode, num_classes=num_classes, masks=keep_masks)
        test_acc = test(net, testloader, criterion, epoch, writer, iteration)
        if lr_mode == 'cosine':
            lr_scheduler.step()

        if test_acc > best_acc and epoch > 10:
            print('Saving..')
            state = {
                'net': net,
                'acc': test_acc,
                'epoch': epoch,
                'args': config,
                'mask': mb.masks,
                'ratio': mb.get_ratio_at_each_layer()
            }
            path = os.path.join(ckpt_path, 'finetune_%s_%s%s_r%s_it%d_best.pth.tar' % (config.dataset,
                                                                                       config.network,
                                                                                       config.depth,
                                                                                       config.target_ratio,
                                                                                       iteration))
            torch.save(state, path)
            best_acc = test_acc
            best_epoch = epoch

    logger.info('Iteration [%d], best acc: %.4f, epoch: %d' %
                (iteration, best_acc, best_epoch))
    return 'Iteration [%d], best acc: %.4f, epoch: %d\n' % (iteration, best_acc, best_epoch), print_inf


def get_exception_layers(net, exception):
    exc = []
    idx = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if idx in exception:
                exc.append(m)
            idx += 1
    return tuple(exc)


def main(config):
    # init logger
    classes = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'tiny_imagenet': 200
    }
    logger, writer = init_logger(config)

    # torch config
    # torch.backends.cudnn.benchmark = True
    # print('Using cudnn.benchmark.')
    # build model
    model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', True))
    mask = None
    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()
    if mask is not None:
        mb.register_mask(mask)
        print_mask_information(mb, logger)

    # preprocessing
    # ====================================== get dataloader ======================================
    trainloader, testloader = get_dataloader(config.dataset, config.batch_size, 256, 4, root=config.dp)
    # ====================================== fetch configs ======================================
    ckpt_path = config.checkpoint_dir
    num_iterations = config.iterations
    target_ratio = config.target_ratio
    normalize = config.normalize
    # ====================================== fetch exception ======================================
    exception = get_exception_layers(mb.model, str_to_list(config.exception, ',', int))
    logger.info('Exception: ')

    for idx, m in enumerate(exception):
        logger.info('  (%d) %s' % (idx, m))

    # ====================================== fetch training schemes ======================================
    ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
    learning_rates = str_to_list(config.learning_rate, ',', float)
    weight_decays = str_to_list(config.weight_decay, ',', float)
    training_epochs = str_to_list(config.epoch, ',', int)
    logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
                (normalize, num_iterations, target_ratio, ratio))
    logger.info('Basic Settings: ')
    for idx in range(len(learning_rates)):
        logger.info('  %d: LR: %.5f, WD: %.5f, Epochs: %d' % (idx,
                                                              learning_rates[idx],
                                                              weight_decays[idx],
                                                              training_epochs[idx]))

    # ====================================== start pruning ======================================
    iteration = 0
    logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                ratio,
                                                                                1,
                                                                                num_iterations))

    pre_ratio = 1 - ((1 - ratio) * 2) if config.prune_mode_pa > 0 else ratio
    # pre_ratio = ratio
    mb.model.apply(weights_init)
    print("=> Applying weight initialization(%s)." % config.get('init_method', 'kaiming'))
    masks, masks_2, first_data = Panning(mb.model, pre_ratio, trainloader, 'cuda',
                                         num_classes=classes[config.dataset],
                                         samples_per_class=config.samples_per_class,
                                         num_iters=config.get('num_iters', 1),
                                         single_data=config.single_data,
                                         prune_mode=config.prune_mode,
                                         prune_conv=config.prune_conv,
                                         add_link=config.core_link,
                                         delete_link=config.core_link,
                                         enlarge=config.enlarge,
                                         prune_link=config.prune_link,
                                         # train_one=True
                                         )
    # # 用于得到不同数据计算出的mask
    # masks_2, masks__, first_data = Panning(mb.model, pre_ratio, trainloader, 'cuda',
    #                                        num_classes=classes[config.dataset],
    #                                        samples_per_class=config.samples_per_class,
    #                                        num_iters=config.get('num_iters', 1),
    #                                        reinit=False,
    #                                        single_data=config.single_data,
    #                                        prune_mode=config.prune_mode,
    #                                        prune_conv=config.prune_conv,
    #                                        add_link=config.core_link,
    #                                        delete_link=config.core_link,
    #                                        enlarge=config.enlarge,
    #                                        prune_link=config.prune_link,
    #                                        # first_data=first_data,
    #                                        )
    # ========== register mask ==================
    mb.register_mask(masks)

    # # 分析剪枝后同数据下的敏感情况
    # masks, _, __ = Panning(mb.model, ratio, trainloader, 'cuda',
    #                        num_classes=classes[config.dataset],
    #                        samples_per_class=config.samples_per_class,
    #                        num_iters=config.get('num_iters', 1),
    #                        reinit=False,
    #                        prune_mode=config.prune_mode,
    #                        prune_conv=config.prune_conv,
    #                        add_link=config.core_link,
    #                        delete_link=config.core_link,
    #                        enlarge=config.enlarge,
    #                        prune_link=config.prune_link,
    #                        first_masks=masks,
    #                        first_data=first_data,
    #                        # train_one=True
    #                        )
    # exit()

    # # 用于分析两次剪枝的筛选情况
    # _coincidence = []
    # _all_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in masks_2.values()]))
    # _and_mask_num = 0
    # for m, g in masks.items():
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         and_mask = masks[m] * masks_2[m]
    #         _now_mask_num = torch.sum(torch.cat([torch.flatten(and_mask == 1)]))
    #         _last_mask_num = torch.sum(torch.cat([torch.flatten(masks_2[m] == 1)]))
    #         print('-' * 20)
    #         print(f'_now_mask_num: {_now_mask_num}')
    #         print(f'_last_mask_num: {_last_mask_num}')
    #         print(f'/: {1 - _now_mask_num / _last_mask_num}')
    #         _coincidence.append(float(_now_mask_num / _last_mask_num))
    #         _and_mask_num += _now_mask_num
    # print(f'_all_mask_num: {_all_mask_num}')
    # print(f'_and_mask_num: {_and_mask_num}')
    # print(f'/: {1 - _and_mask_num / _all_mask_num}')
    # print('_coincidence', _coincidence)

    # ========== register mask ==================
    # mb.register_mask(masks)
    # ========== print pruning details ============
    print_inf = print_mask_information(mb, logger)
    config.send_mail_str += print_inf
    logger.info('  LR: %.5f, WD: %.5f, Epochs: %d' %
                (learning_rates[iteration], weight_decays[iteration], training_epochs[iteration]))
    config.send_mail_str += 'LR: %.5f, WD: %.5f, Epochs: %d, Batch: %d \n' % (
        learning_rates[iteration], weight_decays[iteration], training_epochs[iteration], config.batch_size)

    # ========== finetuning =======================
    tr_str, print_inf = train_once(mb=mb,
                                   net=mb.model,
                                   trainloader=trainloader,
                                   testloader=testloader,
                                   writer=writer,
                                   config=config,
                                   ckpt_path=ckpt_path,
                                   learning_rate=learning_rates[iteration],
                                   weight_decay=weight_decays[iteration],
                                   num_epochs=training_epochs[iteration],
                                   iteration=iteration,
                                   ratio=ratio,
                                   num_classes=classes[config.dataset],
                                   logger=logger,
                                   last_mask=masks_2,
                                   keep_masks=masks,
                                   lr_mode=config.lr_mode
                                   )

    config.send_mail_str += print_inf
    config.send_mail_str += tr_str

    QQmail = mail_log.MailLogs()
    QQmail.sendmail(config.send_mail_str, header=config.send_mail_head)


if __name__ == '__main__':
    config = init_config()
    main(config)
