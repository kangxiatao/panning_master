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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_base import ModelBase
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.base.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import get_network
from pruner.Panning_label import Panning
from utils import mail_log


def init_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='configs/cifar10/resnet32/Panning_98.json')
    parser.add_argument('--config', type=str, default='configs/cifar100/vgg19/Panning_98.json')
    # parser.add_argument('--config', type=str, default='configs/mnist/lenet/Panning_90.json')
    parser.add_argument('--run', type=str, default='expsim1')
    parser.add_argument('--epoch', type=str, default='666')
    parser.add_argument('--prune_mode', type=int, default=3)
    parser.add_argument('--prune_mode_pa', type=int, default=0)  # 第二次修剪模式
    parser.add_argument('--data_mode', type=int, default=1)  # 数据模式
    parser.add_argument('--prune_conv', type=int, default=0)  # 修剪卷积核标志
    parser.add_argument('--prune_conv_pa', type=int, default=0)
    parser.add_argument('--core_link', type=int, default=0)  # 核链标志
    parser.add_argument('--enlarge', type=int, default=0)  # 扩张标志
    parser.add_argument('--prune_link', type=int, default=0)  # 按核链修剪
    parser.add_argument('--prune_epoch', type=int, default=0)  # 第二次修剪时间
    parser.add_argument('--remain', type=float, default=666)
    parser.add_argument('--debug', type=int, default=0)  # 调试标记（打印数据和作图等）
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
    config.data_mode = args.data_mode
    config.prune_conv = True if args.prune_conv == 1 else False
    config.prune_conv_pa = True if args.prune_conv_pa == 1 else False
    config.core_link = True if args.core_link == 1 else False
    config.enlarge = True if args.enlarge == 1 else False
    config.prune_link = True if args.prune_link == 1 else False
    # config.debug = True if args.debug == 1 else False
    config.prune_epoch = args.prune_epoch
    config.debug = args.debug
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


def count_FLOPs(mask, logger):
    logger.info('** Pruned FLOPs:')
    re_str = '** Pruned FLOPs:\n'
    total_2d_num = 0
    pruned_2d_num = 0
    count = 0
    for m, g in mask.items():
        # 按通道或过滤器运算
        if isinstance(m, nn.Conv2d):
            # [n, c, k, k]
            _shape = g.shape
            n = _shape[0]
            c = _shape[1]
            _2d = np.sum(np.abs(g.cpu().detach().numpy()), axis=(2, 3))
            _channel = np.sum(_2d, axis=0)
            _filter = np.sum(_2d, axis=1)
            _ci = np.count_nonzero(_channel == 0)
            _co = np.count_nonzero(_filter == 0)

            print('channel:', np.count_nonzero(_channel == 0))
            print('filter:', np.count_nonzero(_filter == 0))

            pruned_FLOPs = (_co * _ci) / (n * c)
            total_2d_num += n * c
            pruned_2d_num += _co * _ci

            logger.info('  (%d) %s: %.2f%%' % (count, m, pruned_FLOPs * 100))
            re_str += '  (%d) %.2f%%\n' % (count, pruned_FLOPs * 100)
            count += 1

        # 按卷积核运算
        # if isinstance(m, nn.Conv2d):
        #     # [n, c, k, k]
        #     _shape = mask[m].shape
        #     n = _shape[0]
        #     c = _shape[1]
        #     _2d = torch.sum(mask[m], dim=(2, 3))
        #     _pruned_2d = torch.sum(torch.flatten(_2d == 0))
        #     pruned_FLOPs = float(_pruned_2d) / (n * c)
        #     total_2d_num += n * c
        #     pruned_2d_num += _pruned_2d
        #
        #     logger.info('  (%d) %s: %.2f%%' % (count, m, pruned_FLOPs * 100))
        #     re_str += '  (%d) %.2f%%\n' % (count, pruned_FLOPs * 100)
        #     count += 1

        # elif isinstance(m, nn.Linear):
        #     # [c, n]
        #     c = mask[m].shape[0]
        #     n = mask[m].shape[1]
        #     _pruned_w = torch.sum(torch.flatten(mask[m] == 0))
        #     pruned_FLOPs = float(_pruned_w)/(n*c)
        #     total_2d_num += n*c
        #     pruned_2d_num += _pruned_w
        #
        #     logger.info('  (%d) %s: %.2f%%' % (count, m, pruned_FLOPs*100))
        #     re_str += '  (%d) %.2f%%\n' % (count, pruned_FLOPs*100)
        #     count += 1

    logger.info('=> Overall Pruned FLOPs: %.2f%%' % (pruned_2d_num*100/total_2d_num))
    re_str += '=> Overall Pruned FLOPs: %.2f%%\n' % (pruned_2d_num*100/total_2d_num)

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


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, iteration):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_last_lr(), 0, 0, correct, total))

    writer.add_scalar('iter_%d/train/lr' % iteration, lr_scheduler.get_last_lr(), epoch)

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

        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_last_lr(), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('iter_%d/train/loss' % iteration, train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('iter_%d/train/acc' % iteration, 100. * correct / total, epoch)


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
               iteration, ratio, num_classes, logger, prune_masks=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print_inf = ''
    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epochs):

        train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer, iteration=iteration)
        test_acc = test(net, testloader, criterion, epoch, writer, iteration)
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
    masks = Panning(mb.model, pre_ratio, trainloader, 'cuda',
                    num_classes=classes[config.dataset],
                    samples_per_class=config.samples_per_class,
                    num_iters=config.get('num_iters', 1),
                    prune_mode=config.prune_mode,
                    data_mode=config.data_mode,
                    prune_conv=config.prune_conv,
                    add_link=config.core_link,
                    delete_link=config.core_link,
                    enlarge=config.enlarge,
                    prune_link=config.prune_link,
                    debug_mode=config.debug,
                    debug_path=config.summary_dir
                    )
    # ========== register mask ==================
    mb.register_mask(masks)
    # ========== print pruning details ============
    print_inf = print_mask_information(mb, logger)
    config.send_mail_str += print_inf
    # print_inf = count_FLOPs(masks, logger)
    # config.send_mail_str += print_inf
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
                                   prune_masks=masks
                                   )

    config.send_mail_str += print_inf
    config.send_mail_str += tr_str

    QQmail = mail_log.MailLogs()
    QQmail.sendmail(config.send_mail_str, header=config.send_mail_head)


if __name__ == '__main__':
    config = init_config()
    main(config)
