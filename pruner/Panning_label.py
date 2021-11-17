# -*- coding: utf-8 -*-

"""
Created on 09/19/2021
Panning.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""
# -*- coding: utf-8 -*-


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

import copy
import types


def fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y


def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total


def Panning(net, ratio, train_dataloader, device,
            num_classes=10, samples_per_class=10, T=200, reinit=True,
            data_mode=0, grad_mode=0, prune_mode=4, num_group=None, debug_mode=False):
    eps = 1e-10
    keep_ratio = (1 - ratio)
    old_net = net
    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)

    print("fetch data")
    samples_per_class = 10
    inputs, targets = fetch_data(train_dataloader, num_classes, samples_per_class)
    N = inputs.shape[0]
    if num_group is None:
        if data_mode == 0:
            num_group = 2 if num_classes == 10 else 5
        else:
            num_group = 2
    equal_parts = N // num_group
    if data_mode == 0:
        # # 同标签组排列 pass
        # targets, _index = torch.sort(targets)
        # inputs = inputs[_index]
        pass
    else:
        # 不同标签组排列
        _index = []
        for i in range(num_classes):
            _index.extend([i + j for j in range(0, samples_per_class * num_classes, num_classes)])
        inputs = inputs[_index]
        targets = targets[_index]
    print(targets)
    inputs = inputs.to(device)
    targets = targets.to(device)
    print("gradient => g")
    gradg_list = []

    if grad_mode == 0:
        for i in range(num_group):
            _outputs = net.forward(inputs[i*equal_parts:(i+1)*equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i*equal_parts:(i+1)*equal_parts])
            _grad = autograd.grad(_loss, weights, create_graph=True)
            _gz = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    _gz += _grad[_layer].pow(2).sum()  # g * g
                    _layer += 1
            gradg_list.append(autograd.grad(_gz, weights))
    elif grad_mode == 1:
        _grad_ls = []
        for i in range(num_group):
            _outputs = net.forward(inputs[i*equal_parts:(i+1)*equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i*equal_parts:(i+1)*equal_parts])
            _grad_ls.append(autograd.grad(_loss, weights, create_graph=True))
        _grad_and = []
        _layer = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _gand = 0
                for i in range(num_group):
                    _gand += _grad_ls[i][_layer]
                _grad_and.append(_gand)
                _layer += 1
        for i in range(num_group):
            _gz = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    _gz += (_grad_and[_layer]*_grad_ls[i][_layer]).sum()  # ga * gn
                    _layer += 1
            gradg_list.append(autograd.grad(_gz, weights, retain_graph=True))
    else:
        _grad_ls = []
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad_ls.append(autograd.grad(_loss, weights, create_graph=True))

        for i in range(num_group):
            for j in range(i+1, num_group):
                _gz = 0
                _layer = 0
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        _gz += (_grad_ls[i][_layer] * _grad_ls[j][_layer]).sum()  # g1 * g2
                        _layer += 1
                gradg_list.append(autograd.grad(_gz, weights, retain_graph=True))

    # === 剪枝部分 ===
    """
        data_mode:
            0 - 不同标签分组
            1 - 同标签分组
        gard_mode:
            0 - 梯度范数梯度
            1 - 不同组点积（相似度）梯度
            2 - 不同组对应点积
        prune_mode:
            1 - 绝对值和
            2 - 和绝对值
            3 - 乘积
            4 - 欧氏距离
    """

    # === 计算分值 ===
    layer_cnt = 0
    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            kxt = 0
            if prune_mode == 1:
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt += torch.abs(_qhg)
                #     print(torch.mean(_qhg), torch.sum(_qhg))
                # print('-' * 20)

            if prune_mode == 2:
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt += _qhg
                kxt = torch.abs(kxt)

            if prune_mode == 3:
                kxt = 1e6  # 约等于超参，估计值，kxt是👴
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt *= torch.abs(_qhg)  # 最后线性层有bug，？，不解
                    # print(torch.mean(torch.abs(_qhg)), torch.sum(torch.abs(_qhg)))
                # print('-' * 20)

            if prune_mode == 4:
                aef = 1e6  # 约等于超参，估计值
                for i in range(len(gradg_list)):
                    _qhg = layer.weight.data * gradg_list[i][layer_cnt] * aef  # theta_q grad
                    kxt += _qhg.pow(2)
                kxt = kxt.sqrt()

            # if prune_mode == 5:
            #     _qhg = []
            #     for i in range(len(gradg_list)):
            #         _qhg.append(layer.weight.data * gradg_list[i][layer_cnt])  # theta_q grad


            # 评估分数
            grads[old_modules[idx]] = kxt
            # print(torch.mean(kxt), torch.sum(kxt))
            # print('-' * 20)

            layer_cnt += 1

    # === 根据重要度确定masks ===
    keep_masks = dict()

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * keep_ratio)
    threshold, _index = torch.topk(all_scores, num_params_to_rm)
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)

    for m, g in grads.items():
        if prune_mode == 0:
            keep_masks[m] = torch.ones_like(g).float()
        else:
            # # 标签过大，全连接层不做修剪
            # if num_classes > 50:
            #     if isinstance(m, nn.Linear):
            #         keep_masks[m] = torch.ones_like(g).float()
            #     else:
            #         keep_masks[m] = ((g / norm_factor) >= acceptable_score).float()
            # else:
            keep_masks[m] = ((g / norm_factor) >= acceptable_score).float()

    print('Remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    grad_key = [x for x in grads.keys()]

    def _get_connected_scores(info='', mode=0):
        # 计算连通情况
        _connected_scores = 0
        _last_filter = None
        for m, g in keep_masks.items():
            if isinstance(m, nn.Conv2d) and 'padding' in str(m):  # 不考虑resnet的shortcut部分
                # [n, c, k, k]
                _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
                _channel = np.sum(_2d, axis=0)  # 向量

                if _last_filter is not None:  # 第一层不考虑
                    for i in range(_channel.shape[0]):  # 遍历通道过滤器
                        if _last_filter[i] == 0 and _channel[i] != 0:
                            _connected_scores += 1
                        if mode == 1:
                            if _last_filter[i] != 0 and _channel[i] == 0:
                                _connected_scores += 1

                _last_filter = np.sum(_2d, axis=1)

        print(f'{info}-{mode}->_connected_scores: {_connected_scores}')
        return _connected_scores

    _get_connected_scores(f"{'-' * 20}\nBefore", 1)

    return keep_masks
