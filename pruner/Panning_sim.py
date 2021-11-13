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
            num_classes=10, samples_per_class=10, num_iters=1, T=200, reinit=True, prune_masks=None,
            prune_mode=3, data_mode=0, prune_conv=False, add_link=False, delete_link=False, delete_conv=False,
            enlarge=False, prune_link=False, debug_mode=False, debug_path='', debug_epoch=0):
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

    # -------- 这部分为直接求解，显存爆了 --------
    print("fetch data")
    samples_per_class = 10
    inputs, targets = fetch_data(train_dataloader, num_classes, samples_per_class)
    N = inputs.shape[0]
    if data_mode == 0:
        equal_parts = N // 1
    elif data_mode == 1:
        equal_parts = N // 2
    else:
        equal_parts = N // 2
    inputs = inputs.to(device)
    targets = targets.to(device)
    print("gradient => g")
    outputs = net.forward(inputs[:equal_parts]) / T
    loss_a = F.cross_entropy(outputs, targets[:equal_parts])
    grad_a = autograd.grad(loss_a, weights, create_graph=True)
    print("gradient of norm gradient =》 Hg")
    gaa = 0
    _layer = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            gaa += grad_a[_layer].pow(2).sum()  # ga * ga
            _layer += 1
    grad_aa = autograd.grad(gaa, weights)
    # 更新参数
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    correct = 0
    total = 0
    optimizer.zero_grad()
    outputs = net(inputs[:equal_parts])
    loss = criterion(outputs, targets[:equal_parts])
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    _, predicted = outputs.max(1)
    total += targets[:equal_parts].size(0)
    correct += predicted.eq(targets[:equal_parts]).sum().item()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss, 100. * correct / total, correct, total))

    # 第二次推断
    weights_v2 = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights_v2.append(layer.weight)
    for w in weights_v2:
        w.requires_grad_(True)
    print("gradient => g")
    if data_mode == 0:
        outputs = net.forward(inputs) / T
        loss_b = F.cross_entropy(outputs, targets)
    else:
        outputs = net.forward(inputs[equal_parts:2*equal_parts]) / T
        loss_b = F.cross_entropy(outputs, targets[equal_parts:2*equal_parts])
    grad_b = autograd.grad(loss_b, weights_v2, create_graph=True)
    print("gradient of norm gradient =》 Hg")
    gbb = 0
    _layer = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            gbb += grad_b[_layer].pow(2).sum()  # gb * gb
            _layer += 1
    print(gaa, gbb)
    grad_bb = autograd.grad(gbb, weights_v2)

    # 重置w
    weights.clear()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)

    # === 剪枝部分 ===
    """
        prune_mode:
            1 - 差值最大
            2 - 差值最小
            3 - 差值绝对值
            4 - 差值归一化
            5 - 。。。
    """

    # === 计算分值 ===
    layer_cnt = 0
    grads = dict()
    # --- for debug ---
    grads_a = dict()
    grads_b = dict()
    grads_c = dict()
    # -----------------
    grads_prune = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            a = weights[layer_cnt] * grad_aa[layer_cnt]  # theta_q grad_aa
            b = layer.weight.data * grad_bb[layer_cnt]  # theta_q grad_bb
            # print(torch.mean(a), torch.sum(a))
            # print(torch.mean(b), torch.sum(b))
            # print(torch.mean(weights[layer_cnt]), torch.sum(weights[layer_cnt]))
            # print(torch.mean(layer.weight.data), torch.sum(layer.weight.data))
            # print('-' * 20)
            # print('0:', torch.sum(torch.cat([torch.flatten(layer.weight.data == 0)])))

            x = a
            if prune_mode == 1:
                x = a - b
            if prune_mode == 2:
                x = b - a
            if prune_mode == 3:
                x = torch.abs(a - b)
            if prune_mode == 4:
                x = torch.div(torch.abs(a - b), torch.abs(a) + torch.abs(b))
                x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
                x = torch.where(x == 1, torch.full_like(x, 0), x)
            if prune_mode == 5:
                x = torch.abs(a - b) * (torch.abs(a) + torch.abs(b))

            print(torch.max(torch.abs(x)))
            if prune_conv:
                # 卷积根据设定剪枝率按卷积核保留
                if isinstance(layer, nn.Conv2d):
                    # (n, c, k, k)
                    k1 = x.shape[2]
                    k2 = x.shape[3]
                    x = torch.sum(x, dim=(2, 3), keepdim=True)
                    x = x.repeat(1, 1, k1, k2)
                    # 卷积核取均值
                    x = torch.div(x, k1 * k2)

            # 评估分数
            grads[old_modules[idx]] = x

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
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)

    for m, g in grads.items():
        if prune_link or prune_mode == 0:
            keep_masks[m] = torch.ones_like(g).float()
        else:
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
