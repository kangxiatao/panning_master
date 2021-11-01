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
import torch.nn.functional as F
import math
import numpy as np

import copy
import types


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
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
            num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True,
            prune_mode=3, prune_conv=False, add_link=False, delete_link=False, delete_conv=False, enlarge=False,
            prune_link=True):
    eps = 1e-10
    # print(f'ratio:{ratio}')
    keep_ratio = (1 - ratio)
    if enlarge:
        keep_ratio *= 2
    old_net = net

    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)
            # print(layer.weight.shape)
            # print(layer)

    inputs_one = []
    targets_one = []

    # 梯度
    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        # num_classes 个样本，每类样本取 samples_per_class 个
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        # 把数据分为前后两个list
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N // 2])
        targets_one.append(dtarget[:N // 2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net.forward(inputs[:N // 2]) / T
        loss = F.cross_entropy(outputs, targets[:N // 2])
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = net.forward(inputs[N // 2:]) / T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    # 梯度范数的梯度
    grad_l2 = None
    lam_q = 1 / len(inputs_one)
    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = net.forward(inputs) / T
        loss = F.cross_entropy(outputs, targets)

        gr_l2 = 0
        # torch.autograd.grad() 对指定参数求导
        # .torch.autograd.backward() 动态图所有参数的梯度都计算，叠加到 .grad 属性
        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                gr_l2 += torch.sum(grad_f[count].pow(2)) * lam_q
                count += 1
        z.backward(retain_graph=True)

        gr_l2.sqrt()
        if grad_l2 is None:
            grad_l2 = autograd.grad(gr_l2, weights, retain_graph=True)
        else:
            grad_l2 = [grad_l2[i] + autograd.grad(gr_l2, weights, retain_graph=True)[i] for i in range(len(grad_l2))]

    # === 剪枝部分 ===
    """
        prune_mode:
            1 snip
            2 grasp
            3 grasp+gradl2
            4 grasp+snip
    """

    # === 计算 ===
    # for debug
    grads_x = dict()
    grads_q = dict()
    grads_xq = dict()

    layer_cnt = 0
    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            x = -layer.weight.data * layer.weight.grad  # -theta_q Hg
            q = -layer.weight.data * grad_l2[layer_cnt]  # -theta_q grad_l2
            s = -torch.abs(layer.weight.data * grad_w[layer_cnt])  # -theta_q grad_w

            if prune_conv:
                # 卷积根据设定剪枝率按卷积核保留
                if isinstance(layer, nn.Conv2d):
                    # (n, c, k, k)
                    k1 = x.shape[2]
                    k2 = x.shape[3]
                    x = torch.sum(x, dim=(2, 3), keepdim=True)
                    q = torch.sum(q, dim=(2, 3), keepdim=True)
                    s = torch.sum(s, dim=(2, 3), keepdim=True)
                    x = x.repeat(1, 1, k1, k2)
                    q = q.repeat(1, 1, k1, k2)
                    s = s.repeat(1, 1, k1, k2)
                    # 卷积核取均值
                    x = torch.div(x, k1 * k2)
                    q = torch.div(q, k1 * k2)
                    s = torch.div(s, k1 * k2)

            if prune_mode == 1:  # snip
                x = s
            if prune_mode == 2:  # grasp
                pass
            if prune_mode == 3:  # 添加梯度范数
                x += q
            if prune_mode == 4:  # 添加SNIP
                x += s
            if prune_mode == 5:  # gl2_diff
                x -= q
            if prune_mode == 7:  # gl2_diff
                x = -torch.abs(q - x)

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

    num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
    threshold, _index = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)

    for m, g in grads.items():
        if prune_link:
            keep_masks[m] = torch.ones_like(g).float()
        else:
            keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

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

    # --- 补全卷积核，保证连通度 ---
    _add_mask_num = 0
    _pre_mask_num = 0
    _now_mask_num = 0
    if add_link:
        # for debug
        _add_grasp_value = []

        _get_connected_scores(f"{'-' * 20}\nBefore", 1)
        _pre_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()]))

        _pre_layer = 0
        _rep_layer = 1
        _pre_2d = np.sum(np.abs(keep_masks[grad_key[_pre_layer]].cpu().detach().numpy()), axis=(2, 3))
        _rep_2d = np.sum(np.abs(keep_masks[grad_key[_rep_layer]].cpu().detach().numpy()), axis=(2, 3))
        _pre_channel = np.sum(_pre_2d, axis=0)
        _pre_filter = np.sum(_pre_2d, axis=1)
        _rep_channel = np.sum(_rep_2d, axis=0)
        _rep_filter = np.sum(_rep_2d, axis=1)

        for _layer, _key in enumerate(grad_key):
            if isinstance(_key, nn.Conv2d) and 'padding' in str(_key):  # 不考虑resnet的shortcut部分
                # print(_key)
                if _layer >= 2:
                    # [n, c, k, k]
                    _2d = np.sum(np.abs(keep_masks[grad_key[_layer]].cpu().detach().numpy()), axis=(2, 3))
                    _channel = np.sum(_2d, axis=0)  # 向量
                    _filter = np.sum(_2d, axis=1)

                    # 通道和过滤器按排序补
                    for i in range(_rep_channel.shape[0]):  # 遍历通道
                        if _pre_filter[i] != 0 and _rep_channel[i] == 0:
                            temp = grads[grad_key[_rep_layer]][:, i]  # 这里的key值有bug，服了
                            temp = torch.mean(temp, dim=(1, 2))
                            top_k = 1
                            _scores, _index = torch.topk(temp, top_k, largest=False)
                            keep_masks[grad_key[_rep_layer]][_index[:top_k], i] = 1

                    for j in range(_rep_filter.shape[0]):  # 遍历过滤器
                        if _channel[j] != 0 and _rep_filter[j] == 0:
                            temp = grads[grad_key[_rep_layer]][j, :]  # 这里的key值有bug，服了
                            temp = torch.mean(temp, dim=(1, 2))
                            top_k = 1
                            _scores, _index = torch.topk(temp, top_k, largest=False)
                            keep_masks[grad_key[_rep_layer]][j, _index[:top_k]] = 1

                    # 重新计算
                    _pre_layer = _rep_layer
                    _rep_layer = _layer
                    _pre_2d = np.sum(np.abs(keep_masks[grad_key[_pre_layer]].cpu().detach().numpy()), axis=(2, 3))
                    _rep_2d = np.sum(np.abs(keep_masks[grad_key[_rep_layer]].cpu().detach().numpy()), axis=(2, 3))
                    _pre_channel = np.sum(_pre_2d, axis=0)
                    _pre_filter = np.sum(_pre_2d, axis=1)
                    _rep_channel = np.sum(_rep_2d, axis=0)
                    _rep_filter = np.sum(_rep_2d, axis=1)

            # 单独考虑线性层
            if isinstance(_key, nn.Linear):
                # [c, n]
                _linear = np.sum(np.abs(keep_masks[grad_key[_layer]].cpu().detach().numpy()), axis=0)

                # 通道和过滤器按排序补
                for i in range(_rep_channel.shape[0]):  # 遍历通道
                    if _pre_filter[i] != 0 and _rep_channel[i] == 0:
                        temp = grads[grad_key[_rep_layer]][:, i]  # 这里的key值有bug，服了
                        temp = torch.mean(temp, dim=(1, 2))
                        top_k = 1
                        _scores, _index = torch.topk(temp, top_k, largest=False)
                        keep_masks[grad_key[_rep_layer]][_index[:top_k], i] = 1

                for j in range(_rep_filter.shape[0]):  # 遍历过滤器
                    if _linear[j] != 0 and _rep_filter[j] == 0:
                        temp = grads[grad_key[_rep_layer]][j, :]  # 这里的key值有bug，服了
                        temp = torch.mean(temp, dim=(1, 2))
                        top_k = 1
                        _scores, _index = torch.topk(temp, top_k, largest=False)
                        keep_masks[grad_key[_rep_layer]][j, _index[:top_k]] = 1

        _now_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()]))
        print(f'_now_mask_num: {_now_mask_num}')
        _add_mask_num = _now_mask_num - _pre_mask_num
        print(f'_add_mask_num: {_add_mask_num}')
        _get_connected_scores(f"Add", 1)

    # 计算核链，并删除
    if prune_link:
        delete_link = True
        _add_mask_num = int(len(all_scores) * (1 - keep_ratio))
    if enlarge:
        _add_mask_num = _now_mask_num - _pre_mask_num / 2
        print("enlarge:", _add_mask_num)
    if delete_link and _add_mask_num > 0:
        # 计算核链分值和神经元数量
        _conv_link_score = {}
        _conv_link_num = {}
        _pre_layer = 0
        _pre_2d = np.sum(np.abs(grads[grad_key[_pre_layer]].cpu().detach().numpy()), axis=(2, 3))
        _pre_channel = np.sum(_pre_2d, axis=0)
        _pre_filter = np.sum(_pre_2d, axis=1)
        for _layer, _key in enumerate(grad_key):
            if isinstance(_key, nn.Conv2d) and 'padding' in str(_key):  # 不考虑resnet的shortcut部分
                # print(_key)
                if _layer >= 1:
                    # [n, c, k, k]
                    _conv_link_score[_key] = torch.mean(grads[grad_key[_layer]] * keep_masks[grad_key[_layer]], dim=(0, 2, 3)) + \
                                             torch.mean(grads[grad_key[_pre_layer]] * keep_masks[grad_key[_pre_layer]], dim=(1, 2, 3))
                    _conv_link_num[_key] = torch.sum(keep_masks[grad_key[_layer]], dim=(0, 2, 3)) + \
                                           torch.sum(keep_masks[grad_key[_pre_layer]], dim=(1, 2, 3))
                    # print(torch.mean(_conv_link_score[_key]))
                    _pre_layer = _layer
                else:
                    _conv_link_score[_key] = None
                    _conv_link_num[_key] = None
            # 单独考虑线性层
            elif isinstance(_key, nn.Linear):
                # [c, n]
                _conv_link_score[_key] = torch.mean(grads[grad_key[_layer]] * keep_masks[grad_key[_layer]], dim=0) + \
                                         torch.mean(grads[grad_key[_pre_layer]] * keep_masks[grad_key[_pre_layer]],
                                                    dim=(1, 2, 3))
                _conv_link_num[_key] = torch.sum(keep_masks[grad_key[_layer]], dim=0) + \
                                       torch.sum(keep_masks[grad_key[_pre_layer]], dim=(1, 2, 3))
            else:
                _conv_link_score[_key] = None
                _conv_link_num[_key] = None

        # 按分值排序
        _link_all_scores = None
        _link_all_num = None
        for x in _conv_link_score.values():
            if x is not None:
                if _link_all_scores is not None:
                    _link_all_scores = torch.cat([_link_all_scores, torch.flatten(x)])
                else:
                    _link_all_scores = torch.flatten(x)
        for x in _conv_link_num.values():
            if x is not None:
                if _link_all_num is not None:
                    _link_all_num = torch.cat([_link_all_num, torch.flatten(x)])
                else:
                    _link_all_num = torch.flatten(x)
        _link_all_scores, _link_all_scores_index = torch.sort(_link_all_scores, descending=True)
        _link_all_num = _link_all_num[_link_all_scores_index]

        # 得到要删除的链数
        _top = 0
        _delete_num = 0
        for _cnt, _link_s in enumerate(_link_all_scores):
            _delete_num += _link_all_num[_cnt]
            if _add_mask_num <= _delete_num:
                _top = _cnt
                break

        _threshold = _link_all_scores[_top]
        print(f'_top: {_top}')
        print(f'_threshold: {_threshold}')

        _pre_layer = 0
        for _layer, _key in enumerate(_conv_link_score):
            if isinstance(_key, nn.Conv2d) and 'padding' in str(_key):  # 不考虑resnet的shortcut部分
                if _layer >= 1:
                    for _ch, _link_s in enumerate(_conv_link_score[_key]):
                        if _link_s > _threshold:
                            keep_masks[grad_key[_layer]][:, _ch] = 0
                            keep_masks[grad_key[_pre_layer]][_ch, :] = 0
                    _pre_layer = _layer
            # 单独考虑线性层
            elif isinstance(_key, nn.Linear):
                for _ch, _link_s in enumerate(_conv_link_score[_key]):
                    if _link_s > _threshold:
                        keep_masks[grad_key[_layer]][:, _ch] = 0
                        keep_masks[grad_key[_pre_layer]][_ch, :] = 0

        _now_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()]))
        print(f'_now_mask_num: {_now_mask_num}')
        _get_connected_scores(f"Del", 1)

    # 补完可以选择删一次
    if delete_conv:
        _last_filter = None
        for m, g in keep_masks.items():
            if isinstance(m, nn.Conv2d) and 'padding' in str(m):  # 不考虑resnet的shortcut部分
                # [n, c, k, k]
                _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
                _channel = np.sum(_2d, axis=0)  # 向量

                if _last_filter is not None:  # 第一层不考虑
                    for i in range(_channel.shape[0]):  # 遍历通道过滤器
                        if _last_filter[i] == 0 and _channel[i] != 0:
                            keep_masks[m][:, i] = 0  # 去除整个通道

                _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))  # 修改之后重新计算
                _last_filter = np.sum(_2d, axis=1)

        _now_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()]))
        print(f'_now_mask_num: {_now_mask_num}')
        _get_connected_scores(f"Del", 1)

    return keep_masks
