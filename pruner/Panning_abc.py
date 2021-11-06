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
            num_classes=10, samples_per_class=15, num_iters=3, T=200, reinit=True,
            prune_mode=3, prune_conv=False, add_link=False, delete_link=False, delete_conv=False, enlarge=False,
            prune_link=False, debug_mode=False, debug_path='', debug_epoch=0):
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

    # 梯度
    for w in weights:
        w.requires_grad_(True)

    # -------- 这部分为直接求解，显存爆了 --------
    print("gradient => g")
    samples_per_class = 15
    inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
    N = inputs.shape[0]
    equal_parts = N // 3
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = net.forward(inputs[:equal_parts]) / T
    loss_a = F.cross_entropy(outputs, targets[:equal_parts])
    grad_a = autograd.grad(loss_a, weights, create_graph=True)
    outputs = net.forward(inputs[equal_parts:2*equal_parts]) / T
    loss_b = F.cross_entropy(outputs, targets[equal_parts:2*equal_parts])
    grad_b = autograd.grad(loss_b, weights, create_graph=True)
    outputs = net.forward(inputs[2*equal_parts:]) / T
    loss_c = F.cross_entropy(outputs, targets[2*equal_parts:])
    grad_c = autograd.grad(loss_c, weights, create_graph=True)

    print("gradient of norm gradient =》 Hg")
    gab = 0
    gbc = 0
    gac = 0
    # gla = 0
    # glb = 0
    # glc = 0
    _layer = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # print(torch.mean(grad_a[_layer]), torch.mean(grad_b[_layer]), torch.mean(grad_c[_layer]))
            gab += (grad_a[_layer] * grad_b[_layer]).sum()  # ga * gb
            gbc += (grad_c[_layer] * grad_b[_layer]).sum()  # gb * gc
            gac += (grad_a[_layer] * grad_c[_layer]).sum()  # ga * gc
            # grad_l = grad_a[_layer]+grad_b[_layer]+grad_c[_layer]
            # gla += (grad_l * grad_a[_layer]).sum()  # gl * ga
            # glb += (grad_l * grad_b[_layer]).sum()  # gl * gb
            # glc += (grad_l * grad_c[_layer]).sum()  # gl * gc
            _layer += 1
    print(gab, gbc, gac)
    # print(gla, glb, glc)
    grad_gab = autograd.grad(gab, weights, retain_graph=True)
    grad_gbc = autograd.grad(gbc, weights, retain_graph=True)
    grad_gac = autograd.grad(gac, weights, retain_graph=True)
    # grad_gla = autograd.grad(gla, weights, retain_graph=True)
    # grad_glb = autograd.grad(glb, weights, retain_graph=True)
    # grad_glc = autograd.grad(glc, weights, retain_graph=True)

    # -------- 优化，经典的时间换内存，但是有bug，🤮 --------
    # print("gradient => g and Hg")
    # samples_per_class = 15
    # num_iters = 3  # 三等分
    # equal_parts = samples_per_class // num_iters
    # inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
    # inputs_one = []
    # targets_one = []
    # for i in range(num_iters):
    #     inputs_one.append(inputs[equal_parts*i:equal_parts*(i+1)])
    #     targets_one.append(targets[equal_parts*i:equal_parts*(i+1)])
    #
    # # n等份的差值分数
    # grad_gg = {}

    # _cnt = 0
    # for i in range(num_iters):
    #     for j in range(i+1, num_iters):
    #         print('=> ', i, j)
    #         x_a = inputs_one[i].to(device)
    #         y_a = targets_one[i].to(device)
    #         out_a = net.forward(x_a) / T
    #         loss_a = F.cross_entropy(out_a, y_a)
    #         grad_a = autograd.grad(loss_a, weights, create_graph=True)
    #         x_b = inputs_one[j].to(device)
    #         y_b = targets_one[j].to(device)
    #         out_b = net.forward(x_b) / T
    #         loss_b = F.cross_entropy(out_b, y_b)
    #         grad_b = autograd.grad(loss_b, weights, create_graph=True)
    #
    #         gab = 0
    #         _layer = 0
    #         for layer in net.modules():
    #             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #                 # print(torch.mean(grad_a[_layer]), torch.mean(grad_b[_layer]))
    #                 gab += (grad_a[_layer] * grad_b[_layer]).sum()  # ga * gb
    #                 _layer += 1
    #         print(gab)
    #         grad_gg[_cnt] = autograd.grad(gab, weights)
    #         _cnt += 1
    # print(torch.mean(grad_gg[0][10]), torch.mean(grad_gg[1][10]), torch.mean(grad_gg[2][10]))
    # print(grad_gg[0][0].shape)

    # === 剪枝部分 ===
    """
        prune_mode:
            1 - 和最大值
            2 - 和最小值 （分层比较均衡）
            3 - 绝对值和
            4 - 和绝对值
            5 - 依据和得到网络比例，按绝对值和修剪
            6 - 取一组的绝对值
        debug_mode:
            1 - 梯度相似敏感度图
            2 - 梯度相似图（g1,g2,g3）
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
            a = layer.weight.data * grad_gab[layer_cnt]  # theta_q grad_l2
            b = layer.weight.data * grad_gbc[layer_cnt]  # theta_q grad_l2
            c = layer.weight.data * grad_gac[layer_cnt]  # theta_q grad_l2
            # a = layer.weight.data * grad_gg[0][layer_cnt]  # theta_q grad_l2
            # b = layer.weight.data * grad_gg[1][layer_cnt]  # theta_q grad_l2
            # c = layer.weight.data * grad_gg[2][layer_cnt]  # theta_q grad_l2
            # a = layer.weight.data * grad_gla[layer_cnt]  # theta_q grad_l2
            # b = layer.weight.data * grad_glb[layer_cnt]  # theta_q grad_l2
            # c = layer.weight.data * grad_glc[layer_cnt]  # theta_q grad_l2

            x = a
            if prune_mode == 1:
                x = a + b + c
            if prune_mode == 2:
                x = -(a + b + c)
            if prune_mode == 3:
                x = torch.abs(a) + torch.abs(b) + torch.abs(c)
            if prune_mode == 4:
                x = torch.abs(a + b + c)
            if prune_mode == 5:
                x = torch.abs(a) + torch.abs(b) + torch.abs(c)
            if prune_mode == 6:
                x = torch.abs(a)

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
            # --- for debug ---
            if debug_mode == 1:
                grads_a[old_modules[idx]] = torch.abs(a)
                grads_b[old_modules[idx]] = torch.abs(b)
                grads_c[old_modules[idx]] = torch.abs(c)
            if debug_mode == 2:
                grad_l = grad_a[layer_cnt] + grad_b[layer_cnt] + grad_c[layer_cnt]
                grads_a[old_modules[idx]] = grad_l*grad_a[layer_cnt]
                grads_b[old_modules[idx]] = grad_l*grad_b[layer_cnt]
                grads_c[old_modules[idx]] = grad_l*grad_c[layer_cnt]
            # grads_a[old_modules[idx]] = torch.abs(layer.weight.data * grad_gaba[layer_cnt])
            # grads_b[old_modules[idx]] = torch.abs(layer.weight.data * grad_gabb[layer_cnt])
            # grads_c[old_modules[idx]] = torch.abs(layer.weight.data * grad_gaba[layer_cnt]) + torch.abs(layer.weight.data * grad_gabb[layer_cnt])
            # -----------------
            if prune_mode == 5:
                grads_prune[old_modules[idx]] = -(a + b + c)

            layer_cnt += 1

    # === 根据重要度确定masks ===
    keep_masks = dict()
    if prune_mode == 5:
        # 得到剪枝比例
        ratio_layer = []
        prune_masks = dict()
        all_scores = torch.cat([torch.flatten(x) for x in grads_prune.values()])
        norm_factor = torch.abs(torch.sum(all_scores)) + eps
        print("** norm factor:", norm_factor)
        all_scores.div_(norm_factor)
        num_params_to_rm = int(len(all_scores) * keep_ratio)
        threshold, _index = torch.topk(all_scores, num_params_to_rm)
        acceptable_score = threshold[-1]
        print('** accept: ', acceptable_score)
        for m, g in grads_prune.items():
            prune_masks[m] = ((g / norm_factor) >= acceptable_score).float()
        print('Remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in prune_masks.values()])))
        for m, g in prune_masks.items():
            remain_num = torch.sum(torch.flatten(g == 1))
            delete_num = torch.sum(torch.flatten(g == 0))
            all_num = remain_num + delete_num
            ratio = float(remain_num) / float(all_num)
            # print(all_num, remain_num, ratio)
            ratio_layer.append(ratio)

        # 按比例修剪
        _cnt = 0
        for m, g in grads.items():
            _scores = torch.cat([torch.flatten(g)])
            _norm_factor = torch.abs(torch.sum(_scores)) + eps
            _scores.div_(_norm_factor)
            _num_to_rm = int(len(_scores) * ratio_layer[_cnt])
            _thr, _ = torch.topk(_scores, _num_to_rm, sorted=True)
            _acce_score = _thr[-1]
            # print('** ({}) accept: '.format(_cnt), _acce_score)
            keep_masks[m] = ((g / _norm_factor) >= _acce_score).float()
            _cnt += 1
    else:
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

    # ============== debug ==============
    debug = False
    if debug or debug_mode > 0:
        if debug_mode == 1:
            # 得到剪枝比例
            ratio_layer_a = []
            ratio_layer_b = []
            ratio_layer_c = []
            prune_a = dict()
            prune_b = dict()
            prune_c = dict()

            all_scores = torch.cat([torch.flatten(x) for x in grads_a.values()])
            norm_factor = torch.abs(torch.sum(all_scores)) + eps
            all_scores.div_(norm_factor)
            num_params_to_rm = int(len(all_scores) * keep_ratio)
            threshold, _index = torch.topk(all_scores, num_params_to_rm)
            acceptable_score = threshold[-1]
            for m, g in grads_a.items():
                prune_a[m] = ((g / norm_factor) >= acceptable_score).float()
            for m, g in prune_a.items():
                remain_num = torch.sum(torch.flatten(g == 1))
                delete_num = torch.sum(torch.flatten(g == 0))
                all_num = remain_num + delete_num
                ratio = float(remain_num) / float(all_num)
                ratio_layer_a.append(ratio)

            all_scores = torch.cat([torch.flatten(x) for x in grads_b.values()])
            norm_factor = torch.abs(torch.sum(all_scores)) + eps
            all_scores.div_(norm_factor)
            num_params_to_rm = int(len(all_scores) * keep_ratio)
            threshold, _index = torch.topk(all_scores, num_params_to_rm)
            acceptable_score = threshold[-1]
            for m, g in grads_b.items():
                prune_b[m] = ((g / norm_factor) >= acceptable_score).float()
            for m, g in prune_b.items():
                remain_num = torch.sum(torch.flatten(g == 1))
                delete_num = torch.sum(torch.flatten(g == 0))
                all_num = remain_num + delete_num
                ratio = float(remain_num) / float(all_num)
                ratio_layer_b.append(ratio)

            all_scores = torch.cat([torch.flatten(x) for x in grads_c.values()])
            norm_factor = torch.abs(torch.sum(all_scores)) + eps
            all_scores.div_(norm_factor)
            num_params_to_rm = int(len(all_scores) * keep_ratio)
            threshold, _index = torch.topk(all_scores, num_params_to_rm)
            acceptable_score = threshold[-1]
            for m, g in grads_c.items():
                prune_c[m] = ((g / norm_factor) >= acceptable_score).float()
            for m, g in prune_c.items():
                remain_num = torch.sum(torch.flatten(g == 1))
                delete_num = torch.sum(torch.flatten(g == 0))
                all_num = remain_num + delete_num
                ratio = float(remain_num) / float(all_num)
                ratio_layer_c.append(ratio)
            print(ratio_layer_a)
            print(ratio_layer_b)
            print(ratio_layer_c)

            user_layer = 3  # 41.75%
            # user_layer = 0
            for _layer, _key in enumerate(grad_key):
                if _layer == user_layer:
                    import matplotlib.pyplot as plt
                    from matplotlib import cm
                    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

                    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

                    dpi = 60
                    xpixels = 1200
                    ypixels = 1200
                    xinch = xpixels / dpi
                    yinch = ypixels / dpi
                    fig = plt.figure(figsize=(xinch,yinch))
                    ax = fig.add_subplot(111, projection='3d')

                    # 分值转numpy
                    _xa = torch.cat([torch.flatten(grads_a[grad_key[_layer]])])
                    _yb = torch.cat([torch.flatten(grads_b[grad_key[_layer]])])
                    _zc = torch.cat([torch.flatten(grads_c[grad_key[_layer]])])
                    # _np_x = _xa.cpu().detach().numpy()
                    # _np_y = _yb.cpu().detach().numpy()
                    # _np_z = _zc.cpu().detach().numpy()
                    _np_x = np.log(_xa.cpu().detach().numpy())
                    _np_y = np.log(_yb.cpu().detach().numpy())
                    _np_z = np.log(_zc.cpu().detach().numpy())
                    # 画出全部点
                    # ax.scatter(xs=_np_x, ys=_np_y, zs=_np_z, c='ivory', s=1, alpha=0.3, marker='.')
                    ax.scatter(xs=_np_x, ys=_np_y, zs=_np_z, c='black', s=1, alpha=0.3, marker='.')
                    # ax.scatter(xs=_np_x, ys=_np_y, zs=_np_z, c='coral', s=1, alpha=0.3, marker='.')

                    # 取出剪枝保留的权重分值
                    _g_len = len(_np_x)
                    _pr_index_a = np.argsort(_np_x)[int(_g_len * ratio_layer_a[user_layer]):]
                    _np_pr_ax = _np_x[_pr_index_a]
                    _np_pr_ay = _np_y[_pr_index_a]
                    _np_pr_az = _np_z[_pr_index_a]
                    _pr_index_b = np.argsort(_np_y)[int(_g_len * ratio_layer_b[user_layer]):]
                    _np_pr_bx = _np_x[_pr_index_b]
                    _np_pr_by = _np_y[_pr_index_b]
                    _np_pr_bz = _np_z[_pr_index_b]
                    _pr_index_c = np.argsort(_np_z)[int(_g_len * ratio_layer_c[user_layer]):]
                    _np_pr_cx = _np_x[_pr_index_c]
                    _np_pr_cy = _np_y[_pr_index_c]
                    _np_pr_cz = _np_z[_pr_index_c]
                    # # 画出保留点
                    ax.scatter(xs=_np_pr_ax, ys=_np_pr_ay, zs=_np_pr_az, c='gold', s=1, alpha=0.3, marker='o')
                    ax.scatter(xs=_np_pr_bx, ys=_np_pr_by, zs=_np_pr_bz, c='mediumvioletred', s=1, alpha=0.3, marker='+')
                    ax.scatter(xs=_np_pr_cx, ys=_np_pr_cy, zs=_np_pr_cz, c='skyblue', s=1, alpha=0.3, marker='*')

                    # _lim = np.mean(_np_x) * 5
                    # ax.set_xlim([0, _lim])
                    # ax.set_ylim([0, _lim])
                    # ax.set_zlim([0, _lim])
                    plt.savefig(debug_path + 'epoch_' + str(debug_epoch) + '.png', dpi = dpi)
                    plt.show()

        if debug_mode == 2:
            user_layer = 3  # 41.75%
            # user_layer = 0
            for _layer, _key in enumerate(grad_key):
                if _layer == user_layer:
                    import matplotlib.pyplot as plt
                    from matplotlib import cm
                    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

                    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

                    dpi = 60
                    xpixels = 1200
                    ypixels = 1200
                    xinch = xpixels / dpi
                    yinch = ypixels / dpi
                    fig = plt.figure(figsize=(xinch,yinch))
                    ax = fig.add_subplot(111, projection='3d')

                    # 分值转numpy
                    _xa = torch.cat([torch.flatten(grads_a[grad_key[_layer]])])
                    _yb = torch.cat([torch.flatten(grads_b[grad_key[_layer]])])
                    _zc = torch.cat([torch.flatten(grads_c[grad_key[_layer]])])
                    _np_x = _xa.cpu().detach().numpy()
                    _np_y = _yb.cpu().detach().numpy()
                    _np_z = _zc.cpu().detach().numpy()
                    # _np_x = np.log(_xa.cpu().detach().numpy())
                    # _np_y = np.log(_yb.cpu().detach().numpy())
                    # _np_z = np.log(_zc.cpu().detach().numpy())
                    # 画出全部点
                    # ax.scatter(xs=_np_x, ys=_np_y, zs=_np_z, c='ivory', s=1, alpha=0.3, marker='.')
                    ax.scatter(xs=_np_x, ys=_np_y, zs=_np_z, c='mediumvioletred', s=1, alpha=0.3, marker='.')
                    # ax.scatter(xs=_np_x, ys=_np_y, zs=_np_z, c='coral', s=1, alpha=0.3, marker='.')

                    # _lim = np.mean(_np_x) * 5
                    # ax.set_xlim([0, _lim])
                    # ax.set_ylim([0, _lim])
                    # ax.set_zlim([0, _lim])
                    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
                    plt.savefig(debug_path + 'epoch_' + str(debug_epoch) + '.png', dpi=dpi)
                    plt.show()

    # =====================================

    return keep_masks
