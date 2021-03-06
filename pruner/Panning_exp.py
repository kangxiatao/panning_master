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


def GraSP_fetch_data(dataloader, num_classes, samples_per_class, mode=0):
    if mode == 0:
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
    else:
        dataloader_iter = iter(dataloader)
        inputs, targets = next(dataloader_iter)
        X, y = inputs[0:samples_per_class*num_classes], targets[0:samples_per_class*num_classes]
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
            num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True, single_data=0,
            prune_mode=3, prune_conv=False, add_link=False, delete_link=False, delete_conv=False, enlarge=False,
            prune_link=False, first_masks=None, first_data=None, train_one=False):
    eps = 1e-10
    # print(f'ratio:{ratio}')
    keep_ratio = (1 - ratio)
    if enlarge:
        keep_ratio *= 2
    old_net = net

    # layer_key
    layer_key = None
    if first_masks:
        layer_key = [x for x in first_masks.keys()]

    net.zero_grad()
    net = copy.deepcopy(net)  # .eval()

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

    # ??????
    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    if first_data is None:
        for it in range(num_iters):
            print("(1): Iterations %d/%d." % (it, num_iters))
            # num_classes ??????????????????????????? samples_per_class ???
            inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class, single_data)
            N = inputs.shape[0]

            # ?????????????????????
            _index = []
            for i in range(samples_per_class):
                _index.extend([i + j * samples_per_class for j in range(0, num_classes)])
            inputs = inputs[_index]
            targets = targets[_index]
            # print(inputs[1])
            # print(inputs[1][1][1])
            print(targets[:num_classes])

            # ???????????????????????????list
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
            if first_masks:  # ??????????????????
                grad_w_p = [x*first_masks[layer_key[i]] for i, x in enumerate(grad_w_p)]
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

            outputs = net.forward(inputs[N // 2:]) / T
            loss = F.cross_entropy(outputs, targets[N // 2:])
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if first_masks:  # ??????????????????
                grad_w_p = [x*first_masks[layer_key[i]] for i, x in enumerate(grad_w_p)]
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]
    else:
        inputs_one, targets_one = first_data
        # print(inputs_one[0][1][1][1])
        # print(inputs_one[0][1].shape)
        # print(targets_one[0][:num_classes])
        for it in range(num_iters):
            print("(1): Iterations %d/%d." % (it, num_iters))
            # num_classes ??????????????????????????? samples_per_class ???
            inputs = inputs_one[it].to(device)
            targets = targets_one[it].to(device)
            outputs = net.forward(inputs) / T
            loss = F.cross_entropy(outputs, targets)
            grad_w_p = autograd.grad(loss, weights)
            if first_masks:  # ??????????????????
                grad_w_p = [x * first_masks[layer_key[i]] for i, x in enumerate(grad_w_p)]
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

            inputs = inputs_one[it+1].to(device)
            targets = targets_one[it+1].to(device)
            outputs = net.forward(inputs) / T
            loss = F.cross_entropy(outputs, targets)
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if first_masks:  # ??????????????????
                grad_w_p = [x * first_masks[layer_key[i]] for i, x in enumerate(grad_w_p)]
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

    # ?????????????????????
    # all_g1g2 = 0  # g1 * g1 and g2 * g2
    last_grad_f = 0  # for g1 g2
    grad_diff = None
    grad_l2 = None
    lam_q = 1 / len(inputs_one)
    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        # inputs = inputs_one.pop(0).to(device)
        # targets = targets_one.pop(0).to(device)
        inputs = inputs_one[it].to(device)
        targets = targets_one[it].to(device)
        outputs = net.forward(inputs) / T
        loss = F.cross_entropy(outputs, targets)
        print(loss)
        gr_l2 = 0
        g1_g2 = 0
        # torch.autograd.grad() ?????????????????????
        # .torch.autograd.backward() ??????????????????????????????????????????????????? .grad ??????
        grad_f = autograd.grad(loss, weights, create_graph=True)
        if first_masks:  # ??????????????????
            grad_f = [x*first_masks[layer_key[i]] for i, x in enumerate(grad_f)]
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                gr_l2 += torch.sum(grad_f[count].pow(2)) * lam_q
                if it % 2 == 1:
                    g1_g2 += torch.sum(grad_f[count] * last_grad_f[count])  # g1 * g2
                # ????????????debug
                # print(torch.mean(grad_w[count].data), torch.mean(grad_f[count]))
                # print(torch.mean(grad_w_p[count]), torch.mean(grad_f[count]))
                count += 1
        z.backward(retain_graph=True)
        if it % 2 == 0:
            last_grad_f = grad_f
        if grad_l2 is None:
            grad_l2 = autograd.grad(gr_l2, weights, retain_graph=True)
        else:
            grad_l2 = [grad_l2[i] + autograd.grad(gr_l2, weights, retain_graph=True)[i] for i in range(len(grad_l2))]
        if it % 2 == 1:
            if grad_diff is None:
                grad_diff = autograd.grad(g1_g2, weights, retain_graph=True)
            else:
                grad_diff = [grad_diff[i] + autograd.grad(g1_g2, weights, retain_graph=True)[i] for i in range(len(grad_diff))]

        print(z, gr_l2, g1_g2)

    # === ???????????? ===
    """
        prune_mode:
            1 snip
            2 grasp
            3 grasp+gradl2
            4 gl2_diff
            5 abs_diff
            6 ratio_layer
    """

    # === ?????? ===
    # for debug
    grads_p = dict()  # grasp
    grads_l = dict()  # gl2
    grads_s = dict()  # snip
    grads_d = dict()  # gl2_ diff
    grads_pl = dict()  # grasp + gl2

    layer_cnt = 0
    grads = dict()
    grads_prune = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            p = -layer.weight.data * layer.weight.grad  # -theta_q Hg
            l = -layer.weight.data * grad_l2[layer_cnt]  # -theta_q grad_l2
            s = -torch.abs(layer.weight.data * grad_w[layer_cnt])  # -theta_q grad_w
            d = layer.weight.data * grad_diff[layer_cnt]  # theta_q grad_diff  d = x - q

            if prune_conv:
                # ?????????????????????????????????????????????
                if isinstance(layer, nn.Conv2d):
                    # (n, c, k, k)
                    k1 = p.shape[2]
                    k2 = p.shape[3]
                    p = torch.sum(p, dim=(2, 3), keepdim=True)
                    l = torch.sum(l, dim=(2, 3), keepdim=True)
                    s = torch.sum(s, dim=(2, 3), keepdim=True)
                    p = p.repeat(1, 1, k1, k2)
                    l = l.repeat(1, 1, k1, k2)
                    s = s.repeat(1, 1, k1, k2)
                    # ??????????????????
                    p = torch.div(p, k1 * k2)
                    l = torch.div(l, k1 * k2)
                    s = torch.div(s, k1 * k2)

            # ============== debug ==============
            debug = True
            if debug:
                # grads_p[old_modules[idx]] = -torch.abs(p)
                grads_p[old_modules[idx]] = p
                grads_l[old_modules[idx]] = l
                grads_s[old_modules[idx]] = s
                # grads_l[old_modules[idx]] = l
                grads_d[old_modules[idx]] = p - l
                grads_pl[old_modules[idx]] = p + l

                # # ???????????????
                # print(torch.mean(p), torch.sum(p))
                # print(torch.mean(l), torch.sum(l))
                # print(torch.mean(p-l), torch.sum(p-l))
                # print(torch.mean(d), torch.sum(d))
                # print('-' * 20)
                # ??????????????? 49 46 43 40 36 33 30 27 23 20 17 14 10 7 3 0
                use_layer = 666
                if layer_cnt == use_layer:
                    import matplotlib.pyplot as plt
                    plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????????????????
                    plt.rcParams['axes.unicode_minus'] = False  # ????????????????????????
                    plt.figure(1)
                    # ?????????????????????
                    gra_gl2, ind = torch.sort(torch.cat([torch.flatten(p + l)]), descending=True)
                    gra = torch.cat([torch.flatten(p)])
                    gl2 = torch.cat([torch.flatten(l)])
                    np_gra_gl2 = gra_gl2.cpu().detach().numpy()
                    np_gra = gra[ind].cpu().detach().numpy()
                    np_gl2 = gl2[ind].cpu().detach().numpy()
                    zero_ind = np.argwhere(np_gra_gl2 > 0)[-1]
                    plt.plot(range(1, len(gra) + 1, 1), np_gra_gl2, color='violet', label="Gra+SNIP")
                    plt.scatter(range(1, len(gra) + 1, 1), np_gra, color='cornflowerblue', s=1, label="Gra")
                    plt.scatter(range(1, len(gra) + 1, 1), np_gl2, color='rosybrown', s=1, label="SNIP")
                    plt.axvline(zero_ind, color='green', linestyle=':')
                    # plt.axvline(len(gra)*0.98, color='darkorchid', linestyle=':')
                    plt.legend()
                    plt.show()

                    plt.figure(2)
                    plt.plot(range(1, len(gra) + 1, 1), np_gra_gl2, color='violet', label="Gra+SNIP")
                    plt.scatter(range(1, len(gra) + 1, 1), np_gra, color='cornflowerblue', s=1, label="Gra")
                    # plt.scatter(range(1, len(gra) + 1, 1), np_gl2, color='rosybrown', s=1, label="GL2")
                    plt.axvline(zero_ind, color='green', linestyle=':')
                    # plt.axvline(len(gra)*0.98, color='darkorchid', linestyle=':')
                    plt.legend()
                    plt.show()

                    plt.figure(3)
                    plt.plot(range(1, len(gra) + 1, 1), np_gra_gl2, color='violet', label="Gra+SNIP")
                    # plt.scatter(range(1, len(gra) + 1, 1), np_gra, color='cornflowerblue', s=1, label="Gra")
                    plt.scatter(range(1, len(gra) + 1, 1), np_gl2, color='rosybrown', s=1, label="SNIP")
                    plt.axvline(zero_ind, color='green', linestyle=':')
                    # plt.axvline(len(gra)*0.98, color='darkorchid', linestyle=':')
                    plt.legend()
                    plt.show()
            # ===============================

            x = p
            if prune_mode == 1:  # snip
                x = s
            if prune_mode == 2:  # grasp
                x = p
            if prune_mode == 3:  # grasp+gradl2
                x = p + l
            if prune_mode == 4:  # gl2_diff
                x = p - l
            if prune_mode == 5:  # gl2_diff
                # x = -torch.abs(p - l)
                x = -torch.abs(d)
            if prune_mode == 6:  # ratio_layer
                x = -torch.abs(l - p)
            if prune_mode == 7:  # abs_gra
                x = -torch.abs(p)

            # ????????????
            grads[old_modules[idx]] = x
            if prune_mode == 6:
                grads_prune[old_modules[idx]] = p + l
            layer_cnt += 1

    # ----------------------------------
    debug = False
    if debug:
        # grasp+gl2????????????????????????
        gra_gl2_sort, _gra_gl2_ind = torch.sort(torch.cat([torch.flatten(x) for x in grads_pl.values()]))
        gra = torch.cat([torch.flatten(x) for x in grads_p.values()])
        gl2 = torch.cat([torch.flatten(x) for x in grads_d.values()])
        gra_sort, _ = torch.sort(gra)
        gl2_sort, _ = torch.sort(gl2)
        np_gra_sort = gra_sort.cpu().detach().numpy()
        np_gl2_sort = gl2_sort.cpu().detach().numpy()

        all_num = len(gra_gl2_sort)
        remain_index = _gra_gl2_ind[:int(all_num * 0.02)]
        gra_re_sort, _gra_ind = torch.sort(gra[remain_index])
        np_gra_re_sort = gra_re_sort.cpu().detach().numpy()
        gl2_re_sort, _gl2_ind = torch.sort(gl2[remain_index])
        np_gl2_re_sort = gl2_re_sort.cpu().detach().numpy()

        gra_02_value = np_gra_sort[int(all_num * 0.02)]
        gra_05_value = np_gra_sort[int(all_num * 0.05)]
        gra_10_value = np_gra_sort[int(all_num * 0.10)]
        gl2_02_value = np_gl2_sort[int(all_num * 0.02)]
        gl2_05_value = np_gl2_sort[int(all_num * 0.05)]
        gl2_10_value = np_gl2_sort[int(all_num * 0.10)]

        gra_02_ind = np.argwhere(np_gra_re_sort <= gra_02_value)[-1]
        gra_05_ind = np.argwhere(np_gra_re_sort <= gra_05_value)[-1]
        gra_10_ind = np.argwhere(np_gra_re_sort <= gra_10_value)[-1]
        gl2_02_ind = np.argwhere(np_gl2_re_sort <= gl2_02_value)[-1]
        gl2_05_ind = np.argwhere(np_gl2_re_sort <= gl2_05_value)[-1]
        gl2_10_ind = np.argwhere(np_gl2_re_sort <= gl2_10_value)[-1]

        print('2%:')
        print('gra => ', gra_02_ind / (len(np_gra_re_sort)))
        print('gl2 => ', gl2_02_ind / (len(np_gl2_re_sort)))
        print('5%:')
        print('gra => ', gra_05_ind / (len(np_gra_re_sort)))
        print('gl2 => ', gl2_05_ind / (len(np_gl2_re_sort)))
        print('10%:')
        print('gra => ', gra_10_ind / (len(np_gra_re_sort)))
        print('gl2 => ', gl2_10_ind / (len(np_gl2_re_sort)))
        print(len(np_gra_re_sort), len(np_gl2_re_sort))

    # === ?????????????????????masks ===
    keep_masks = dict()
    keep_masks_98 = dict()

    if prune_mode == 6:
        # ??????????????????
        ratio_layer = []
        prune_masks = dict()
        all_scores = torch.cat([torch.flatten(x) for x in grads_prune.values()])
        norm_factor = torch.abs(torch.sum(all_scores)) + eps
        print("** norm factor:", norm_factor)
        all_scores.div_(norm_factor)
        num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
        threshold, _index = torch.topk(all_scores, num_params_to_rm, sorted=True)
        acceptable_score = threshold[-1]
        print('** accept: ', acceptable_score)
        for m, g in grads_prune.items():
            prune_masks[m] = ((g / norm_factor) <= acceptable_score).float()
        print('Remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in prune_masks.values()])))
        for m, g in prune_masks.items():
            remain_num = torch.sum(torch.flatten(g == 1))
            delete_num = torch.sum(torch.flatten(g == 0))
            all_num = remain_num + delete_num
            ratio = float(remain_num) / float(all_num)
            # print(all_num, remain_num, ratio)
            ratio_layer.append(ratio)

        # ???????????????
        _cnt = 0
        for m, g in grads.items():
            _scores = torch.cat([torch.flatten(g)])
            _norm_factor = torch.abs(torch.sum(_scores)) + eps
            _scores.div_(_norm_factor)
            _num_to_rm = int(len(_scores) * (1 - ratio_layer[_cnt]))
            _thr, _ = torch.topk(_scores, _num_to_rm, sorted=True)
            _acce_score = _thr[-1]
            # print('** ({}) accept: '.format(_cnt), _acce_score)
            keep_masks[m] = ((g / _norm_factor) <= _acce_score).float()
            _cnt += 1
    else:
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

        # ?????????????????????
        # num_params_to_rm__ = int(len(all_scores) * (1 - 0.02))
        # threshold__, _index__ = torch.topk(all_scores, num_params_to_rm__, sorted=True)
        # # import pdb; pdb.set_trace()
        # acceptable_score__ = threshold__[-1]
        # print('** accept: ', acceptable_score__)

        # ??????????????????
        all_scores__ = torch.cat([torch.flatten(x) for x in grads_p.values()])
        norm_factor__ = torch.abs(torch.sum(all_scores__)) + eps
        print("** norm factor:", norm_factor__)
        all_scores__.div_(norm_factor__)
        num_params_to_rm__ = int(len(all_scores__) * (1 - keep_ratio))
        threshold__, _index = torch.topk(all_scores__, num_params_to_rm__, sorted=True)
        # import pdb; pdb.set_trace()
        acceptable_score__ = threshold__[-1]
        print('** accept: ', acceptable_score__)

        for m, g in grads.items():
            if prune_link or prune_mode == 0:
                keep_masks[m] = torch.ones_like(g).float()
            else:
                keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()
                # keep_masks_98[m] = ((g / norm_factor) <= acceptable_score__).float()  # ???????????????
        # ????????????
        for m, g in grads_p.items():
            keep_masks_98[m] = ((g / norm_factor__) <= acceptable_score__).float()
        print('keep_masks_98 remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks_98.values()])))

        print('Remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    grad_key = [x for x in grads.keys()]

    def _get_connected_scores(info='', mode=0):
        # ??????????????????
        _connected_scores = 0
        _last_filter = None
        for m, g in keep_masks.items():
            if isinstance(m, nn.Conv2d) and 'padding' in str(m):  # ?????????resnet???shortcut??????
                # [n, c, k, k]
                _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
                _channel = np.sum(_2d, axis=0)  # ??????

                if _last_filter is not None:  # ??????????????????
                    for i in range(_channel.shape[0]):  # ?????????????????????
                        if _last_filter[i] == 0 and _channel[i] != 0:
                            _connected_scores += 1
                        if mode == 1:
                            if _last_filter[i] != 0 and _channel[i] == 0:
                                _connected_scores += 1

                _last_filter = np.sum(_2d, axis=1)

        print(f'{info}-{mode}->_connected_scores: {_connected_scores}')
        return _connected_scores

    _get_connected_scores(f"{'-' * 20}\nBefore", 1)

    # --- ????????????????????????????????? ---
    _add_mask_num = 0
    _pre_mask_num = 0
    _now_mask_num = 0
    if add_link:
        # for debug
        _add_grasp_value = []

        # _get_connected_scores(f"{'-' * 20}\nBefore", 1)
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
            if isinstance(_key, nn.Conv2d) and 'padding' in str(_key):  # ?????????resnet???shortcut??????
                # print(_key)
                if _layer >= 2:
                    # [n, c, k, k]
                    _2d = np.sum(np.abs(keep_masks[grad_key[_layer]].cpu().detach().numpy()), axis=(2, 3))
                    _channel = np.sum(_2d, axis=0)  # ??????
                    _filter = np.sum(_2d, axis=1)

                    # ??????????????????????????????
                    for i in range(_rep_channel.shape[0]):  # ????????????
                        if _pre_filter[i] != 0 and _rep_channel[i] == 0:
                            temp = grads[grad_key[_rep_layer]][:, i]  # ?????????key??????bug?????????
                            temp = torch.mean(temp, dim=(1, 2))
                            top_k = 1
                            _scores, _index = torch.topk(temp, top_k, largest=False)
                            keep_masks[grad_key[_rep_layer]][_index[:top_k], i] = 1
                            # for debug
                            _add_grasp_value.append(grads[grad_key[_rep_layer]][_index[:top_k], i, 0, 0] / norm_factor)

                    for j in range(_rep_filter.shape[0]):  # ???????????????
                        if _channel[j] != 0 and _rep_filter[j] == 0:
                            temp = grads[grad_key[_rep_layer]][j, :]  # ?????????key??????bug?????????
                            temp = torch.mean(temp, dim=(1, 2))
                            top_k = 1
                            _scores, _index = torch.topk(temp, top_k, largest=False)
                            keep_masks[grad_key[_rep_layer]][j, _index[:top_k]] = 1
                            # for debug
                            _add_grasp_value.append(grads[grad_key[_rep_layer]][j, _index[:top_k], 0, 0] / norm_factor)

                    # ????????????
                    _pre_layer = _rep_layer
                    _rep_layer = _layer
                    _pre_2d = np.sum(np.abs(keep_masks[grad_key[_pre_layer]].cpu().detach().numpy()), axis=(2, 3))
                    _rep_2d = np.sum(np.abs(keep_masks[grad_key[_rep_layer]].cpu().detach().numpy()), axis=(2, 3))
                    _pre_channel = np.sum(_pre_2d, axis=0)
                    _pre_filter = np.sum(_pre_2d, axis=1)
                    _rep_channel = np.sum(_rep_2d, axis=0)
                    _rep_filter = np.sum(_rep_2d, axis=1)

            # ?????????????????????
            if isinstance(_key, nn.Linear):
                # [c, n]
                _linear = np.sum(np.abs(keep_masks[grad_key[_layer]].cpu().detach().numpy()), axis=0)

                # ??????????????????????????????
                for i in range(_rep_channel.shape[0]):  # ????????????
                    if _pre_filter[i] != 0 and _rep_channel[i] == 0:
                        temp = grads[grad_key[_rep_layer]][:, i]  # ?????????key??????bug?????????
                        temp = torch.mean(temp, dim=(1, 2))
                        top_k = 1
                        _scores, _index = torch.topk(temp, top_k, largest=False)
                        keep_masks[grad_key[_rep_layer]][_index[:top_k], i] = 1
                        # for debug
                        _add_grasp_value.append(grads[grad_key[_rep_layer]][_index[:top_k], i, 0, 0] / norm_factor)

                for j in range(_rep_filter.shape[0]):  # ???????????????
                    if _linear[j] != 0 and _rep_filter[j] == 0:
                        temp = grads[grad_key[_rep_layer]][j, :]  # ?????????key??????bug?????????
                        temp = torch.mean(temp, dim=(1, 2))
                        top_k = 1
                        _scores, _index = torch.topk(temp, top_k, largest=False)
                        keep_masks[grad_key[_rep_layer]][j, _index[:top_k]] = 1
                        # for debug
                        _add_grasp_value.append(grads[grad_key[_rep_layer]][j, _index[:top_k], 0, 0] / norm_factor)

        _now_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()]))
        print(f'_now_mask_num: {_now_mask_num}')
        _add_mask_num = _now_mask_num - _pre_mask_num
        print(f'_add_mask_num: {_add_mask_num}')
        _get_connected_scores(f"Add", 1)

        # # for debug
        # if len(_add_grasp_value) > 0:
        #     _mean_ratio = 0
        #     _maxr = 0
        #     _minr = 1
        #     all_scores, _ = torch.sort(all_scores)
        #     # norm_factor
        #     for _value in _add_grasp_value:
        #         _value = torch.mean(_value)
        #         _index = int(torch.nonzero(all_scores <= float(_value))[-1])
        #         _ratio = _index / len(all_scores)
        #         _mean_ratio += _ratio
        #         if _ratio > _maxr:
        #             _maxr = _ratio
        #         if _ratio < _minr:
        #             _minr = _ratio
        #         # print(_ratio*100)
        #     print(f"{'-' * 20}\nmean:")
        #     print(_mean_ratio / len(_add_grasp_value))
        #     print(f'_maxr: {_maxr}')
        #     print(f'_minr: {_minr}')

    # --- ??????????????? ---
    debug = False
    if debug:
        _analyse_layer = 11
        _analyse_weight = keep_masks[grad_key[_analyse_layer]]
        k1 = _analyse_weight.shape[2]
        k2 = _analyse_weight.shape[3]
        _analyse_weight = torch.sum(_analyse_weight, dim=(2, 3), keepdim=True)
        _analyse_weight = _analyse_weight.repeat(1, 1, k1, k2)
        # ??????????????????
        # _analyse_weight = torch.div(_analyse_weight, k1 * k2)
        _conv_zero_num = torch.sum(torch.cat([torch.flatten(_analyse_weight == 0)]))
        _conv_fail_num = torch.sum(torch.cat([torch.flatten(_analyse_weight <= 1)]))
        _conv_all_num = _analyse_weight.shape[0] * _analyse_weight.shape[1]
        print(f'_conv_zero_num: {_conv_zero_num / 9}')
        print(f'_conv_fail_num: {_conv_fail_num / 9}')
        print(f'_conv_all_num: {_conv_all_num}')

    # ????????????????????????
    if prune_link:
        delete_link = True
        _add_mask_num = int(len(all_scores) * (1 - keep_ratio))
    if enlarge:
        _add_mask_num = _now_mask_num - _pre_mask_num / 2
        print("enlarge:", _add_mask_num)
    if delete_link and _add_mask_num > 0:
        # ????????????????????????????????????
        _conv_link_score = {}
        _conv_link_num = {}
        # ????????????????????????
        _conv_link_score_prop = {}
        _conv_link_num_prop = {}
        _pre_layer = 0
        _pre_2d = np.sum(np.abs(grads[grad_key[_pre_layer]].cpu().detach().numpy()), axis=(2, 3))
        _pre_channel = np.sum(_pre_2d, axis=0)
        _pre_filter = np.sum(_pre_2d, axis=1)
        for _layer, _key in enumerate(grad_key):
            if isinstance(_key, nn.Conv2d) and 'padding' in str(_key):  # ?????????resnet???shortcut??????
                # print(_key)
                if _layer >= 1:
                    # [n, c, k, k]
                    _filter_score = torch.mean(grads[grad_key[_layer]] * keep_masks[grad_key[_layer]], dim=(0, 2, 3))
                    _channel_score = torch.mean(grads[grad_key[_pre_layer]] * keep_masks[grad_key[_pre_layer]], dim=(1, 2, 3))
                    _filter_num =torch.sum(keep_masks[grad_key[_layer]], dim=(0, 2, 3))
                    _channel_num = torch.sum(keep_masks[grad_key[_pre_layer]], dim=(1, 2, 3))
                    _conv_link_score[_key] = _filter_score + _channel_score
                    _conv_link_num[_key] = _filter_num + _channel_num
                    _conv_link_score_prop[_key] = torch.log(_filter_score / _channel_score)
                    _conv_link_num_prop[_key] = torch.log(_filter_num / _channel_num)
                    # print(torch.mean(_conv_link_score[_key]))
                    _pre_layer = _layer
                else:
                    _conv_link_score[_key] = None
                    _conv_link_num[_key] = None
                    _conv_link_score_prop[_key] = None
                    _conv_link_num_prop[_key] = None
            # ?????????????????????
            elif isinstance(_key, nn.Linear):
                # [c, n]
                _filter_score = torch.mean(grads[grad_key[_layer]] * keep_masks[grad_key[_layer]], dim=0)
                _channel_score = torch.mean(grads[grad_key[_pre_layer]] * keep_masks[grad_key[_pre_layer]], dim=(1, 2, 3))
                _filter_num = torch.sum(keep_masks[grad_key[_layer]], dim=0)
                _channel_num = torch.sum(keep_masks[grad_key[_pre_layer]], dim=(1, 2, 3))
                _conv_link_score[_key] = _filter_score + _channel_score
                _conv_link_num[_key] = _filter_num + _channel_num
                _conv_link_score_prop[_key] = torch.log(_filter_score / _channel_score)
                _conv_link_num_prop[_key] = torch.log(_filter_num / _channel_num)
            else:
                _conv_link_score[_key] = None
                _conv_link_num[_key] = None
                _conv_link_score_prop[_key] = None
                _conv_link_num_prop[_key] = None

        # ???????????????
        _link_all_scores = None
        _link_all_num = None
        _link_all_scores_prop = None
        _link_all_num_prop = None
        for p in _conv_link_score.values():
            if p is not None:
                if _link_all_scores is not None:
                    _link_all_scores = torch.cat([_link_all_scores, torch.flatten(p)])
                else:
                    _link_all_scores = torch.flatten(p)
        for p in _conv_link_num.values():
            if p is not None:
                if _link_all_num is not None:
                    _link_all_num = torch.cat([_link_all_num, torch.flatten(p)])
                else:
                    _link_all_num = torch.flatten(p)
        for p in _conv_link_score_prop.values():
            if p is not None:
                if _link_all_scores_prop is not None:
                    _link_all_scores_prop = torch.cat([_link_all_scores_prop, torch.flatten(p)])
                else:
                    _link_all_scores_prop = torch.flatten(p)
        for p in _conv_link_num_prop.values():
            if p is not None:
                if _link_all_num_prop is not None:
                    _link_all_num_prop = torch.cat([_link_all_num_prop, torch.flatten(p)])
                else:
                    _link_all_num_prop = torch.flatten(p)
        _link_all_scores, _link_all_scores_index = torch.sort(_link_all_scores, descending=True)
        _link_all_num = _link_all_num[_link_all_scores_index]
        _link_all_scores_prop = _link_all_scores_prop[_link_all_scores_index]
        _link_all_num_prop = _link_all_num_prop[_link_all_scores_index]

        debug = False
        if debug:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????????????????
            plt.rcParams['axes.unicode_minus'] = False  # ????????????????????????
            fig, ax1 = plt.subplots()
            np_link_all_scores = _link_all_scores.cpu().detach().numpy()
            np_link_all_num = _link_all_num.cpu().detach().numpy()
            # np_link_all_num = np_link_all_num * np_link_all_scores.min() / np_link_all_num.max()
            zero_ind = np.argwhere(np_link_all_scores < 0)[0]
            ax1.plot(range(1, len(np_link_all_scores) + 1, 1), np_link_all_scores, color='violet', label="scores")
            ax1.axvline(zero_ind, color='green', linestyle=':')
            ax1.set_ylabel(u"scores")
            # ??????????????????x????????????????????????
            ax2 = ax1.twinx()
            ax2.scatter(range(1, len(np_link_all_scores) + 1, 1), np_link_all_num, color='rosybrown', s=1, label="quantity")
            ax2.set_ylabel(u"quantity")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            plt.legend(h1 + h2, l1 + l2, loc='center left')
            plt.show()

            # ?????????????????????????????????????????????
            np_link_all_scores_prop = _link_all_scores_prop.cpu().detach().numpy()
            np_link_all_num_prop = _link_all_num_prop.cpu().detach().numpy()
            fig, ax1 = plt.subplots()
            ax1.scatter(range(1, len(np_link_all_scores) + 1, 1), np_link_all_scores_prop, color='darkkhaki', s=1, label="scores_prop")
            ax1.set_ylabel(u"scores_prop")
            ax2 = ax1.twinx()
            ax2.scatter(range(1, len(np_link_all_scores) + 1, 1), np_link_all_num_prop, color='salmon', s=1, label="quantity_prop")
            ax2.set_ylabel(u"quantity_prop")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            plt.legend(h1 + h2, l1 + l2, loc='best')
            plt.show()

        # ????????????????????????
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
            if isinstance(_key, nn.Conv2d) and 'padding' in str(_key):  # ?????????resnet???shortcut??????
                if _layer >= 1:
                    for _ch, _link_s in enumerate(_conv_link_score[_key]):
                        if _link_s > _threshold:
                            keep_masks[grad_key[_layer]][:, _ch] = 0
                            keep_masks[grad_key[_pre_layer]][_ch, :] = 0
                    _pre_layer = _layer
            # ?????????????????????
            elif isinstance(_key, nn.Linear):
                for _ch, _link_s in enumerate(_conv_link_score[_key]):
                    if _link_s > _threshold:
                        keep_masks[grad_key[_layer]][:, _ch] = 0
                        keep_masks[grad_key[_pre_layer]][_ch, :] = 0

        _now_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()]))
        print(f'_now_mask_num: {_now_mask_num}')
        _get_connected_scores(f"Del", 1)

    # ???????????????????????????
    if delete_conv:
        _last_filter = None
        for m, g in keep_masks.items():
            if isinstance(m, nn.Conv2d) and 'padding' in str(m):  # ?????????resnet???shortcut??????
                # [n, c, k, k]
                _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
                _channel = np.sum(_2d, axis=0)  # ??????

                if _last_filter is not None:  # ??????????????????
                    for i in range(_channel.shape[0]):  # ?????????????????????
                        if _last_filter[i] == 0 and _channel[i] != 0:
                            keep_masks[m][:, i] = 0  # ??????????????????

                _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))  # ????????????????????????
                _last_filter = np.sum(_2d, axis=1)

        _now_mask_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()]))
        print(f'_now_mask_num: {_now_mask_num}')
        _get_connected_scores(f"Del", 1)

    # Gra???GL2????????????
    # _difference = []
    # for _layer, _key in enumerate(grad_key):
    #     if isinstance(_key, nn.Conv2d) or isinstance(_key, nn.Linear):
    #         _gra = (grads_x[grad_key[_layer]] * keep_masks[grad_key[_layer]])
    #         _gl2 = (grads_q[grad_key[_layer]] * keep_masks[grad_key[_layer]])
    #         _difference.append(float(torch.sum(torch.abs(_gra-_gl2)).cpu().detach()))
    # print('-'*20)
    # print(_difference)

    # ?????????????????????????????????????????????????????????????????????
    # _variance_x = []
    # _variance_s = []
    # _layermean_x = []
    # _layermean_s = []
    # for _layer, _key in enumerate(grad_key):
    #     if _layer > 1 and _layer < 14:
    #         if isinstance(_key, nn.Conv2d) or isinstance(_key, nn.Linear):
    #             _var_x = torch.var(grads_x[grad_key[_layer]])
    #             _var_s = torch.var(grads_s[grad_key[_layer]])
    #             _mean_x = torch.mean(grads_x[grad_key[_layer]])
    #             _mean_s = torch.mean(grads_s[grad_key[_layer]])
    #             _variance_x.append(float(_var_x.cpu().detach()))
    #             _variance_s.append(float(_var_s.cpu().detach()))
    #             _layermean_x.append(float(_mean_x.cpu().detach()))
    #             _layermean_s.append(float(_mean_s.cpu().detach()))
    #
    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????????????????
    # plt.rcParams['axes.unicode_minus'] = False  # ????????????????????????
    # fig, ax1 = plt.subplots()
    # ax1.plot(range(1, len(_variance_x) + 1, 1), _variance_x, color='lightsalmon', label="var_gra")
    # ax1.plot(range(1, len(_variance_s) + 1, 1), _variance_s, color='greenyellow', label="var_snip")
    # ax1.set_ylabel(u"variance")
    # # ??????????????????x????????????????????????
    # ax2 = ax1.twinx()
    # ax2.plot(range(1, len(_layermean_x) + 1, 1), _layermean_x, linestyle='--', color='deepskyblue', label="mean_gra")
    # ax2.plot(range(1, len(_layermean_s) + 1, 1), _layermean_s, linestyle='--', color='deeppink', label="mean_snip")
    # ax2.set_ylabel(u"mean")
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # plt.legend(h1 + h2, l1 + l2, loc='best')
    # plt.show()

    # # ???????????????????????????
    # _score_x = []
    # _score_s = []
    # for _layer, _key in enumerate(grad_key):
    #     if _layer > 5 and _layer < 16:
    #     # if _layer > 2:
    #         if isinstance(_key, nn.Conv2d) or isinstance(_key, nn.Linear):
    #             _score_x.append(torch.flatten(grads_x[grad_key[_layer]]).cpu().detach().numpy())
    #             _score_s.append(torch.flatten(grads_s[grad_key[_layer]]).cpu().detach().numpy())
    #
    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????????????????
    # plt.rcParams['axes.unicode_minus'] = False  # ????????????????????????
    # # ????????????
    # fig, ax1 = plt.subplots()
    # ax1.set_ylabel(u"grasp")
    # # ??????????????????x????????????????????????
    # ax2 = ax1.twinx()
    # ax2.set_ylabel(u"snip")
    # for i in range(len(_score_x)):
    #     if i == 0:
    #         ax1.scatter([i+5.75] * len(_score_x[i]), _score_x[i], color='deeppink', s=1, label="grasp")
    #         ax2.scatter([i+6.25] * len(_score_s[i]), _score_s[i], color='deepskyblue', s=1, label="snip")
    #     else:
    #         ax1.scatter([i+5.75]*len(_score_x[i]), _score_x[i], color='deeppink', s=1)
    #         ax2.scatter([i+6.25]*len(_score_s[i]), _score_s[i], color='deepskyblue', s=1)
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # plt.legend(h1 + h2, l1 + l2, loc='best')
    # plt.show()
    # # ????????????
    # # fig, ax1 = plt.subplots()
    # # for i in range(len(_score_x)):
    # #     if i == 0:
    # #         ax1.scatter([i + 6.25] * len(_score_s[i]), _score_s[i], color='lightsalmon', s=1, label="snip")
    # #     else:
    # #         ax1.scatter([i + 6.25] * len(_score_s[i]), _score_s[i], color='lightsalmon', s=1)
    # # plt.legend()
    # # plt.show()

    # # ??????????????????????????????????????????
    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????????????????
    # plt.rcParams['axes.unicode_minus'] = False  # ????????????????????????
    # plt.figure(1)
    #
    # for _layer, _key in enumerate(grad_key):
    #     if 16 > _layer > -1:
    #         _label1 = 'gss'
    #         _label2 = 'grasp'
    #         # ??????
    #         soc1, ind = torch.sort(torch.cat([torch.flatten(grads[grad_key[_layer]])]), descending=True)
    #         soc2, ind = torch.sort(torch.cat([torch.flatten(grads_p[grad_key[_layer]])]), descending=True)
    #         # soc2 = torch.cat([torch.flatten(grads_s[grad_key[_layer]])])
    #         np_s1 = soc1.cpu().detach().numpy()
    #         np_s2 = soc2.cpu().detach().numpy()
    #         # np_s2 = soc2[ind].cpu().detach().numpy()[int(thre_ind):]
    #         # np_s2 = soc2[ind].cpu().detach().numpy()
    #         thre_ind = np.argwhere(np_s1 > float(acceptable_score*norm_factor))[-1]
    #         thre_ind__ = np.argwhere(np_s2 > float(acceptable_score__*norm_factor__))[-1]
    #         # zero_ind = np.argwhere(np_gra > 0)[-1]
    #         # plt.subplot(4, 4, _layer + 1)
    #         # plt.plot(range(1, len(np_gra) + 1, 1), np_gra, color='violet', label="grasp")
    #         # plt.scatter(range(len(np_gra) - len(np_snip) + 1, len(np_gra) + 1, 1), np_snip, color='cornflowerblue', s=1, label="snip")
    #         # plt.axvline(zero_ind, color='darkorchid', linestyle=':')
    #         # plt.axvline(thre_ind, color='green', linestyle=':')
    #         # plt.legend()
    #         ax1 = plt.subplot(4, 4, _layer + 1)
    #         ax1.plot(range(1, len(np_s1) + 1, 1), np_s1, color='violet', label=_label1)
    #         ax1.axvline(thre_ind, color='violet', linestyle=':')
    #         ax1.set_ylabel(_label1)
    #         # ??????????????????x????????????????????????
    #         ax2 = ax1.twinx()
    #         ax2.plot(range(1, len(np_s2) + 1, 1), np_s2, color='cornflowerblue', label=_label2)
    #         ax2.axvline(thre_ind__, color='cornflowerblue', linestyle=':')
    #         # ax2.scatter(range(len(np_gra)-len(np_snip)+1, len(np_gra)+1, 1), np_snip, color='cornflowerblue', s=1, label="snip")
    #         # ax2.scatter(range(1, len(np_snip)+1, 1), np_snip, color='cornflowerblue', s=1, label="snip")
    #         ax2.set_ylabel(_label2)
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # plt.legend(h1 + h2, l1 + l2, loc='best')
    # plt.show()

    # # ????????????
    # _cut_core_num = []
    # _all_core_num = []
    # _last_filter = None
    # for m, g in keep_masks.items():
    #     _2d = 0
    #     if isinstance(m, nn.Conv2d) and 'padding' in str(m):  # ?????????resnet???shortcut??????
    #         # [n, c, k, k]
    #         _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
    #     elif isinstance(m, nn.Linear):
    #         # [c, n]
    #         _2d = np.abs(keep_masks[m].cpu().detach().numpy())
    #     _channel = np.sum(_2d, axis=0)  # ??????
    #     _cut_cnt = 0
    #     _all_cnt = 0
    #     if _last_filter is not None:  # ??????????????????
    #         for i in range(_channel.shape[0]):  # ?????????????????????
    #             if _last_filter[i] == 0 and _channel[i] != 0:
    #                 _cut_cnt += 1
    #             if _last_filter[i] != 0 and _channel[i] == 0:
    #                 _cut_cnt += 1
    #             if _last_filter[i] == 0 and _channel[i] == 0:
    #                 _all_cnt += 1
    #         _cut_core_num.append(_cut_cnt)
    #         _remain_cnt = _channel.shape[0] - _all_cnt
    #         _all_core_num.append(_remain_cnt)
    #
    #     _last_filter = np.sum(_2d, axis=1)
    #
    # # print(_cut_core_num)
    # # print(_all_core_num)
    # # ??????
    # print([round(_cut_core_num[i]/_all_core_num[i] if _all_core_num[i] != 0 else 1, 3) for i in range(len(_cut_core_num))])

    # # ???????????????
    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????????????????
    # plt.rcParams['axes.unicode_minus'] = False  # ????????????????????????
    # plt.figure(1)
    #
    # _layer_len = 0
    # for _layer, _key in enumerate(grad_key):
    #     if isinstance(_key, nn.Conv2d):
    #         _layer_len += 1
    # print(f'_layer_len: {_layer_len}')
    #
    # for _layer, _key in enumerate(grad_key):
    #     if isinstance(_key, nn.Conv2d):
    #         np_gra = (-1)*np.sum(grads_x[grad_key[_layer]].cpu().detach().numpy(), axis=(2, 3))
    #
    #         plt.subplot(math.ceil((math.sqrt(_layer_len))), math.ceil((math.sqrt(_layer_len))), _layer + 1)
    #         plt.imshow(np_gra, cmap=plt.cm.hot)
    #         plt.colorbar()
    # plt.show()
    #
    # plt.figure(2)
    # for _layer, _key in enumerate(grad_key):
    #     if isinstance(_key, nn.Conv2d):
    #         np_snip = np.sum(grads_s[grad_key[_layer]].cpu().detach().numpy(), axis=(2, 3))
    #
    #         plt.subplot(math.ceil((math.sqrt(_layer_len))), math.ceil((math.sqrt(_layer_len))), _layer + 1)
    #         plt.imshow(np_snip, cmap=plt.cm.hot)
    #         plt.colorbar()
    # plt.show()

    # # ?????????1.5???????????????grasp???snip
    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????????????????
    # plt.rcParams['axes.unicode_minus'] = False  # ????????????????????????
    # plt.figure(1)
    #
    # # ??????????????????
    # _grasp_scores = torch.cat([torch.flatten(x) for x in grads_x.values()])
    # _grasp_norm_factor = torch.abs(torch.sum(_grasp_scores)) + eps
    # print("** _grasp_norm_factor:", _grasp_norm_factor)
    # _grasp_scores.div_(_grasp_norm_factor)
    #
    # _snip_scores = torch.cat([torch.flatten(x) for x in grads_s.values()])
    # _snip_norm_factor = torch.abs(torch.sum(_snip_scores)) + eps
    # print("** _snip_norm_factor:", _snip_norm_factor)
    # _snip_scores.div_(_snip_norm_factor)
    #
    # num_params_to_rm = int(len(_grasp_scores) * (1 - keep_ratio))
    # _grasp_thr, _index = torch.topk(_grasp_scores, num_params_to_rm, sorted=True)
    # _snip_thr, _index = torch.topk(_snip_scores, num_params_to_rm, sorted=True)
    # _grasp_thr_score = float(_grasp_thr[-1]*_grasp_norm_factor)
    # _snip_thr_score = float(_snip_thr[-1]*_snip_norm_factor)
    # print('** _grasp_thr_score: ', _grasp_thr_score)
    # print('** _snip_thr_score: ', _snip_thr_score)
    #
    # # ?????????
    # _layer = 2
    #
    # np_gra = np.mean(grads_x[grad_key[_layer]].cpu().detach().numpy(), axis=(2, 3))
    # np_snip = np.mean(grads_s[grad_key[_layer]].cpu().detach().numpy(), axis=(2, 3))
    # np_gra = np.where(np_gra > _grasp_thr_score, np_gra, np_gra.max())
    # np_snip = (-1)*np.where(np_snip > _snip_thr_score, np_snip, np_snip.max())
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(np_gra, cmap=plt.cm.hot)
    # plt.colorbar()
    # plt.subplot(2, 1, 2)
    # plt.imshow(np_snip, cmap=plt.cm.hot)
    # plt.colorbar()
    # plt.show()

    # # ?????????2??????gra????????????snip?????????????????????
    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # ??????????????????????????????
    # plt.rcParams['axes.unicode_minus'] = False  # ????????????????????????
    # plt.figure(1)
    #
    # _layer_len = 0
    # for _layer, _key in enumerate(grad_key):
    #     if isinstance(_key, nn.Conv2d):
    #         _layer_len += 1
    # # print(f'_layer_len: {_layer_len}')
    #
    # # ????????????
    # # for _layer, _key in enumerate(grad_key):
    # #     if isinstance(_key, nn.Conv2d):
    # #         gra = torch.cat([torch.flatten(torch.mean(grads_x[grad_key[_layer]], dim=(2, 3)))])
    # #         snip = (-1)*torch.cat([torch.flatten(torch.mean(grads_s[grad_key[_layer]], dim=(2, 3)))])
    # #         np_gra = gra.cpu().detach().numpy()
    # #         np_snip = snip.cpu().detach().numpy()
    # #         plt.subplot(math.ceil((math.sqrt(_layer_len))), math.ceil((math.sqrt(_layer_len))), _layer + 1)
    # #         plt.scatter(np_gra, np_snip, s=1, alpha=0.3, cmap='rainbow')
    #
    # # ????????????
    # # ?????????
    # # gra = None
    # # snip = None
    # # for _layer, _key in enumerate(grad_key):
    # #     if isinstance(_key, nn.Conv2d):
    # #         if gra is None:
    # #             gra = torch.cat([torch.flatten(torch.mean(grads_p[grad_key[_layer]], dim=(2, 3)))])
    # #             snip = torch.cat([(-1)*torch.flatten(torch.mean(grads_s[grad_key[_layer]], dim=(2, 3)))])
    # #         else:
    # #             gra = torch.cat([torch.flatten(torch.mean(grads_p[grad_key[_layer]], dim=(2, 3))), gra])
    # #             snip = torch.cat([(-1)*torch.flatten(torch.mean(grads_s[grad_key[_layer]], dim=(2, 3))), snip])
    # # ??????
    # gra = torch.cat([torch.flatten(x) for x in grads.values()])
    # snip = torch.cat([torch.flatten(x) for x in grads_s.values()])
    #
    # np_gra = -gra.cpu().detach().numpy()
    # np_snip = -snip.cpu().detach().numpy()
    # # gra_index = np.argsort(-np_gra)  # ??????
    # # gra_len = len(np_gra)
    # # gra_02_ind = gra_index[int(gra_len * 0.02)]
    # # gra_05_ind = gra_index[int(gra_len * 0.05)]
    # # gra_10_ind = gra_index[int(gra_len * 0.1)]
    # # plt.axvline(np_gra[gra_02_ind], color='red', linestyle=':')
    # # plt.axvline(np_gra[gra_05_ind], color='green', linestyle=':')
    # # plt.axvline(np_gra[gra_10_ind], color='blue', linestyle=':')
    # plt.scatter(np_gra, np_snip, s=1, alpha=0.7, color='coral')
    # # plt.scatter(np_gra, np_snip, s=1, c=np.abs(np_gra*np_snip), cmap = plt.cm.plasma)
    # _lim_x = np.max(np_gra) / 4
    # __lim_x = np.min(np_gra) / 4
    # _lim_y = np.max(np_snip) / 4
    # __lim_y = np.min(np_snip) / 4
    # plt.xlim([__lim_x, _lim_x])
    # plt.ylim([__lim_y, _lim_y])
    # plt.show()

    # ??????????????????
    if train_one:
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for it in range(len(inputs_one)):
            inputs = inputs_one[it].to(device)
            targets = targets_one[it].to(device)
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

            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (train_loss / (it + 1), 100. * correct / total, correct, total))
            print(desc)

    # if first_data is None:
    #     Panning(net, ratio, train_dataloader, device,
    #             num_classes, samples_per_class, num_iters, T, reinit, single_data,
    #             prune_mode, prune_conv, add_link, delete_link, delete_conv, enlarge,
    #             prune_link, first_masks, first_data=(inputs_one, targets_one), train_one=False)

    return keep_masks, keep_masks_98, (inputs_one, targets_one)
