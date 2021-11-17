# -*- coding: utf-8 -*-

"""
Created on 08/21/2021
gogogo.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""


import os

# os.system("python main_prune_panning_exp.py --config configs/cifar10/vgg19/Panning_98.json --epoch 180 --run cosine_100 --prune_mode 0 --prune_conv 0 --core_link 0 --lr_mode cosine")
# os.system("python main_prune_panning_exp.py --config configs/cifar10/vgg19/Panning_98.json --epoch 180 --run preset_100 --prune_mode 0 --prune_conv 0 --core_link 0 --lr_mode preset")
# os.system("python main_prune_panning_exp.py --config configs/cifar10/vgg19/Panning_98.json --epoch 180 --run grasp_preset_98 --prune_mode 2 --prune_conv 1 --core_link 0 --lr_mode preset")
# os.system("python main_prune_panning_exp.py --config configs/cifar10/vgg19/Panning_98.json --epoch 180 --run grasp_cosine_98 --prune_mode 2 --prune_conv 1 --core_link 0 --lr_mode cosine")

os.system("python main_prune_panning_label.py --config configs/cifar10/vgg19/Panning_95.json --run gss_t1 --grad_mode 1 --prune_mode 4")
os.system("python main_prune_panning_label.py --config configs/cifar10/vgg19/Panning_90.json --run gss_t2 --grad_mode 1 --prune_mode 4")
os.system("python main_prune_panning_label.py --config configs/cifar100/vgg19/Panning_95.json --run gss_t3 --grad_mode 1 --prune_mode 4")
os.system("python main_prune_panning_label.py --config configs/cifar100/vgg19/Panning_90.json --run gss_t4 --grad_mode 1 --prune_mode 4")
os.system("python main_prune_panning_label.py --config configs/cifar10/resnet32/Panning_95.json --run gss_t5 --grad_mode 1 --prune_mode 4")
os.system("python main_prune_panning_label.py --config configs/cifar10/resnet32/Panning_90.json --run gss_t6 --grad_mode 1 --prune_mode 4")

