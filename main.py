#good
import sys
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass
from einops import rearrange, repeat, einsum # ??

from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls # Get Data loaders：原始作者定義的載入資料模組。根據提供的參數，載入並處理對應的資料集，最後輸出 PyTorch 標準格式的 DataLoaders（訓練與測試用）。

import time
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

import random, datetime
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from packaging import version

import torch
from torch import nn # import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

assert torch.__version__ >= '1.8.0', "DDP-based MoE requires Pytorch >= 1.8.0"

if version.parse(torch.__version__) >= version.parse("2.1.0"):
    import torch.utils.cpp_extension # Monkey Patch to fix include_paths(cuda=True) for torch >= 2.1.0
    # 取得原始 include_paths()，避免遞迴呼叫自己
    _original_include_paths = torch.utils.cpp_extension.include_paths # 備份原本的 include_paths() 函式
    def include_paths_patched(*args, **kwargs): # 攔截呼叫，把多餘的 cuda 參數去掉
        if 'cuda' in kwargs:
            del kwargs['cuda']  # 移除不支援的參數
        return _original_include_paths(*args, **kwargs)
    torch.utils.cpp_extension.include_paths = include_paths_patched # 進用新的版本取代原來的函式（Monkey Patch）
    print("✅ Patched torch.utils.cpp_extension.include_paths successfully!")
from model import xlstm
from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig

from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

import argparse

parser = argparse.ArgumentParser() # 解析命令列參數（Command-line arguments）、控制所有訓練與測試參數。

# TODO 【1】確實有用到的重要核心參數（✅代表有用到。）
# training
parser.add_argument('--is_train', type=int, default=0, help='training the model') # 控制是否訓練或測試。( 1: train, 0: test ) ✅
parser.add_argument('--context_points', type=int, default=512, help='sequence length') # 輸入序列長度（如 336） ✅
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon') # 預測序列長度、預測步數（如 96） ✅
parser.add_argument('--batch_size', type=int, default=64, help='batch size') # DataLoader批次大小，在 get_dls() 會用到。 ✅

parser.add_argument('--dset', type=str, default='ettm1', help='dataset name') # 資料集名稱（如 ettm1） ✅
parser.add_argument('--model_name2', type=str, default='xLSTMTime', help='model_name2') # 模型命名，在 args.save_model_name 會用到。 ✅

parser.add_argument('--model_id', type=int, default=1, help='id of the saved model') # 模型版本號（便於存檔），在 args.save_model_name 會用到。
# Optimization args
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs') # 訓練總迭代次數，在 learn.fit_one_cycle 會用到。
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') # 學習率（可被 find_lr() 覆蓋）
parser.add_argument('--n2', type=int, default=256, help='Second Embedded representation') # 要傳入 xLSTMBlockStack 的嵌入維度（可理解為 embedding_dim），用在 model.py。 ✅

parser.add_argument('--use_time_features', type=int, default=1, help='whether to use time features or not') # 是否加入時間欄位特徵，用在 datautils.py。✅
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model') # 特徵類型（M: multivariate 多變量、 S: Single單變量），用在 datautils.py。✅
                                                                                                            # 單變量（S）=> 每筆資料只有一種特徵（只有一個欄位要預測）
                                                                                                            # 多變量（M）=> 每筆資料有多種特徵（同時觀察/預測多個欄位）
                                                                                                            # NOTE MS -> 多變量預測單變量（multi→single）。
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader') # DataLoader 多執行緒設定，用在 datautils.py。✅

# TODO 【2】定義了但目前未被使用的參數（可能是保留、兼容或暫未實作）（❌代表未用到。）
# 模型初始化
parser.add_argument('--n1', type=int, default=128, help='First Embedded representation')  #256 # 原意應為第一層 embedding，未使用。 ❌
# 原本可能是用於 Mamba 模型的設定，但目前 xlstm 未使用
parser.add_argument('--d_state', type=int, default=128, help='d_state parameter of Mamba')  #256 ❌
parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba') # ❌
parser.add_argument('--e_fact', type=int, default=2, help='expand factor parameter of Mamba') # ❌
parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0') # 殘差設定❌
# 和 Transformer 架構相關，原始碼中未被 xlstm 調用。
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers') # ❌
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model') # ❌
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout') # 沒有實際實作❌
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
# parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
# parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
# parser = argparse.ArgumentParser(description='Swin Transformer training and evaluation script', add_help=False)
# parser.add_argument('Swin Transformer training and evaluation script', add_help=False)
# 保留給 config 檔用，但目前未使用。
parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file') # ❌
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+') # ❌
# 可能在多模型版本中有用，但目前 xlstm 還沒切換架構
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model') # 多架構選擇時可用，目前僅支援 xLSTM。 ❌
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data') # 特徵標準化方法，未用到。❌
parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0') # 是否讓每個通道（feature）獨立建模，而不是共享參數或進行聯合建模。 #可能是為 Mamba 模型預留的❌

# TODO 【3】取決於是否啟用某些功能的參數
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization') # 啟用 RevIN（可逆標準化），在 RevInCB 有用到。 # cbs = [RevInCB(dls.vars)] if args.revin else []
                                                                                              # RevIN（Reversible Instance Normalization）可逆標準化技術 => 讓模型在統一的數值世界裡學習，學完再把預測翻譯回原本的語言。
# Patch 時間補丁設定（用在 PatchCB），用在部分 callback 或未啟用。
# patch補丁：把一整段長時間序列，切成一小段一小段的區塊（時間片段）來處理。
parser.add_argument('--patch_len', type=int, default=12, help='patch length') # 每段看多長。 目前未啟用 PatchCB。❌
parser.add_argument('--stride', type=int, default=12, help='stride between patch') # 每次滑動多少秒 目前未啟用 PatchCB。 ❌

args = parser.parse_args()
print('args:', args)

# 設定儲存模型的名稱與路徑
args.save_model_name = str(args.model_name2) + '_cw' + str(args.context_points) + '_tw' + str(
    args.target_points) + '_patch' + str(args.patch_len) + '_stride' + str(args.stride) + '_epochs' + str(
    args.n_epochs) + '_model' + str(args.model_id) # 模型名稱
args.save_path = 'saved_models/' + args.dset  # 儲存模型位置路徑
if not os.path.exists(args.save_path): os.makedirs(args.save_path) # 建立資料夾

configs = args # 全域變數（global variable），只要在 get_model() 裡沒有重新定義名為 configs 的區域變數，Python 就會使用外層的 全域變數 configs。


def get_model(c_in, args):
    """
    產出模型結構
    -- c_in: number of input variables （輸入特徵數，也就是 features 數量。 enc_in = c_in = features）
    -- 若想導入其他模型（如 LSTM、Transformer），可以直接改寫這裡。
    """

    # * patch補丁，但實際上沒被使用！
    # 計算資料在經過 Patch 分段處理後，會被切成多少個時間片段（patches）。
    # 1.) 加入局部時間資訊（例如：用一小段資料判斷未來趨勢）。
    # 2.) 讓模型能觀察多個區間（patches）而非整體序列。
    # 3.) 模型更容易聚焦局部資訊（短期趨勢）。
    # 4.) 降低記憶體需求。
    # 5.) 可以重疊（用 stride 控制），保留更多上下文。
    # -- num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1 # 滑動視窗切patch，總共可以切幾段？
    # max(args.context_points, args.patch_len)：保證序列長度至少不小於 patch 長度（安全設計）。
    # max(...) - patch_len：可滑動的「剩餘距離」。
    # 除以stride：每次滑 stride 那麼遠，能滑幾次？
    # +1：加上第一次切（從 0 開始）
    # -- print('number of patches:', num_patch) # get number of patches  EX. 模型會把一筆長為 336 的序列切成 28 段，每段 12 個時間點。

    # todo get model
    model = xlstm(configs, enc_in=c_in) # 把特徵數交給模型
                                        # xlstm() 是主模型結構，搭配 xLSTMBlockStack。
    return model


def combined_loss(input, target, alpha=0.5):
    """
    A combined loss function that computes a weighted sum of MSELoss and L1Loss.
    `alpha` is the weight for MSELoss and (1-alpha) is the weight for L1Loss.
    """
    mse_loss = torch.nn.MSELoss(reduction='mean')
    l1_loss = torch.nn.L1Loss(reduction='mean')
    return alpha * mse_loss(input, target) + (1 - alpha) * l1_loss(input, target)


def find_lr():
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)

    # get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')
    #loss_func=combined_loss
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    # define learner
    learn = Learner(dls, model, loss_func, cbs=cbs)  #cbs=cbs
    # fit the data to the model
    return learn.lr_finder()


def train_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    #print('in out', dls.vars, dls.c, dls.len)

    # get model
    model = get_model(dls.vars, args)
    #model = get_model(dls.vars, args, model_type)

    # get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')
    #loss_func=combined_loss

    #delta = 0.25
    #loss_func = HuberLoss(delta)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
        #cbs = [
        #PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_loss', fname=args.save_model_name,
                    path=args.save_path)
    ]

    # define learner
    learn = Learner(dls, model, loss_func,
                    lr=lr,
                    cbs=cbs,
                    metrics=[mse, mae]
                    )

    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def test_func():
    weight_path = args.save_path + '/' + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    #model = torch.load(weight_path)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, cbs=cbs)  #cbs=cbs
    out = learn.test(dls.test, weight_path=weight_path, scores=[mse, mae])  # out: a list of [pred, targ, score_values]
    return out


import matplotlib.pyplot as plt


def plot_feature_actual_vs_predicted(actual, predicted, feature_idx):
    """
    Plot the actual vs predicted values for a specific feature for the first sequence.

    Parameters:
    - actual (np.array or torch.Tensor): Array of actual values.
    - predicted (np.array or torch.Tensor): Array of predicted values.
    - feature_idx (int): Index of the feature to plot.
    """

    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()

    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    ## Selecting the feature across all time steps
    #actual_feature = actual[0:, feature_idx]
    #predicted_feature = predicted[0:, feature_idx]

    # Select the first sequence for the given feature index
    actual_feature = actual[0, :, feature_idx]
    predicted_feature = predicted[0, :, feature_idx]
    #actual_feature = np.mean(actual[: , : ,feature_idx ], axis=0 )
    #predicted_feature = np.mean(predicted[: , : ,feature_idx ], axis=0)

    # Plot the first sequence
    plt.figure(figsize=(10, 6))
    plt.plot(actual_feature, label="Actual", color='blue')
    plt.plot(predicted_feature, label="Predicted", color='red', linestyle='--')
    plt.title(f"Actual vs Predicted for Feature {feature_idx}, Sequence 0")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    if args.is_train:

        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)

    else:  # testing mode

        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)

        for feature_idx in range(7):  # Assuming there are 7 features
            plot_feature_actual_vs_predicted(out[1], out[0], feature_idx)

    print('----------- Complete! -----------')
