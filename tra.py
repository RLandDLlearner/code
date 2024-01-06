# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import math
import json
import collections  
import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm  

from qlib.utils import get_or_create_path  
from qlib.log import get_module_logger
from qlib.model.base import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

# TRA模型主类
# 包括验证集指标记录,保存模型和预测结果等
# 全局超参数也在此处定义
class TRAModel(Model):
    
    def __init__(
        self,
        model_config,
        tra_config,
        model_type="LSTM",  
        lr=1e-3,
        n_epochs=500,
        early_stop=50,
        smooth_steps=5,
        max_steps_per_epoch=None,
        freeze_model=False,
        model_init_state=None,
        lamb=0.0,
        rho=0.99,
        seed=None, 
        logdir=None,
        eval_train=True,
        eval_test=False,
        avg_params=True,
        **kwargs,
    ):
        
        # 随机数种子
        np.random.seed(seed) 
        torch.manual_seed(seed)
        
        // 省略部分内容
                // 省略部分内容
        
        # LSTM或Transformer作为底层模型
        self.model = eval(model_type)(**model_config).to(device)  
        if model_init_state:
            # 加载预训练模型
            self.model.load_state_dict(torch.load(model_init_state, map_location="cpu")["model"])
        if freeze_model:
            # 冻结底层模型参数
            for param in self.model.parameters():  
                param.requires_grad_(False)  
                
        # 构建TRA路由适配模块    
        self.tra = TRA(self.model.output_size, **tra_config).to(device)
        
        # 定义优化器,更新底层模型和TRA的参数
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=lr)
        
        // 记录配置参数
        
        # 初始化训练状态
        self.fitted = False  
        self.global_step = -1

    def train_epoch(self, data_set):
        
        # 设置训练模式
        self.model.train()
        self.tra.train() 

        // 样本读取,模型推理等代码
        
        # 返回平均loss
        total_loss /= total_count

        return total_loss

    def test_epoch(self, data_set, return_pred=False):
        
        // 测试模式下的前向推理和指标计算
            def test_epoch(self, data_set, return_pred=False):

       // 测试模式下的前向推理和指标计算

        if return_pred:
            # 返回预测数据框
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)

        return metrics, preds

    def fit(self, dataset, evals_result=dict()):
       
        // 训练集、验证集和测试集
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])
        
        // 保存最佳验证分数
        best_score = -1
        best_epoch = 0
        
        // early stop和模型平均
        stop_rounds = 0
        best_params = {
            "model": copy.deepcopy(self.model.state_dict()),
            "tra": copy.deepcopy(self.tra.state_dict()),
        }
        
        // 训练主循环
        for epoch in range(self.n_epochs):
            
            // 保存中间模型参数
            params_list["model"].append(copy.deepcopy(self.model.state_dict()))
            params_list["tra"].append(copy.deepcopy(self.tra.state_dict()))
            
            // 计算指标
            metrics, preds = self.test_epoch(test_set, return_pred=True)
            
            // 更新最优模型  
            if valid_metrics["IC"] > best_score:
                best_score = valid_metrics["IC"]  
                stop_rounds = 0
                best_epoch = epoch
                best_params = {
                    "model": copy.deepcopy(self.model.state_dict()),
                    "tra": copy.deepcopy(self.tra.state_dict()),
                }
                
            // early stop判断
            else:
                stop_rounds += 1
                if stop_rounds >= self.early_stop:  
                    break
                    
         // 保存最佳模型和预测       
         self.model.load_state_dict(best_params["model"])
         self.tra.load_state_dict(best_params["tra"])
         
         metrics, preds = self.test_epoch(test_set, return_pred=True)
         self.logger.info("test metrics: %s" % metrics)
            class LSTM(nn.Module):

    """LSTM Model
    
    LSTM作为底层特征提取器
    
    Args:
        input_size: 输入维度        
        hidden_size: 隐状态维度
        num_layers: LSTM层数
        use_attn: 是否使用Attention    
        dropout: Dropout比率 
        input_drop: 输入Dropout比率
        noise_level: 噪声比率
    """

    def __init__(self, input_size, hidden_size, num_layers, use_attn, 
                 dropout, input_drop, noise_level):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attn = use_attn
        self.noise_level = noise_level

        // LSTM层
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout)
            
        // Attention层 
        if self.use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)  
            self.output_size = hidden_size * 2
            
        else:
            self.output_size = hidden_size

    def forward(self, x):
        
        // 添加高斯噪声
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x) 
            x = x + noise * self.noise_level
        
        // LSTM前向计算
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1]
        
        // Attention融合
        if self.use_attn:
            laten = self.W(rnn_out) 
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1).squeeze()
            last_out = torch.cat([last_out, att_out], dim=1)

        return last_out

class Transformer(nn.Module):

    """Transformer Model
    
    Transformer作为底层特征提取器
    
    Args: 
        输入和LSTM模型相同,不再赘述
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_heads,
                dropout, input_drop, noise_level):
       super().__init__()
       
       // 编码器层定义
       self.encoder = nn.TransformerEncoder(
           nn.TransformerEncoderLayer(
               d_model=hidden_size, nhead=num_heads, dropout=dropout
           ), num_layers=num_layers)
       
       self.output_size = hidden_size

    def forward(self, x):

       // 添加噪声 
       
       // Transformer编码
       x = self.encoder(x)

       return x[-1]

class TRA(nn.Module):

    """Temporal Routing Adaptor (TRA)
    
    TRA takes historical prediction errors & latent representation as inputs,
    then routes the input sample to a specific predictor for training & inference.

    Args:
        input_size: RNN/Transformer的隐状态大小
        num_states: 路由状态数,默认为1
        hidden_size: 路由器LSTM的隐状态大小  
        tau: Gumbel Softmax的温度系数
    """
    
    def __init__(self, input_size, num_states=1, hidden_size=8, tau=1.0, src_info="LR_TPE"):
        super().__init__()
        
        self.num_states = num_states   
        self.tau = tau
        self.src_info = src_info    

        if num_states > 1:
        
            self.router = nn.LSTM(  
                input_size=num_states,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True, 
            )
            
            self.fc = nn.Linear(hidden_size + input_size, num_states)

        self.predictors = nn.Linear(input_size, num_states)  
        
    def forward(self, hidden, hist_loss):
      
        preds = self.predictors(hidden)   
        
        if self.num_states == 1:
            return preds.squeeze(-1), preds, None
        
        router_out, _ = self.router(hist_loss)
        
        // 整合路由器状态和底层模型特征
        
        out = self.fc(torch.cat([temporal_pred_error, latent_representation], dim=-1))  
        prob = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=False)
        
        if self.training:
            final_pred = (preds * prob).sum(dim=-1)  
        else:
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]

        return final_pred, preds, prob
