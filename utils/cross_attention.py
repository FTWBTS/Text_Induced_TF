import matplotlib.pyplot as plt
import seaborn as sns
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer,LlamaForCausalLM
import os
import time
import warnings
import numpy as np

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.linear_enc = nn.Linear(1, embed_dim)  # 将 x_enc 的嵌入维度调整为 embed_dim
        self.linear_text = nn.Linear(8, embed_dim)  # 将 x_text 的嵌入维度调整为 embed_dim
    
    def forward(self, x_enc, x_text):
        # print(x_enc.shape)
        # print(x_text.shape)
        # Ensure the model and input tensors are on the same device
        device = x_enc.device  # 获取 x_enc 的设备
        x_text = x_text.to(device)  # 将 x_text 移动到与 x_enc 相同的设备

        # Print the devices of the tensors for debugging
        
        # Apply linear transformation and transpose
        x_enc = self.linear_enc(x_enc).transpose(0, 1)
        if x_text.dim() == 2:
            x_text = x_text.unsqueeze(-1)  # [B, D] -> [1, B, D]
        x_text = self.linear_text(x_text).transpose(0, 1)
        
        # Now, x_enc and x_text should be [L, B, D] (3D)
        attn_output, attn_weights = self.attn(x_enc, x_text, x_text)
        
        # Return the attention weights (shape: [B, num_heads, N, T])
        return attn_weights.transpose(0, 1)  # [B, num_heads, N, T]



# ======================= Visualization Function =======================
def plot_heatmap(attn_weights, title='Attention Heatmap'):
    """ Plot a heatmap for the attention weights """
    attn_weights = attn_weights.cpu().detach().numpy()
    # print(attn_weights.shape)
    # Take the average across heads (if multiple heads)
    attn_weights = attn_weights.mean(axis=1) 
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap='viridis', annot=False, fmt='.2f', cbar=True)
    plt.title(title)
    plt.xlabel('Text Timesteps')
    plt.ylabel('Encoded Timesteps')
    plt.savefig(f"{title}.png")