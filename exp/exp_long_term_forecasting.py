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
from pathlib import Path
from utils.tical_analysis_tools import plot_kappa_heatmap, plot_pi_heatmap, plot_gate_hist,plot_kappa_multiple_lines,plot_Tir_line
from utils.cross_attention import CrossAttention,plot_heatmap

def norm(input_emb):
    input_emb=input_emb- input_emb.mean(1, keepdim=True).detach()
    input_emb=input_emb/torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5)
   
    return input_emb
class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)  
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  
                x = F.relu(x)
                x = self.dropout(x)  
        return x
    
warnings.filterwarnings('ignore')





class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        args.task_name = 'long_term_forecast'
        
        configs=args
        self.text_path=configs.text_path
        self.prompt_weight=configs.prompt_weight
        self.attribute="final_sum"
        self.type_tag=configs.type_tag
        self.text_len=configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len=configs.pred_len
        self.text_embedding_dim = configs.text_emb
        self.pool_type=configs.pool_type
        self.use_fullmodel=configs.use_fullmodel
        self.hug_token=configs.huggingface_token
        # self.prompt_weight = nn.Parameter(torch.tensor(0.5))
        mlp_sizes=[self.d_llm,int(self.d_llm/8),self.text_embedding_dim]
        self.loss_weight = configs.adjust_loss#1e-1
        self.Doc2Vec=False
        self.lmb_cot = configs.TICAL_lmb_cot
        self.lmb_delta = configs.TICAL_lmb_delta
        self.lmb_entropy = configs.TICAL_lmb_entropy
        self.lmb_tv = configs.TICAL_lmb_tv
        super(Exp_Long_Term_Forecast, self).__init__(args)
        if mlp_sizes is not None:
            # self.mlp = MLP(mlp_sizes,dropout_rate=0.3)
            self.mlp = nn.Sequential(
                nn.Linear(mlp_sizes[0], mlp_sizes[1]),
                nn.ReLU(),
                nn.Linear(mlp_sizes[1], mlp_sizes[2]),
                nn.ReLU(),
                nn.Dropout(0.5)
            )

            # print number of parameters of self.model
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f'Number of parameters in TS model: {num_params}')
            # print number of parameters of self.mlp
            num_params_mlp = sum(p.numel() for p in self.mlp.parameters())
            print(f'Number of parameters in MLP: {num_params_mlp}')
            print(f'Total number of parameters: {num_params + num_params_mlp}')
        else:
            self.mlp = None
        mlp_sizes2=[self.text_embedding_dim+self.args.pred_len,self.args.pred_len]
        if mlp_sizes2 is not None:
            self.mlp_proj = MLP(mlp_sizes2,dropout_rate=0.3)

        self.language_to_time_series_projection = nn.Sequential(
            nn.Linear(self.d_llm, 12),
            nn.ReLU()
        ).cuda()

        if configs.llm_model == 'Doc2Vec':
            print('Cannot using Doc2Vec')
            print("Training Doc2Vec model")
            raise Exception('Doc2Vec model is not supported')
        else:
            if configs.llm_model == 'LLAMA2':
                print("use llmam2")
                self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
                self.llama_config.num_hidden_layers = configs.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                # except EnvironmentError:  # downloads model from HF is not already done
                except:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                # except EnvironmentError:  # downloads the tokenizer from HF if not already done
                except:  # downloads model from HF is not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'LLAMA3':
                # Automatically load the configuration, model, and tokenizer for LLaMA-3-8B
                llama3_path = "meta-llama/Meta-Llama-3-8B-Instruct"
                cache_path = "./"

                # Load the configuration with custom adjustments
                self.config =  LlamaConfig.from_pretrained(llama3_path,token=self.hug_token,cache_dir=cache_path)

                self.config.num_hidden_layers = configs.llm_layers
                self.config.output_attentions = True
                self.config.output_hidden_states = True

                self.llm_model  = LlamaModel.from_pretrained(
                    llama3_path,
                    config=self.config,
                    token=self.hug_token,cache_dir=cache_path
                )
                self.tokenizer = AutoTokenizer.from_pretrained(llama3_path,use_auth_token=self.hug_token,cache_dir=cache_path)
            
            # if configs.llm_model == 'LLAMA2':
            #     print("use llama2")
                
            #     # 使用本地路径配置
            #     local_model_path = '/home/ghl/MMTS/TaTS_TICAL_For_Para/LM_Model/llama2'
                
            #     # 加载LlamaConfig
            #     self.llama_config = LlamaConfig.from_pretrained(local_model_path)
            #     self.llama_config.num_hidden_layers = configs.llm_layers
            #     self.llama_config.output_attentions = True
            #     self.llama_config.output_hidden_states = True

            #     try:
            #         # 加载本地LlamaModel
            #         self.llm_model = LlamaModel.from_pretrained(
            #             local_model_path,
            #             config=self.llama_config,
            #             # load_in_4bit=True  # 如果需要4bit加载
            #         )
            #     except:
            #         print("Local model files not found. Please check the model files.")

            #     try:
            #         # 加载本地LlamaTokenizer
            #         self.tokenizer = LlamaTokenizer.from_pretrained(
            #             local_model_path
            #         )
            #     except:
            #         print("Local tokenizer files not found. Please check the tokenizer files.")
                    
            #     if not hasattr(self, 'tokenizer') or not self.tokenizer:
            #         raise ValueError("Tokenizer was not successfully loaded.")

            
            elif configs.llm_model == 'GPT2':
                model_path = "/home/ghl/MMTS/MM-TSFlib-TICAL/GPT-2"
                self.gpt2_config = GPT2Config.from_pretrained(model_path)

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                
                self.llm_model = GPT2Model.from_pretrained(
                    model_path,
                    config=self.gpt2_config,
                    local_files_only=True,
                )

                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                )
            elif configs.llm_model == 'GPT2M':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-medium')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )
                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2L':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-large')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )
                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2XL':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-xl')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'BERT':
                print("use bert text encoder")
                self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                try:
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.bert_config,
                    )
                # except EnvironmentError:  # downloads model from HF is not already done
                except:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                    )

                try:
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                # except EnvironmentError:  # downloads the tokenizer from HF if not already done
                except:  # downloads model from HF is not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            
            else:
                raise Exception('LLM model is not defined')

            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model=self.llm_model.to(self.device)
        if args.init_method == 'uniform':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.uniform_(self.weight1.weight)
            nn.init.uniform_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        elif args.init_method == 'normal':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.normal_(self.weight1.weight)
            nn.init.normal_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        else:
            raise ValueError('Unsupported initialization method')
        
        self.mlp=self.mlp.to(self.device)
        self.mlp_proj=self.mlp_proj.to(self.device)
        self.learning_rate = 1e-2
        self.learning_rate2=1e-2
        self.learning_rate3=1e-3
        self.use_prior_blend = True
        self.cross_attention = CrossAttention(embed_dim=configs.d_model, num_heads=configs.n_heads).to(self.device)
        
    # ======= 放在 __init__ 末尾附近，增加一个便捷函数取 DP 包裹前的模型 =======
    def _core_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(
            self.args
        ).float()
        
    # def _build_model(self):
    #     model = self.model_dict[self.args.model].Model(
    #         self.args,
    #         use_text=self.args.use_text,
    #         txt_dim=self.text_embedding_dim,   # 不要手写常数，直接用 text_embedding_dim
    #         K=self.args.TICAL_k,
    #         shape_types=self.args.TICAL_shape_types,
    #         kernel_emb_dim=self.args.TICAL_kernel_emb_dim,
    #         cot_eps=self.args.TICAL_cot_eps,
    #         cot_iters=self.args.TICAL_cot_iters,
    #         cot_bandwidth=self.args.TICAL_cot_bandwidth,
    #         cot_alpha=self.args.TICAL_cot_alpha,
    #         cot_beta=self.args.TICAL_cot_beta,
    #         lmb_cot=self.args.TICAL_lmb_cot,
    #         lmb_delta=self.args.TICAL_lmb_delta,
    #         lmb_entropy=self.args.TICAL_lmb_entropy,
    #         lmb_tv=self.args.TICAL_lmb_tv,
    #         gating_dim=self.args.TICAL_gate_dim,
    #         gate_weight=self.args.TICAL_gate_weight,
    #     ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.llm_model, self.tokenizer)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    def _select_optimizer_mlp(self):
        model_optim = optim.Adam(self.mlp.parameters(), lr=self.args.learning_rate2)
        return model_optim
    def _select_optimizer_proj(self):
        model_optim = optim.Adam(self.mlp_proj.parameters(), lr=self.args.learning_rate3)
        return model_optim
    def _select_optimizer_weight(self):
        model_optim = optim.Adam([{'params': self.weight1.parameters()},
                              {'params': self.weight2.parameters()}], lr=self.args.learning_rate_weight)
        return model_optim
    def _select_criterion(self):
        loss_type = self.args.loss_function.lower()
        print(loss_type)
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'smoothl1':
            return nn.SmoothL1Loss()
        
        else:
            return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion, all_metric=False):
        total_loss = []
        if all_metric:
            total_mae, total_mse, total_rmse, total_mape, total_mspe = [], [], [], [], []

        self.model.eval()
        self.mlp.eval()
        self.mlp_proj.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
                batch_x       = batch_x.float().to(self.device)            # [B, L, D]
                batch_y  = batch_y.float().to(self.device)            # [B, L+H, D] or [B, H, D] 取决于数据接口
                batch_x_mark  = batch_x_mark.float().to(self.device)
                batch_y_mark  = batch_y_mark.float().to(self.device)

                # 未来用于辅助损失与监督
                f_dim = -1 if self.args.features == 'MS' else 0
                y_future = batch_y[:, -self.args.pred_len:, f_dim:].contiguous()   # [B, H, D_y]
                # 如果是单变量，你原来就只预测最后一列：保持一致
                y_future = y_future[:, :, 0].unsqueeze(-1)

                batch_text_embeddings = vali_data.get_text_embeddings(index)

                prior_y = torch.from_numpy(vali_data.get_prior_y(index)).float().to(self.device)
                prompt_emb = self.mlp(batch_text_embeddings)  # [B, txt_dim]
                
                # 保存当前的随机状态
                random_weight = self.args.random_weight  # 0 <= random_weight <= 1
                if random_weight > 0:
                    # 设置随机种子
                    seed = 42
                    rng_state = torch.random.get_rng_state()
                    torch.manual_seed(seed)
                    # 生成固定的随机张量
                    random_noise = torch.randn_like(prompt_emb)
                    # 计算随机化后的prompt_emb
                    prompt_emb_randomized = prompt_emb + random_weight * random_noise
                    # 恢复原始的随机状态，不影响其他地方的随机性
                    torch.random.set_rng_state(rng_state)
                
                
                # 解码器的数值部分仍按原逻辑构造（不再拼文本）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向：把 text_emb 与 y_future 交给模型；让模型内部融合与计算 aux
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            y_hat, aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,text_emb=prompt_emb, y_future=y_future, compute_aux=True) 
                        else:
                            y_hat, aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,text_emb=prompt_emb, y_future=y_future, compute_aux=True) 

                else:
                    if self.args.output_attention:
                        y_hat, aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,text_emb=prompt_emb, y_future=y_future, compute_aux=True) 

                    else:
                        y_hat, aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,text_emb=prompt_emb, y_future=y_future, compute_aux=True) 
                        
                # 取监督维度（与你原来一致）
                y_hat = y_hat[:, :, f_dim:]
                y_hat = y_hat[:, :, 0].unsqueeze(-1)

                m = self._core_model()
                
                weight_forecasting = (1-self.prompt_weight) + self.lmb_cot + self.lmb_delta + self.lmb_entropy + self.lmb_tv + self.args.delta + self.args.random_weight
                
                if self.use_prior_blend:
                    y_hat = weight_forecasting * y_hat + (1-weight_forecasting) * prior_y

                # 监督目标
                y_true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                y_true = y_true[:, :, 0].unsqueeze(-1)

                # 基础 MSE
                loss = criterion(y_hat, y_true)

                # 加上辅助损失（权重从模型里拿）
                
                if aux:
                    if 'loss_cot' in aux:     loss = loss + self.loss_weight*self.lmb_cot     * aux['loss_cot']
                    if 'loss_delta' in aux:   loss = loss + self.loss_weight*self.lmb_delta   * aux['loss_delta']
                    if 'loss_entropy' in aux: loss = loss + self.loss_weight*self.lmb_entropy * aux['loss_entropy']
                    if 'loss_tv' in aux:      loss = loss + self.loss_weight*self.lmb_tv      * aux['loss_tv']

                total_loss.append(loss.item())

                if all_metric:
                    pred = y_hat.detach().cpu().numpy()
                    true = y_true.detach().cpu().numpy()
                    mae, mse, rmse, mape, mspe = metric(np.array(pred), np.array(true))
                    total_mae.append(mae); total_mse.append(mse); total_rmse.append(rmse)
                    total_mape.append(mape); total_mspe.append(mspe)
        total_loss = np.average(total_loss)
        self.model.train(); self.mlp.train(); self.mlp_proj.train()
        if all_metric:
            return total_loss, np.average(total_mae), np.average(total_mse), np.average(total_rmse), np.average(total_mape), np.average(total_mspe)
        return total_loss


    def train(self, setting):
        total_gpt2_time = 0  # 初始化总的 GPT-2 时间
        start_time = time.time()
        train_data, train_loader = self._get_data(flag='train')
        vali_data,  vali_loader  = self._get_data(flag='val')
        test_data,  test_loader  = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim       = self._select_optimizer()
        model_optim_mlp   = self._select_optimizer_mlp()
        model_optim_proj  = self._select_optimizer_proj()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train(); self.mlp.train(); self.mlp_proj.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                gpt2_start_time = time.time()
                iter_count += 1
                model_optim.zero_grad(); model_optim_mlp.zero_grad(); model_optim_proj.zero_grad()

                batch_x       = batch_x.float().to(self.device)
                batch_y  = batch_y.float().to(self.device)
                prior_y=torch.from_numpy(train_data.get_prior_y(index)).float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                prior_y = torch.from_numpy(train_data.get_prior_y(index)).float().to(self.device)

                batch_text_embeddings = train_data.get_text_embeddings(index)

                prompt_emb = self.mlp(batch_text_embeddings)
                gpt2_end_time = time.time()
                gpt2_time = gpt2_end_time - gpt2_start_time
                total_gpt2_time += gpt2_time  # 累加 GPT-2 时间
                infer_start_time = time.time()
                # if i==0:
                #     print(prompt_emb) 
                
                random_weight = self.args.random_weight  # 0 <= random_weight <= 1
                if random_weight > 0:
                    # 设置随机种子
                    seed = 42
                    rng_state = torch.random.get_rng_state()
                    torch.manual_seed(seed)
                    # 生成固定的随机张量
                    random_noise = torch.randn_like(prompt_emb)
                    # 计算随机化后的prompt_emb
                    prompt_emb_randomized = prompt_emb + random_weight * random_noise
                    # 恢复原始的随机状态，不影响其他地方的随机性
                    torch.random.set_rng_state(rng_state)            


                f_dim = -1 if self.args.features == 'MS' else 0
                y_future = batch_y[:, -self.args.pred_len:, f_dim:].contiguous()
                y_future = y_future[:, :, 0].unsqueeze(-1)

                # print("batch x shape:",batch_x.shape)
                # print("batch y shape:",batch_y.shape)
                # print("prompt_emb shape:",prompt_emb.shape) 
                
                # decoder 的数值输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向
                # print("use amp:",self.args.use_amp)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        y_hat, aux = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            text_emb=prompt_emb, y_future=y_future, compute_aux=True
                        )
                        y_hat = y_hat[:, :, f_dim:]
                        y_hat = y_hat[:, :, 0].unsqueeze(-1)

                        m = self._core_model()
                
                        weight_forecasting = (1-self.prompt_weight) + self.lmb_cot + self.lmb_delta + self.lmb_entropy + self.lmb_tv + + self.args.delta + self.args.random_weight
                        
                        if self.use_prior_blend:
                            
                            y_hat = weight_forecasting * y_hat + (1-weight_forecasting) * prior_y

                        y_true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        y_true = y_true[:, :, 0].unsqueeze(-1)

                        loss = criterion(y_hat, y_true)
                        if aux:
                            if 'loss_cot' in aux:     loss = loss + self.loss_weight*self.lmb_cot     * aux['loss_cot']
                            if 'loss_delta' in aux:   loss = loss + self.loss_weight*self.lmb_delta   * aux['loss_delta']
                            if 'loss_entropy' in aux: loss = loss + self.loss_weight*self.lmb_entropy * aux['loss_entropy']
                            if 'loss_tv' in aux:      loss = loss + self.loss_weight*self.lmb_tv      * aux['loss_tv']
                else:
                    y_hat, aux = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        text_emb=prompt_emb, y_future=y_future, compute_aux=True
                    )
                    infer_end_time = time.time()
                    infer_time = infer_end_time - infer_start_time  # 计算单次推理时间
                    
                    y_hat = y_hat[:, :, f_dim:]
                    y_hat = y_hat[:, :, 0].unsqueeze(-1)

                    m = self._core_model()
                
                    # print(self.lmb_cot,self.lmb_delta,self.lmb_entropy,self.lmb_tv)
                    # print(self.prompt_weight)
                    weight_forecasting = (1-self.prompt_weight) + self.lmb_cot + self.lmb_delta + self.lmb_entropy + self.lmb_tv + self.args.delta + self.args.random_weight
                    

                    # print(weight_forecasting)
                    if self.use_prior_blend:
                        y_hat = weight_forecasting * y_hat + (1-weight_forecasting) * prior_y

                    # print("prior_y:",prior_y[0])
                    
                    y_true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    y_true = y_true[:, :, 0].unsqueeze(-1)

                    # print("ytrue:",y_true[0])
                    
                    loss = criterion(y_hat, y_true)
                   
                    if aux:
                        if 'loss_cot' in aux:     loss = loss + self.loss_weight*self.lmb_cot     * aux['loss_cot']
                        if 'loss_delta' in aux:   loss = loss + self.loss_weight*self.lmb_delta   * aux['loss_delta']
                        if 'loss_entropy' in aux: loss = loss + self.loss_weight*self.lmb_entropy * aux['loss_entropy']
                        if 'loss_tv' in aux:      loss = loss + self.loss_weight*self.lmb_tv      * aux['loss_tv']


                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0; time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    model_optim_mlp.step()
                    model_optim_proj.step()
            
            
            ######绘图部分开始#########        
            # if (epoch + 1) % 3 == 0:
            #     # 取一小批验证数据
            #     batch_x, batch_y, batch_x_mark, batch_y_mark, index = next(iter(vali_loader))
            #     batch_x = batch_x.float().to(self.device)
                
            #     batch_y = batch_y.float().to(self.device)
            #     batch_x_mark = batch_x_mark.float().to(self.device)
            #     batch_y_mark = batch_y_mark.float().to(self.device)
            #     batch_text_embeddings = vali_data.get_text_embeddings(index)

            #     prompt_emb = self.mlp(batch_text_embeddings)
                
            #     print(f"第{epoch+1}的文本信息是:")
            #     print(vali_data.get_text(index)[0])

            #     dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            #     dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            #     # 前向，拿到 aux
            #     y_pred, aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
            #                             text_emb=prompt_emb,
            #                             y_future=batch_y[:, -self.args.pred_len:, :],
            #                             compute_aux=True)

            #     plot_dir = Path("plot_kappa_pi_gate")
            #     plot_dir.mkdir(parents=True, exist_ok=True)
            #     # print("kappa shape:",aux['kappa'].shape)
            #     plot_kappa_heatmap(aux['kappa'].detach().cpu().numpy(), sample=0,savepath=str(plot_dir / f"kappa_e{epoch+1}_{setting[:100]}.png"))
            #     plot_pi_heatmap(aux['Pi'].detach().cpu().numpy(), sample=0,savepath=str(plot_dir / f"pi_e{epoch+1}_{setting[:100]}.png"))
            #     if aux['gate'] is not None:
            #         plot_gate_hist(aux['gate'].detach().cpu().numpy(),savepath=str(plot_dir / f"gate_e{epoch+1}_{setting[:100]}.png"))  
            #     # cross_attn_weights = self.cross_attention(batch_y[:, -self.args.pred_len:, :], prompt_emb)  
            #     # plot_heatmap(cross_attn_weights, title="Cross-Attention Heatmap")
            #     plot_kappa_multiple_lines(aux['kappa'].detach().cpu().numpy(), epoch, setting, str(plot_dir / f"kappa_multiple_lines_e{epoch + 1}_{setting[:100]}.png"))
            #     plot_Tir_line(aux['kappa'].detach().cpu().numpy(), epoch, setting, str(plot_dir / f"y_tir_lines_e{epoch + 1}_{setting[:100]}.png"))
            
            ######绘图部分结束#########    
                    
            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss  = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, _, _, _, _ = self.vali(test_data, test_loader, criterion, all_metric=True)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss (MSE): {test_loss:.7f} Test MAE: {test_mae:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # print(f"Total GPT-2 embedding time during training: {total_gpt2_time:.4f} seconds")
        end_time = time.time()
        total_train_time = end_time - start_time
        # print(f"Total training time: {total_train_time:.4f} seconds")
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval(); self.mlp.eval(); self.mlp_proj.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                batch_x       = batch_x.float().to(self.device)
                batch_y  = batch_y.float().to(self.device)
                batch_x_mark  = batch_x_mark.float().to(self.device)
                batch_y_mark  = batch_y_mark.float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                prior_y = torch.from_numpy(test_data.get_prior_y(index)).float().to(self.device)
                y_future = batch_y[:, -self.args.pred_len:, f_dim:].contiguous()
                y_future = y_future[:, :, 0].unsqueeze(-1)
                batch_text=test_data.get_text(index)
                batch_text_flattened = batch_text_flattened = batch_text.reshape(-1).tolist()
                if self.Doc2Vec==False:
                    tokenized_output = self.tokenizer(
                        batch_text_flattened,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256
                    )
                    language_max_len = tokenized_output['input_ids'].shape[1]
                    input_ids = tokenized_output['input_ids'].view(self.args.batch_size, self.args.seq_len, language_max_len).to(self.device)
                    attn_mask = tokenized_output['attention_mask'].view(self.args.batch_size, self.args.seq_len, language_max_len).to(self.device)
                    prompt_embeddings = self.llm_model.get_input_embeddings()(input_ids)

                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(self.device)
                if self.use_fullmodel:
                    prompt_emb =self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                else:
                    prompt_emb=prompt_embeddings

                if self.Doc2Vec == False:
                    # Expand attn_mask to match prompt_emb dimensions
                    expanded_mask = attn_mask.unsqueeze(-1).expand_as(prompt_emb)

                    if self.pool_type == "avg":
                        # Mask the embeddings by setting padded tokens to 0
                        masked_emb = prompt_emb * expanded_mask
                        valid_counts = expanded_mask.sum(dim=2, keepdim=True).clamp(min=1)
                        pooled_emb = masked_emb.sum(dim=2) / valid_counts.squeeze(2)
                        prompt_emb = pooled_emb

                    elif self.pool_type == "max":
                        # Mask the embeddings by setting padded tokens to a very small value
                        masked_emb = prompt_emb.masked_fill(expanded_mask == 0, float('-inf'))
                        pooled_emb, _ = masked_emb.max(dim=2)
                        prompt_emb = pooled_emb

                    elif self.pool_type == "min":
                        # Mask the embeddings by setting padded tokens to a very large value
                        masked_emb = prompt_emb.masked_fill(expanded_mask == 0, float('inf'))
                        pooled_emb, _ = masked_emb.min(dim=2)
                        prompt_emb = pooled_emb
                else:
                    prompt_emb = prompt_emb
                    
                prompt_emb = self.mlp(prompt_emb)                  # [B, txt_dim]
                
                random_weight = self.args.random_weight  # 0 <= random_weight <= 1
                if random_weight > 0:
                    # 设置随机种子
                    seed = 42
                    rng_state = torch.random.get_rng_state()
                    torch.manual_seed(seed)
                    # 生成固定的随机张量
                    random_noise = torch.randn_like(prompt_emb)
                    # 计算随机化后的prompt_emb
                    prompt_emb_randomized = prompt_emb + random_weight * random_noise
                    # 恢复原始的随机状态，不影响其他地方的随机性
                    torch.random.set_rng_state(rng_state)
                
                # 解码器数值部分
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向（不需要 aux 也行，但这里开着以便导出门控/核等可视化）
                y_hat, aux = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    text_emb=prompt_emb, y_future=y_future, compute_aux=False
                )
                y_hat = y_hat[:, :, f_dim:]
                y_hat = y_hat[:, :, 0].unsqueeze(-1)

                m = self._core_model()
                
                weight_forecasting = (1-self.prompt_weight) + self.lmb_cot + self.lmb_delta + self.lmb_entropy + self.lmb_tv + self.args.delta + self.args.random_weight
                    # print(weight_forecasting)
                if self.use_prior_blend:
                    y_hat = weight_forecasting * y_hat + (1-weight_forecasting) * prior_y

                y_true = batch_y[:, -self.args.pred_len:, :].to(self.device)

                # 还原尺度（如开启 inverse）
                out_np = y_hat.detach().cpu().numpy()
                true_np= y_true.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    # print("_______++++++++___________")
                    # print(shape)
                    shape = out_np.shape
                    out_np  = test_data.inverse_transform(out_np.squeeze(0)).reshape(shape)
                    true_np = test_data.inverse_transform(true_np.squeeze(0)).reshape(shape)

                out_np  = out_np[:, :, f_dim:]
                true_np = true_np[:, :, f_dim:]

                # print("out_np.shape:,",out_np.shape)
                # print("true_np.shape:,",true_np.shape)
                
                pred = out_np
                true = true_np
                
                preds.append(out_np)
                trues.append(true_np)
                
                # print("pred shape:",pred.shape)
                # print("true shape:",true.shape)

                if i > 0 or i == 0:
                    input = batch_x.detach().cpu().numpy()
                    # print("input:",input.shape)
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        
                    # print("visualizing sample: ", i)
                    
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds); trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open(self.args.save_name, 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
            f.write('\n\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mse
