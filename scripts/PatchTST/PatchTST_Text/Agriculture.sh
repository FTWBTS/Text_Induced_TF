#!/bin/bash

all_models=("PatchTST")

GPU=0
root_path=./data
seeds=(2025)
datasets=("Agriculture")
current_dir=$(pwd)
random_weight=0.3


###############################################################################pred_length = 6##################################################################################################
prior_weight=0.65
text_emb=12
pred_len=6

use_text=True
TICAL_k=8
TICAL_shape_types=8
TICAL_kernel_emb_dim=64
TICAL_cot_eps=0.3
TICAL_cot_iters=50
TICAL_cot_bandwidth=6
TICAL_cot_alpha=1.0
TICAL_cot_beta=0.02
TICAL_lmb_cot=0.2
TICAL_lmb_delta=0.25
TICAL_lmb_entropy=0.13
TICAL_lmb_tv=0.12
delta=-0.5
TICAL_kernel_H_scale=0.2
TICAL_gate_weight=1e-2
TICAL_gate_dim=64
loss_function='mse'
adjust_loss=1e-2


for seed in "${seeds[@]}"
do
    for model_name in "${all_models[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            data_path=${dataset}.csv
            model_id=${model_name}_${dataset}
                echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
                CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path $root_path \
                    --data_path $data_path \
                    --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel} \
                    --model $model_name \
                    --data custom \
                    --seq_len 24 \
                    --label_len 12 \
                    --pred_len $pred_len \
                    --text_emb $text_emb \
                    --prior_weight $prior_weight \
                    --use_text $use_text \
                    --TICAL_k $TICAL_k \
                    --TICAL_shape_types $TICAL_shape_types \
                    --TICAL_kernel_emb_dim $TICAL_kernel_emb_dim \
                    --TICAL_cot_eps $TICAL_cot_eps \
                    --TICAL_cot_iters $TICAL_cot_iters \
                    --TICAL_cot_bandwidth $TICAL_cot_bandwidth \
                    --TICAL_cot_alpha $TICAL_cot_alpha \
                    --TICAL_cot_beta $TICAL_cot_beta \
                    --TICAL_lmb_cot $TICAL_lmb_cot \
                    --TICAL_lmb_delta $TICAL_lmb_delta \
                    --TICAL_lmb_entropy $TICAL_lmb_entropy \
                    --TICAL_lmb_tv $TICAL_lmb_tv \
                    --TICAL_kernel_H_scale $TICAL_kernel_H_scale \
                    --TICAL_gate_weight $TICAL_gate_weight \
                    --TICAL_gate_dim $TICAL_gate_dim \
                    --loss_function $loss_function \
                    --adjust_loss $adjust_loss \
                    --delta $delta \
                    --random_weight $random_weight \
                    --save_name result_Algri_gpt2 \
                    --llm_model GPT2 \
                    --huggingface_token NA \
                    --train_epochs 5 \
                    --patience 5
        done
    done
done






###############################################################################pred_length = 8##################################################################################################
prior_weight=0.45
text_emb=6
pred_len=8

use_text=True
TICAL_k=8
TICAL_shape_types=8
TICAL_kernel_emb_dim=128
TICAL_cot_eps=0.1
TICAL_cot_iters=50
TICAL_cot_bandwidth=6
TICAL_cot_alpha=1.0
TICAL_cot_beta=0.02
TICAL_lmb_cot=0.2
TICAL_lmb_delta=0.25
TICAL_lmb_entropy=0.13
TICAL_lmb_tv=0.12
delta=-0.5
TICAL_kernel_H_scale=0.2
TICAL_gate_weight=1e-2
TICAL_gate_dim=128
loss_function='mse'
adjust_loss=1e-2


for seed in "${seeds[@]}"
do
    for model_name in "${all_models[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            data_path=${dataset}.csv
            model_id=${model_name}_${dataset}
                echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
                CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path $root_path \
                    --data_path $data_path \
                    --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel} \
                    --model $model_name \
                    --data custom \
                    --seq_len 24 \
                    --label_len 12 \
                    --pred_len $pred_len \
                    --text_emb $text_emb \
                    --prior_weight $prior_weight \
                    --use_text $use_text \
                    --TICAL_k $TICAL_k \
                    --TICAL_shape_types $TICAL_shape_types \
                    --TICAL_kernel_emb_dim $TICAL_kernel_emb_dim \
                    --TICAL_cot_eps $TICAL_cot_eps \
                    --TICAL_cot_iters $TICAL_cot_iters \
                    --TICAL_cot_bandwidth $TICAL_cot_bandwidth \
                    --TICAL_cot_alpha $TICAL_cot_alpha \
                    --TICAL_cot_beta $TICAL_cot_beta \
                    --TICAL_lmb_cot $TICAL_lmb_cot \
                    --TICAL_lmb_delta $TICAL_lmb_delta \
                    --TICAL_lmb_entropy $TICAL_lmb_entropy \
                    --TICAL_lmb_tv $TICAL_lmb_tv \
                    --TICAL_kernel_H_scale $TICAL_kernel_H_scale \
                    --TICAL_gate_weight $TICAL_gate_weight \
                    --TICAL_gate_dim $TICAL_gate_dim \
                    --loss_function $loss_function \
                    --adjust_loss $adjust_loss \
                    --delta $delta \
                    --random_weight $random_weight \
                    --save_name result_Algri_gpt2 \
                    --llm_model GPT2 \
                    --huggingface_token NA \
                    --train_epochs 10 \
                    --patience 5
        done
    done
done


###############################################################################pred_length = 10##################################################################################################

prior_weight=0.35
text_emb=8
pred_len=10

use_text=True
TICAL_k=8
TICAL_shape_types=8
TICAL_kernel_emb_dim=128
TICAL_cot_eps=0.1
TICAL_cot_iters=50
TICAL_cot_bandwidth=6
TICAL_cot_alpha=1.0
TICAL_cot_beta=0.02
TICAL_lmb_cot=0.3
TICAL_lmb_delta=0.25
TICAL_lmb_entropy=0.13
TICAL_lmb_tv=0.12
delta=-0.64
TICAL_kernel_H_scale=0.2
TICAL_gate_weight=1e-2
TICAL_gate_dim=128
loss_function='mse'
adjust_loss=1e-2


for seed in "${seeds[@]}"
do
    for model_name in "${all_models[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            data_path=${dataset}.csv
            model_id=${model_name}_${dataset}
                echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
                CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path $root_path \
                    --data_path $data_path \
                    --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel} \
                    --model $model_name \
                    --data custom \
                    --seq_len 24 \
                    --label_len 12 \
                    --pred_len $pred_len \
                    --text_emb $text_emb \
                    --prior_weight $prior_weight \
                    --use_text $use_text \
                    --TICAL_k $TICAL_k \
                    --TICAL_shape_types $TICAL_shape_types \
                    --TICAL_kernel_emb_dim $TICAL_kernel_emb_dim \
                    --TICAL_cot_eps $TICAL_cot_eps \
                    --TICAL_cot_iters $TICAL_cot_iters \
                    --TICAL_cot_bandwidth $TICAL_cot_bandwidth \
                    --TICAL_cot_alpha $TICAL_cot_alpha \
                    --TICAL_cot_beta $TICAL_cot_beta \
                    --TICAL_lmb_cot $TICAL_lmb_cot \
                    --TICAL_lmb_delta $TICAL_lmb_delta \
                    --TICAL_lmb_entropy $TICAL_lmb_entropy \
                    --TICAL_lmb_tv $TICAL_lmb_tv \
                    --TICAL_kernel_H_scale $TICAL_kernel_H_scale \
                    --TICAL_gate_weight $TICAL_gate_weight \
                    --TICAL_gate_dim $TICAL_gate_dim \
                    --loss_function $loss_function \
                    --adjust_loss $adjust_loss \
                    --delta $delta \
                    --random_weight $random_weight \
                    --save_name result_Algri_gpt2 \
                    --llm_model GPT2 \
                    --huggingface_token NA \
                    --train_epochs 10 \
                    --patience 5
        done
    done
done


# ###############################################################################pred_length = 12##################################################################################################

prior_weight=0.58
text_emb=12
pred_len=12

use_text=True
TICAL_k=10
TICAL_shape_types=8
TICAL_kernel_emb_dim=64
TICAL_cot_eps=0.2
TICAL_cot_iters=50
TICAL_cot_bandwidth=6
TICAL_cot_alpha=1.0
TICAL_cot_beta=0.02
TICAL_lmb_cot=0.3
TICAL_lmb_delta=0.35
TICAL_lmb_entropy=0.23
TICAL_lmb_tv=0.02
delta=-0.7
TICAL_kernel_H_scale=0.2
TICAL_gate_weight=1e-2
TICAL_gate_dim=128
loss_function='mse'
adjust_loss=1e-2


for seed in "${seeds[@]}"
do
    for model_name in "${all_models[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            data_path=${dataset}.csv
            model_id=${model_name}_${dataset}
                echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
                CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path $root_path \
                    --data_path $data_path \
                    --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel} \
                    --model $model_name \
                    --data custom \
                    --seq_len 24 \
                    --label_len 12 \
                    --pred_len $pred_len \
                    --text_emb $text_emb \
                    --prior_weight $prior_weight \
                    --use_text $use_text \
                    --TICAL_k $TICAL_k \
                    --TICAL_shape_types $TICAL_shape_types \
                    --TICAL_kernel_emb_dim $TICAL_kernel_emb_dim \
                    --TICAL_cot_eps $TICAL_cot_eps \
                    --TICAL_cot_iters $TICAL_cot_iters \
                    --TICAL_cot_bandwidth $TICAL_cot_bandwidth \
                    --TICAL_cot_alpha $TICAL_cot_alpha \
                    --TICAL_cot_beta $TICAL_cot_beta \
                    --TICAL_lmb_cot $TICAL_lmb_cot \
                    --TICAL_lmb_delta $TICAL_lmb_delta \
                    --TICAL_lmb_entropy $TICAL_lmb_entropy \
                    --TICAL_lmb_tv $TICAL_lmb_tv \
                    --TICAL_kernel_H_scale $TICAL_kernel_H_scale \
                    --TICAL_gate_weight $TICAL_gate_weight \
                    --TICAL_gate_dim $TICAL_gate_dim \
                    --loss_function $loss_function \
                    --adjust_loss $adjust_loss \
                    --delta $delta \
                    --random_weight $random_weight \
                    --save_name result_Algri_gpt2 \
                    --llm_model GPT2 \
                    --huggingface_token NA \
                    --train_epochs 10 \
                    --patience 5
        done
    done
done
