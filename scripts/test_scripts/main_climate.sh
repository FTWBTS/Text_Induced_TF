#!/bin/bash

# Define model list, GPU, and other paths
all_models=("PatchTST" "DLinear" "iTransformer" "FEDformer" "TimeMixer" "TimeBridge" "FITS" "WPMixer")

GPU=0
root_path=./data
seeds=(2025)
datasets=("Climate")
current_dir=$(pwd)

# TICAL parameters
prior_weight=0.5
text_emb=8
pred_lengths=(6)

use_text=True
TICAL_k=4
TICAL_shape_types=4
TICAL_kernel_emb_dim=64
TICAL_cot_eps=0.1
TICAL_cot_iters=50
TICAL_cot_bandwidth=6
TICAL_cot_alpha=1.0
TICAL_cot_beta=0.02
TICAL_lmb_cot=0.1
TICAL_lmb_delta=0.05
TICAL_lmb_entropy=0.03
TICAL_lmb_tv=0.02
TICAL_kernel_H_scale=0.2
TICAL_gate_weight=1e-2
TICAL_gate_dim=128
loss_function='mae'
adjust_loss=1e-1





# Loop over seeds, models, datasets, and prediction lengths
for seed in "${seeds[@]}"
do
    for model_name in "${all_models[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            data_path=${dataset}.csv
            model_id=${model_name}_${dataset}


            for pred_len in "${pred_lengths[@]}"
            do
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
                    --save_name result_climate_gpt2 \
                    --llm_model GPT2 \
                    --huggingface_token NA \
                    --train_epochs 20 \
                    --patience 5
            done
        done
    done
done
