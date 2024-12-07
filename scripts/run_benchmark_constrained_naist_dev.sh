#!/bin/bash

#SBATCH -p cluster_short       # 使用するパーティション
#SBATCH -c 2                   # 1ジョブあたりのCPUコア数
#SBATCH --time=4:00:00         # 最大実行時間（必要に応じて調整）
#SBATCH --output=logs/%x-%j.out  # 標準出力ログ
#SBATCH --error=logs/%x-%j.err   # エラーログ
#SBATCH --job-name=benchmark_constrained   # ジョブ名
#SBATCH --array=0-59          # ジョブ配列 (計算タスクの総数に応じて設定)

# 必要に応じてディレクトリを作成
mkdir -p logs

# グローバルにタイムスタンプを定義し固定
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
timestamp="2024-12-07_16-50-00"
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"  # Copy script for reproducibility

# 各実行のパラメータを定義
functions=("warcraft")
dimensions=(4)
map_options=(1 2)
seed_list=(0 1 2 3 4 5 6 7 8 9)
sampler_list=("tpe" "random" "gp")

# Experiment parameters
cp_rank=3
cp_mask_ratio=0.9
decomp_num=10
acquisition_function="ei"  # "ei" or "ucb"
acq_trade_off_param=1.0
acq_batch_size=1
acq_maximize=false
cp_als_iterations=100
n_startup_trials=1
unique_sampling=false
include_observed_points=false
iter_bo=2000

# 設定パラメータの総数を計算
num_functions=${#functions[@]}
num_dimensions=${#dimensions[@]}
num_map_options=${#map_options[@]}
num_seeds=${#seed_list[@]}
num_samplers=${#sampler_list[@]}
total_tasks=$((num_functions * num_dimensions * num_map_options * num_seeds * num_samplers))

# 配列インデックスに基づく計算パラメータの設定
index=$SLURM_ARRAY_TASK_ID

if [ $index -ge $total_tasks ]; then
    echo "Index exceeds total tasks. Exiting."
    exit 1
fi

function_idx=$((index / (num_dimensions * num_map_options * num_seeds * num_samplers)))
remainder=$((index % (num_dimensions * num_map_options * num_seeds * num_samplers)))
dimension_idx=$((remainder / (num_map_options * num_seeds * num_samplers)))
remainder=$((remainder % (num_map_options * num_seeds * num_samplers)))
map_option_idx=$((remainder / (num_seeds * num_samplers)))
remainder=$((remainder % (num_seeds * num_samplers)))
sampler_idx=$((remainder / num_seeds))
seed_idx=$((remainder % num_seeds))

function=${functions[$function_idx]}
dimension=${dimensions[$dimension_idx]}
map_option=${map_options[$map_option_idx]}
sampler=${sampler_list[$sampler_idx]}
seed=${seed_list[$seed_idx]}

# Restrict iter_bo to a maximum of 500 for gp sampler
if [ "$sampler" == "gp" ] && [ "$iter_bo" -gt 500 ]; then
    iter_bo=500
fi

# 実験の実行
python3 experiments/parafac-constrained.py \
    --timestamp "$timestamp" \
    --function "$function" \
    --dimension "$dimension" \
    --iter_bo "$iter_bo" \
    --seed "$seed" \
    --map_option "$map_option" \
    --acq_trade_off_param "$acq_trade_off_param" \
    --acq_batch_size "$acq_batch_size" \
    --cp_rank "$cp_rank" \
    --cp_als_iterations "$cp_als_iterations" \
    --cp_mask_ratio "$cp_mask_ratio" \
    --n_startup_trials "$n_startup_trials" \
    --acquisition_function "$acquisition_function" \
    $( [ "$acq_maximize" = true ] && echo "--acq_maximize" ) \
    $( [ "$unique_sampling" = true ] && echo "--unique_sampling" ) \
    $( [ "$include_observed_points" = true ] && echo "--include_observed_points" )

# タスク完了の記録
completion_file="$results_dir/completion.txt"
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)" >> "$completion_file"