#!/bin/bash

#SBATCH -p cluster_short       # 使用するパーティション
#SBATCH -c 2                   # 1ジョブあたりのCPUコア数
#SBATCH --time=4:00:00         # 最大実行時間（必要に応じて調整）
#SBATCH --output=logs/%x-%j.out  # 標準出力ログ
#SBATCH --error=logs/%x-%j.err   # エラーログ
#SBATCH --job-name=benchmark_constrained   # ジョブ名
#SBATCH --array=0-269          # ジョブ配列 (計算タスクの総数に応じて設定)

# 必要に応じてディレクトリを作成
mkdir -p logs

# グローバルにタイムスタンプを定義し固定
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
timestamp="2024-12-06_17-00-00"
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"  # Copy script for reproducibility

# 各実行のパラメータを定義
functions=("warcraft")
dimensions=(4)
map_options=(1 2 3)
seed_list=(0 1 2 3 4 5 6 7 8 9)
sampler_list=("tpe" "random" "gp")

# General experiment parameters
iter_bo=2000
n_startup_trials=1

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
python3 experiments/benchmark-constrained.py \
    --timestamp "$timestamp" \
    --function "$function" \
    --sampler "$sampler" \
    --dimension "$dimension" \
    --iter_bo "$iter_bo" \
    --seed "$seed" \
    --map_option "$map_option" \
    --n_startup_trials "$n_startup_trials"

# タスク完了の記録
completion_file="$results_dir/completion.txt"
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)" >> "$completion_file"