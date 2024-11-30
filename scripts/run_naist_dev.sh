#!/bin/bash

#SBATCH -p cluster_short       # 使用するパーティション
#SBATCH -c 2                   # 1ジョブあたりのCPUコア数
#SBATCH --time=4:00:00         # 最大実行時間（必要に応じて調整）
#SBATCH --output=logs/%x-%j.out  # 標準出力ログ
#SBATCH --error=logs/%x-%j.err   # エラーログ
#SBATCH --job-name=benchmark   # ジョブ名
#SBATCH --array=0-719          # ジョブ配列 (計算タスクの総数に応じて設定)

# 必要に応じてディレクトリを作成
mkdir -p logs

# グローバルにタイムスタンプを定義し固定
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
timestamp="2024-11-28_10-30-00"
results_dir="results/$timestamp"
mkdir -p "$results_dir"

# 各実行のパラメータを定義
functions=("warcraft" "sphere" "ackley")
dimensions=(2 3 4 5 6 7 8 9)
map_options=(1 2 3)
sampler_list=("botorch")
seed_list=(0 1 2 3 4 5 6 7 8 9)

# 設定パラメータの総数を計算
num_functions=${#functions[@]}
num_dimensions=${#dimensions[@]}
num_samplers=${#sampler_list[@]}
num_seeds=${#seed_list[@]}
num_map_options=${#map_options[@]}

# 配列インデックスに基づく計算パラメータの設定
index=$SLURM_ARRAY_TASK_ID
total_tasks=$((num_functions * num_dimensions * num_samplers * num_seeds + num_map_options * num_samplers * num_seeds))

if [ $index -ge $total_tasks ]; then
    echo "Index exceeds total tasks. Exiting."
    exit 1
fi

function_idx=$((index / (num_dimensions * num_samplers * num_seeds + num_map_options * num_samplers * num_seeds)))
remainder=$((index % (num_dimensions * num_samplers * num_seeds + num_map_options * num_samplers * num_seeds)))
function=${functions[$function_idx]}

if [ "$function" == "sphere" ] || [ "$function" == "ackley" ]; then
    dim_idx=$((remainder / (num_samplers * num_seeds)))
    remainder=$((remainder % (num_samplers * num_seeds)))
    sampler_idx=$((remainder / num_seeds))
    seed_idx=$((remainder % num_seeds))

    dimension=${dimensions[$dim_idx]}
    sampler=${sampler_list[$sampler_idx]}
    seed=${seed_list[$seed_idx]}

    python3 experiments/benchmark.py \
        --timestamp "$timestamp" \
        --function "$function" \
        --sampler "$sampler" \
        --dimension "$dimension" \
        --iter_bo 500 \
        --seed "$seed" \
        --map_option 1 \
        --n_startup_trials 1

elif [ "$function" == "warcraft" ]; then
    map_idx=$((remainder / (num_samplers * num_seeds)))
    remainder=$((remainder % (num_samplers * num_seeds)))
    sampler_idx=$((remainder / num_seeds))
    seed_idx=$((remainder % num_seeds))

    map_option=${map_options[$map_idx]}
    sampler=${sampler_list[$sampler_idx]}
    seed=${seed_list[$seed_idx]}

    python3 experiments/benchmark.py \
        --timestamp "$timestamp" \
        --function "$function" \
        --sampler "$sampler" \
        --dimension 2 \
        --iter_bo 500 \
        --seed "$seed" \
        --map_option "$map_option" \
        --n_startup_trials 1
fi

# タスク完了の記録
completion_file="$results_dir/completion.txt"
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)" >> "$completion_file"


