#!/bin/bash

# Path to the benchmark script
EXE_FILE="experiments/tf.py"

# Function to run a single experiment
run_experiment() {
    local function=$1
    local timestamp=$2
    local seed=$3
    local map_option=$4
    local tf_method=$5
    local tf_rank=$6
    local tf_fill_method=$7

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --function "$function"
        --seed "$seed"
        --map_option "$map_option"
        --tf_method "$tf_method"
        --tf_rank "$tf_rank"
        --tf_fill_method "$tf_fill_method"
        --iter_bo 300
        # TF parameters
        --tf_lr 0.01
        --tf_max_iter 1000
        --tf_tol 1e-5
        --tf_reg_lambda 1e-3
        --tf_constraint_lambda 1.0
        # Sampler parameters
        --decomp_iter_num 10
        --mask_ratio 0.9
        --n_startup_trials 10
        # Acquisition function parameters
        --acquisition_function "ucb"
        --acq_trade_off_param 3.0
    )

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Initialize timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"

# Experiment configurations
function="warcraft"
map_options=(1)
tf_methods=("cp")
tf_ranks=(3)
tf_fill_methods=("zero")
seed_list=(0)

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for tf_fill_method in "${tf_fill_methods[@]}"; do
                for seed in "${seed_list[@]}"; do
                    run_experiment "$function" "$timestamp" "$seed" "$map_option" \
                        "$tf_method" "$tf_rank" "$tf_fill_method"
                done
            done
        done
    done
done

echo "All experiments completed."