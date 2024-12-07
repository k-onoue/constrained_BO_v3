#!/bin/bash

# File to execute
EXE_FILE="experiments/tucker-constrained.py"

# Function to run the experiment
run_experiment() {
    local function=$1
    local timestamp=$2
    local dimension=$3
    local iter_bo=$4
    local seed=$5
    local map_option=$6
    local acq_trade_off_param=$7
    local acq_batch_size=$8
    local acq_maximize=$9
    local tucker_rank=${10}
    local tucker_als_iterations=${11}
    local tucker_mask_ratio=${12}
    local decomp_num=${13}
    local n_startup_trials=${14}
    local unique_sampling=${15}
    local include_observed_points=${16}
    local acquisition_function=${17}

    # Build the command
    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --function "$function"
        --dimension "$dimension"
        --iter_bo "$iter_bo"
        --seed "$seed"
        --map_option "$map_option"
        --acq_trade_off_param "$acq_trade_off_param"
        --acq_batch_size "$acq_batch_size"
        --tucker_rank "$tucker_rank"
        --tucker_als_iterations "$tucker_als_iterations"
        --tucker_mask_ratio "$tucker_mask_ratio"
        --n_startup_trials "$n_startup_trials"
        --acquisition_function "$acquisition_function"
    )

    # Add optional flags if enabled
    [ "$acq_maximize" = true ] && cmd+=(--acq_maximize)
    [ "$unique_sampling" = true ] && cmd+=(--unique_sampling)
    [ "$include_observed_points" = true ] && cmd+=(--include_observed_points)

    # Execute the command in the background
    "${cmd[@]}" &
}

# Initialize timestamp and results directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"  # Copy script for reproducibility

# Experiment parameters
tucker_rank=3
tucker_mask_ratio=0.9
decomp_num=10
acquisition_function="ei"  # "ei" or "ucb"
acq_trade_off_param=1.0
acq_batch_size=1
acq_maximize=false
tucker_als_iterations=100
n_startup_trials=1
unique_sampling=false
include_observed_points=false
iter_bo=2000

# Functions, dimensions, and seeds
functions=("warcraft")
dimensions=(4)
map_options=(1 2)
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Run experiments for each function
for function in "${functions[@]}"; do
    case $function in
        "sphere" | "ackley")
            for dimension in "${dimensions[@]}"; do
                for seed in "${seed_list[@]}"; do
                    run_experiment "$function" "$timestamp" "$dimension" "$iter_bo" "$seed" 1 \
                        "$acq_trade_off_param" "$acq_batch_size" "$acq_maximize" \
                        "$tucker_rank" "$tucker_als_iterations" "$tucker_mask_ratio" \
                        "$decomp_num" "$n_startup_trials" \
                        "$unique_sampling" "$include_observed_points" "$acquisition_function"
                done
                # Wait for current dimension experiments to finish
                wait
            done
            ;;
        "warcraft")
            for map_option in "${map_options[@]}"; do
                for seed in "${seed_list[@]}"; do
                    run_experiment "$function" "$timestamp" 2 "$iter_bo" "$seed" "$map_option" \
                        "$acq_trade_off_param" "$acq_batch_size" "$acq_maximize" \
                        "$tucker_rank" "$tucker_als_iterations" "$tucker_mask_ratio" \
                        "$decomp_num" "$n_startup_trials" \
                        "$unique_sampling" "$include_observed_points" "$acquisition_function"
                done
                # Wait for current map option experiments to finish
                wait
            done
            ;;
    esac
done

# Wait for all background processes to complete
wait

# Record completion
completion_file="$results_dir/completion.txt"
echo "All tasks completed at $(date)" > "$completion_file"