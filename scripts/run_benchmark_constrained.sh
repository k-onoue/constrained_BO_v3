#!/bin/bash

EXE_FILE="experiments/benchmark-constrained.py"

run_benchmark() {
    local function=$1
    local timestamp=$2
    local sampler=$3
    local dimension=$4
    local iter_bo=$5
    local seed=$6
    local map_option=$7
    local n_init_violation_paths=$8
    local n_startup_trials=$9

    # Restrict iter_bo to a maximum of 500 for gp sampler
    if [ "$sampler" == "gp" ] && [ "$iter_bo" -gt 500 ]; then
        iter_bo=500
    fi

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --function "$function"
        --sampler "$sampler"
        --dimension "$dimension"
        --iter_bo "$iter_bo"
        --n_init_violation_paths "$n_init_violation_paths"
        --seed "$seed"
        --map_option "$map_option"
        --n_startup_trials "$n_startup_trials"
    )

    # Execute the command in the background
    echo "Running: ${cmd[*]}"
    "${cmd[@]}" &
}

# Initialize timestamp and results directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"  # Copy script for reproducibility

# General experiment parameters
iter_bo=2000
n_startup_trials=1
n_init_violation_paths=1000  # Number of initial violation paths

# Experiment configurations
# sampler_list=("tpe" "gp")
sampler_list=("tpe")
functions=("warcraft")
dimensions=(4)
map_options=(1 2)
seed_list=(0 1 2 3 4 5 6 7 8 9) 

# Main experiment loop
for function in "${functions[@]}"; do
    case $function in
        "sphere" | "ackley")
            for dimension in "${dimensions[@]}"; do
                for sampler in "${sampler_list[@]}"; do
                    for seed in "${seed_list[@]}"; do
                        run_benchmark "$function" "$timestamp" "$sampler" "$dimension" "$iter_bo" "$seed" 1 "$n_init_violation_paths" "$n_startup_trials"
                    done
                    # Wait for current sampler's processes to complete
                    wait
                done
            done
            ;;
        "warcraft")
            for map_option in "${map_options[@]}"; do
                for sampler in "${sampler_list[@]}"; do
                    for seed in "${seed_list[@]}"; do
                        run_benchmark "$function" "$timestamp" "$sampler" 2 "$iter_bo" "$seed" "$map_option" "$n_init_violation_paths" "$n_startup_trials"
                    done
                    # Wait for current sampler's processes to complete
                    wait
                done
            done
            ;;
    esac
done