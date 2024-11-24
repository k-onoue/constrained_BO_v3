#!/bin/bash

# File to execute
EXE_FILE="experiments/benchmark.py"

# Function to run the benchmark
run_benchmark() {
    local function=$1
    local timestamp=$2
    local sampler=$3
    local dimension=$4
    local iter_bo=$5
    local seed=$6
    local map_option=$7

    # Build the command
    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --function "$function"
        --sampler "$sampler"
        --dimension "$dimension"
        --iter_bo "$iter_bo"
        --seed "$seed"
        --map_option "$map_option"
        --n_startup_trials "$n_startup_trials"
    )

    # Execute the command in the background
    "${cmd[@]}" &
}

# Initialize timestamp and results directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"
cp "$0" "$results_dir"  # Copy script for reproducibility

# General experiment parameters
iter_bo=500
n_startup_trials=1

# Experiment configurations
# sampler_list=("tpe" "random" "gp" "hgp")
sampler_list=("gp" "hgp")
functions=("warcraft" "sphere" "ackley")  # Add "sphere" and "ackley" as needed
dimensions=(2 3 4 5 6 7 8 9)
map_options=(1 2 3)
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Main experiment loop

for function in "${functions[@]}"; do
    case $function in
        "sphere" | "ackley")
            for dimension in "${dimensions[@]}"; do
                for sampler in "${sampler_list[@]}"; do
                    for seed in "${seed_list[@]}"; do
                        run_benchmark "$function" "$timestamp" "$sampler" "$dimension" "$iter_bo" "$seed" 1
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
                        run_benchmark "$function" "$timestamp" "$sampler" 2 "$iter_bo" "$seed" "$map_option"
                    done

                    # Wait for current sampler's processes to complete
                    wait
                done
            done
            ;;
    esac

    # Wait for current function's processes to complete
    # wait
done



# Wait for all background processes to complete
wait

# Record completion
completion_file="$results_dir/completion.txt"
echo "All tasks completed at $(date)" > "$completion_file"
