#!/bin/bash

# File to execute
EXE_FILE="experiments/benchmark.py"

# Function to run the benchmark
run() {
    local function=$1
    local timestamp=$2
    local sampler=$3
    local dimension=$4
    local iter_bo=$5
    local seed=$6
    local map_option=$7

    cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --function "$function"
        --sampler "$sampler"
        --dimension "$dimension"
        --iter_bo "$iter_bo"
        --seed "$seed"
        --map_option "$map_option"
    )
    "${cmd[@]}" &
}

# Main script
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

sampler_list=("tpe" "random" "gp" "hgp")
seed_list=(0 1 2 3 4 5 6 7 8 9)
iter_bo=20

for function in "sphere" "ackley" "warcraft"; do
    case $function in
        "sphere" | "ackley")
            for dimension in 2 3 4 5 6 7; do
                for sampler in "${sampler_list[@]}"; do
                    for seed in "${seed_list[@]}"; do
                        run "$function" "$timestamp" "$sampler" "$dimension" "$iter_bo" "$seed" 1
                    done
                done
                # Wait for all processes in the current dimension to complete
                wait
            done
            ;;
        "warcraft")
            for map_option in 1 2 3; do
                for sampler in "${sampler_list[@]}"; do
                    for seed in "${seed_list[@]}"; do
                        run "$function" "$timestamp" "$sampler" 2 "$iter_bo" "$seed" "$map_option"
                    done
                done
                # Wait for all processes in the current map option to complete
                wait
            done
            ;;
    esac
done

# Final wait to ensure all background processes are complete
wait
