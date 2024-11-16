#!/bin/bash

# File to execute
EXE_FILE="experiments/parafac.py"

# Function to run the experiment
run() {
    local function=$1
    local timestamp=$2
    local dimension=$3
    local iter_bo=$4
    local seed=$5
    local map_option=$6
    local acq_trade_off_param=$7
    local acq_batch_size=$8
    local acq_maximize=$9
    local cp_rank=${10}
    local cp_als_iterations=${11}
    local cp_mask_ratio=${12}
    local cp_random_dist_type=${13}
    local decomp_num=${14}
    local n_startup_trials=${15}

    cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --function "$function"
        --dimension "$dimension"
        --iter_bo "$iter_bo"
        --seed "$seed"
        --map_option "$map_option"
        --acq_trade_off_param "$acq_trade_off_param"
        --acq_batch_size "$acq_batch_size"
        --acq_maximize "$acq_maximize"
        --cp_rank "$cp_rank"
        --cp_als_iterations "$cp_als_iterations"
        --cp_mask_ratio "$cp_mask_ratio"
        --cp_random_dist_type "$cp_random_dist_type"
        --n_startup_trials "$n_startup_trials"
    )
    "${cmd[@]}" &
}


# Main script
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

iter_bo=20

acq_trade_off_param=1.0
acq_batch_size=10
acq_maximize=false
cp_rank=2
cp_als_iterations=100
cp_mask_ratio=0.1
cp_random_dist_type="uniform"
decomp_num=5
n_startup_trials=1

functions=("sphere" "ackley" "warcraft")
dimensions=(2 3 4 5 6 7)
map_options=(1 2)
seed_list=(0 1 2 3 4 5 6 7 8 9)

for function in "${functions[@]}"; do
    case $function in
        "sphere" | "ackley")
            for dimension in "${dimensions[@]}"; do
                for seed in "${seed_list[@]}"; do
                    run "$function" "$timestamp" "$dimension" "$iter_bo" "$seed" 1 \
                        "$acq_trade_off_param" "$acq_batch_size" "$acq_maximize" \
                        "$cp_rank" "$cp_als_iterations" "$cp_mask_ratio" \
                        "$cp_random_dist_type" "$decomp_num" "$n_startup_trials"
                done
                # Wait for all processes in the current dimension to complete
                wait
            done
            ;;
        "warcraft")
            for map_option in "${map_options[@]}"; do
                for seed in "${seed_list[@]}"; do
                    run "$function" "$timestamp" 2 "$iter_bo" "$seed" "$map_option" \
                        "$acq_trade_off_param" "$acq_batch_size" "$acq_maximize" \
                        "$cp_rank" "$cp_als_iterations" "$cp_mask_ratio" \
                        "$cp_random_dist_type" "$decomp_num" "$n_startup_trials"
                done
                # Wait for all processes in the current map option to complete
                wait
            done
            ;;
    esac
done

# Final wait to ensure all background processes are complete
wait
