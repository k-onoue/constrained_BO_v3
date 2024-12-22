#!/bin/bash

# File to execute
EXE_FILE="experiments/parafac.py"

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
    local cp_rank=${10}
    local cp_als_iterations=${11}
    local cp_mask_ratio=${12}
    local decomp_num=${13}
    local n_startup_trials=${14}
    local unique_sampling=${15}
    local include_observed_points=${16}
    local acquisition_function=${17}
    local plot_save_dir=${18}

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
        --cp_rank "$cp_rank"
        --cp_als_iterations "$cp_als_iterations"
        --cp_mask_ratio "$cp_mask_ratio"
        --n_startup_trials "$n_startup_trials"
        --acquisition_function "$acquisition_function"
        --decomp_num "$decomp_num"
        --plot_save_dir "$exp_dir"
    )

    [ "$acq_maximize" = true ] && cmd+=(--acq_maximize)
    [ "$unique_sampling" = true ] && cmd+=(--unique_sampling)
    [ "$include_observed_points" = true ] && cmd+=(--include_observed_points)

    echo "Running experiment in $exp_dir"
    "${cmd[@]}" 2>&1 | tee "$exp_dir/experiment.log" &
}

# Initialize directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment parameters
functions=("warcraft")
dimensions=(4)
map_options=(1)
seed_list=(0 1 2 3 4 5 6 7 8 9)
iter_bo=2000

# CP decomposition parameters
cp_rank=3
cp_als_iterations=100
cp_mask_ratio=0.9
decomp_num=10

# Acquisition function parameters
acquisition_function="ts"
acq_trade_off_param=1.0
acq_batch_size=1
acq_maximize=false

# Sampling parameters
n_startup_trials=1
unique_sampling=false
include_observed_points=false

# Run experiments
for function in "${functions[@]}"; do
    case $function in
        "sphere" | "ackley")
            for dimension in "${dimensions[@]}"; do
                for seed in "${seed_list[@]}"; do
                    run_experiment "$function" "$timestamp" "$dimension" "$iter_bo" "$seed" 1 \
                        "$acq_trade_off_param" "$acq_batch_size" "$acq_maximize" \
                        "$cp_rank" "$cp_als_iterations" "$cp_mask_ratio" \
                        "$decomp_num" "$n_startup_trials" \
                        "$unique_sampling" "$include_observed_points" \
                        "$acquisition_function" "$plot_save_dir"
                done
                wait
            done
            ;;
        "warcraft")
            for map_option in "${map_options[@]}"; do
                for seed in "${seed_list[@]}"; do
                    run_experiment "$function" "$timestamp" 2 "$iter_bo" "$seed" "$map_option" \
                        "$acq_trade_off_param" "$acq_batch_size" "$acq_maximize" \
                        "$cp_rank" "$cp_als_iterations" "$cp_mask_ratio" \
                        "$decomp_num" "$n_startup_trials" \
                        "$unique_sampling" "$include_observed_points" \
                        "$acquisition_function" "$plot_save_dir"
                done
                wait
            done
            ;;
    esac
done

# Record completion and parameters
cat > "$results_dir/experiment_config.txt" << EOL
Timestamp: $timestamp
Functions: ${functions[*]}
CP rank: $cp_rank
CP mask ratio: $cp_mask_ratio
Acquisition function: $acquisition_function
Completed: $(date)
EOL

echo "All experiments completed!"