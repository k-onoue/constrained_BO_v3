#!/bin/bash

EXE_FILE="experiments/tf.py"

run_experiment() {
    local timestamp=$1
    local seed=$2
    local map_option=$3
    local tf_method=$4
    local tf_rank=$5
    local constraint=$6
    local direction=$7
    local plot_save_dir=$8
    local decomp_parallel=$9
    local tf_max_iter=${10}
    local n_startup_trials=${11}  # Added
    local iter_bo=${12}          # Added

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --function "warcraft"
        --seed "$seed"
        --map_option "$map_option"
        # TF parameters
        --tf_method "$tf_method"
        --tf_rank "$tf_rank"
        --tf_lr 0.01
        --tf_tol 1e-6
        --tf_reg_lambda 0
        --tf_constraint_lambda 1.0
        # Sampler parameters
        --decomp_iter_num 10
        --mask_ratio 0.9
        --n_startup_trials "$n_startup_trials"  # Use parameter
        # Acquisition function parameters
        --acquisition_function "ei"
        --acq_trade_off_param 1.0
        # Other parameters
        --iter_bo "$iter_bo"  # Use parameter
        --plot_save_dir "$plot_save_dir"
    )

    # Add optional flags
    [ "$tf_max_iter" != "None" ] && cmd+=(--tf_max_iter "$tf_max_iter")
    [ "$constraint" = true ] && cmd+=(--constraint)
    [ "$direction" = true ] && cmd+=(--direction)
    [ "$decomp_parallel" = true ] && cmd+=(--decomp_parallel)

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
map_options=(1)
tf_methods=("cp")
tf_ranks=(3)
seed_list=(0 1 2 3 4 5 6 7 8 9)
constraint=false
direction=false
decomp_parallel=true
tf_max_iter=1000

n_startup_trials=1  # Added
iter_bo=2000        # Added

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                run_experiment "$timestamp" "$seed" "$map_option" \
                    "$tf_method" "$tf_rank" \
                    "$constraint" "$direction" "$plot_save_dir" \
                    "$decomp_parallel" "$tf_max_iter" \
                    "$n_startup_trials" "$iter_bo" &
            done
            wait
        done
    done
done

echo "All experiments completed!"