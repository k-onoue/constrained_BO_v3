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
    local constraint=$7
    local direction=$8
    local plot_save_dir=$9
    local decomp_parallel=${10}
    local tf_max_iter=${11}  # Added parameter

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --function "$function"
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
        --n_startup_trials 10
        # Acquisition function parameters
        --acquisition_function "ucb"
        --acq_trade_off_param 3.0
        # Other parameters
        --iter_bo 500
        --plot_save_dir "$plot_save_dir"
    )

    # Add tf_max_iter if not None
    if [ "$tf_max_iter" != "None" ]; then
        cmd+=(--tf_max_iter "$tf_max_iter")
    fi

    # Add optional flags
    [ "$constraint" = true ] && cmd+=(--constraint)
    [ "$direction" = true ] && cmd+=(--direction)
    [ "$decomp_parallel" = true ] && cmd+=(--decomp_parallel)

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}


# Initialize timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/temp_plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
function="warcraft"
map_options=(1 2)
tf_methods=("cp")
tf_ranks=(3)
seed_list=(0 1 2 3 4 5 6 7 8 9)
constraint=false  # Use constraint for warcraft
direction=false  # Minimize objective
decomp_parallel=false  # Enable parallel decomposition
tf_max_iter=None  # Set to integer or "None"

# Run experiments
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                run_experiment "$function" "$timestamp" "$seed" "$map_option" \
                    "$tf_method" "$tf_rank" \
                    "$constraint" "$direction" "$plot_save_dir" "$decomp_parallel" \
                    "$tf_max_iter" &
            done

            # Wait for all experiments to finish
            wait
        done
    done
done


wait
echo "All experiments completed!"