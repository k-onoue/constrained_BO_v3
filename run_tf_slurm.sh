#!/bin/bash -l
#SBATCH --job-name=tf_experiment
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=cluster_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00

EXE_FILE="experiments/tf_continual.py"

# Common parameters for argparse
COMMON_ARGS=(
    --function "warcraft"
    --tf_lr 0.01
    --tf_tol 1e-6
    --tf_reg_lambda 0
    --tf_constraint_lambda 1.0
    --decomp_iter_num 10
    --acq_trade_off_param 1.0
    --mask_ratio 1
    --n_startup_trials 1
    --iter_bo 2000
    --acquisition_function "ei"
    --map_option 2
    --tf_method train
    --tf_rank 4
)

# Optional parameters
[ "$tf_max_iter" != "None" ] && COMMON_ARGS+=(--tf_max_iter "$tf_max_iter")
[ "$constraint" = true ] && COMMON_ARGS+=(--constraint)
[ "$direction" = true ] && COMMON_ARGS+=(--direction)

run_experiment() {
    local timestamp=$1
    local seed=$2
    local plot_save_dir=$3

    local cmd=(
        python3 "$EXE_FILE"
        --timestamp "$timestamp"
        --seed "$seed"
        --plot_save_dir "$plot_save_dir"
        "${COMMON_ARGS[@]}"
    )

    echo "Running: ${cmd[*]}"
    eval "${cmd[@]}"
}

# Initialize timestamp and directories
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
results_dir="results/$timestamp"
plot_save_dir="$results_dir/plots"
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Seed list
seed_list=(0 1 2 3 4 5 6 7 8 9)

# Run experiments
for seed in "${seed_list[@]}"; do
    run_experiment "$timestamp" "$seed" "$plot_save_dir"
done

echo "All experiments completed!"
