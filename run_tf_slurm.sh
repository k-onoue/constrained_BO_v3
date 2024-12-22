#!/bin/bash -l

# Define experiment parameters
EXE_FILE="experiments/tf.py"
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
timestamp="2024-12-23_00-30-00"
results_dir="/work/$USER/results/$timestamp"
plot_save_dir="$results_dir/plots"

# Create result directories
mkdir -p "$results_dir" "$plot_save_dir"
cp "$0" "$results_dir"

# Experiment configurations
map_options=(2)
seed_list=(0 1 2 3 4 5 6 7 8 9)
tf_methods=("cp")
tf_ranks=(3)
acquisition_function="ts"
mask_ratio=0.9
tf_max_iter="None"

# Flags and other settings
constraint=true
direction=false
decomp_parallel=true
n_startup_trials=1
iter_bo=2000

# Function to submit a job
submit_job() {
    local seed=$1
    local map_option=$2
    local tf_method=$3
    local tf_rank=$4

    sbatch <<EOF
#!/bin/bash -l
#SBATCH -p cluster_short
#SBATCH -J tf_experiment_${seed}
#SBATCH -o $results_dir/output_${seed}.log
#SBATCH -e $results_dir/error_${seed}.log
#SBATCH -c 4
#SBATCH --time=4:00:00

module load python/3.8  # Load appropriate modules
module load some_other_module  # Replace with required modules

python3 "$EXE_FILE" \
    --timestamp "$timestamp" \
    --function "warcraft" \
    --seed "$seed" \
    --map_option "$map_option" \
    --tf_method "$tf_method" \
    --tf_rank "$tf_rank" \
    --tf_lr 0.01 \
    --tf_tol 1e-6 \
    --tf_reg_lambda 0 \
    --tf_constraint_lambda 3.0 \
    --decomp_iter_num 10 \
    --mask_ratio "$mask_ratio" \
    --n_startup_trials "$n_startup_trials" \
    --acquisition_function "$acquisition_function" \
    --acq_trade_off_param 1.0 \
    --iter_bo "$iter_bo" \
    --plot_save_dir "$plot_save_dir" \
    $( [ "$tf_max_iter" != "None" ] && echo "--tf_max_iter $tf_max_iter" ) \
    $( [ "$constraint" = true ] && echo "--constraint" ) \
    $( [ "$direction" = true ] && echo "--direction" ) \
    $( [ "$decomp_parallel" = true ] && echo "--decomp_parallel" )
EOF
}

# Submit jobs
for map_option in "${map_options[@]}"; do
    for tf_method in "${tf_methods[@]}"; do
        for tf_rank in "${tf_ranks[@]}"; do
            for seed in "${seed_list[@]}"; do
                submit_job "$seed" "$map_option" "$tf_method" "$tf_rank"
            done
        done
    done
done

echo "All jobs have been submitted!"

