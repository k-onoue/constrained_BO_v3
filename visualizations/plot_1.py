import os
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Function to convert hex to rgba
def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

# Plot and save figure function
def plot_and_save_fig(sampler_files, db_folder, color_scale, save_path, width=800, height=600):
    # Initialize a figure for the plot
    fig = go.Figure()

    # Loop over each sampler and assign colors from the Plotly qualitative scale
    for idx, (sampler, db_files) in enumerate(sampler_files.items()):

        # Initialize a list to store dataframes for each seed
        dataframes = []

        # Loop over each file, load the study, and calculate best values
        for db_file in db_files:
            path = os.path.join(db_folder, db_file)
            storage_url = f"sqlite:///{path}"

            # Load the study
            study_name = db_file.split('_bo_')[1].replace('.db', '')
            study = optuna.load_study(study_name="bo_" + study_name, storage=storage_url)

            # Get the trials as a dataframe
            df = study.trials_dataframe().copy()

            # Compute the best values for each iteration
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                df['best_value'] = df['value'].cummax()
            else:
                df['best_value'] = df['value'].cummin()

            # Store the dataframe for the current seed
            dataframes.append(df[['number', 'best_value']])

        # Merge dataframes by iteration number
        merged_df = pd.concat(dataframes, axis=1, keys=[f'seed_{i}' for i in range(len(db_files))])

        # Compute mean and standard deviation of best values across seeds
        best_values = merged_df.xs('best_value', axis=1, level=1)
        mean_best = best_values.mean(axis=1)
        std_best = best_values.std(axis=1)

        # Compute upper and lower bounds for shaded area
        upper_bound = mean_best + std_best
        lower_bound = mean_best - std_best

        # Add the mean line for the current sampler using the color from the color scale
        fig.add_trace(go.Scatter(
            x=best_values.index,
            y=mean_best,
            mode='lines',
            name=f'Mean Best Value ({sampler})',
            line=dict(color=color_scale[idx])
        ))

        # Add the shaded error region for the current sampler without a legend
        fig.add_trace(go.Scatter(
            x=best_values.index,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),  # No line for upper bound
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=best_values.index,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),  # No line for lower bound
            fill='tonexty',  # Fill the area between the upper and lower bound
            fillcolor=hex_to_rgba(color_scale[idx]),  # Convert hex to rgba with transparency
            showlegend=False
        ))

    # Set plot labels and title
    fig.update_layout(
        title="Best Value with Shaded Error Region for Different Samplers",
        xaxis_title="Iteration",
        yaxis_title="Best Value",
        template="plotly_white",
        legend=dict(
            orientation="h",
            x=0,
            y=1
        ),
        width=width,
        height=height
    )

    # Save the figure to the specified path
    fig.write_image(save_path)


# Usage example
if __name__ == '__main__':
    result_path = "/Users/keisukeonoue/ws/constrained_BO_v2/results"
    db_folder = os.path.join(result_path, "dbs")
    
    # sampler_files = {
    #     "random": ['2024-10-12_15-27-35_bo_benchmark_random_seed0.db', '2024-10-12_15-27-44_bo_benchmark_random_seed1.db', '2024-10-12_15-27-53_bo_benchmark_random_seed2.db', '2024-10-12_15-28-02_bo_benchmark_random_seed3.db', '2024-10-12_15-28-10_bo_benchmark_random_seed4.db'],
    #     "tpe": ['2024-10-12_15-29-04_bo_benchmark_tpe_seed0.db', '2024-10-12_15-29-15_bo_benchmark_tpe_seed1.db', '2024-10-12_15-29-26_bo_benchmark_tpe_seed2.db', '2024-10-12_15-29-38_bo_benchmark_tpe_seed3.db', '2024-10-12_15-29-50_bo_benchmark_tpe_seed4.db'],
    #     "gp": ['2024-10-13_10-30-10_bo_benchmark_gp_seed0.db', '2024-10-13_10-31-37_bo_benchmark_gp_seed1.db', '2024-10-13_10-33-12_bo_benchmark_gp_seed2.db', '2024-10-13_10-34-52_bo_benchmark_gp_seed3.db', '2024-10-13_10-36-29_bo_benchmark_gp_seed4.db'],
    #     "parafac": ['2024-10-12_15-32-23_bo_parafac_seed0.db', '2024-10-12_15-33-00_bo_parafac_seed1.db', '2024-10-12_15-33-37_bo_parafac_seed2.db', '2024-10-12_15-34-14_bo_parafac_seed3.db', '2024-10-12_15-34-52_bo_parafac_seed4.db']
    # }


    sampler_files = {
        "random": [
            '2024-10-13_11-29-06_bo_benchmark_random_map2_seed0.db',
            '2024-10-13_11-29-18_bo_benchmark_random_map2_seed1.db',
            '2024-10-13_11-29-31_bo_benchmark_random_map2_seed2.db',
            '2024-10-13_11-29-43_bo_benchmark_random_map2_seed3.db',
            '2024-10-13_11-29-57_bo_benchmark_random_map2_seed4.db'
        ],
        "tpe": [
            '2024-10-13_11-30-12_bo_benchmark_tpe_map2_seed0.db',
            '2024-10-13_11-30-28_bo_benchmark_tpe_map2_seed1.db',
            '2024-10-13_11-30-47_bo_benchmark_tpe_map2_seed2.db',
            '2024-10-13_11-31-04_bo_benchmark_tpe_map2_seed3.db',
            '2024-10-13_11-31-22_bo_benchmark_tpe_map2_seed4.db'
        ],
        "gp": [
            '2024-10-13_11-31-39_bo_benchmark_gp_map2_seed0.db',
            '2024-10-13_11-34-55_bo_benchmark_gp_map2_seed1.db',
            '2024-10-13_11-38-50_bo_benchmark_gp_map2_seed2.db',
            '2024-10-13_11-42-44_bo_benchmark_gp_map2_seed3.db',
            '2024-10-13_11-46-30_bo_benchmark_gp_map2_seed4.db'
        ],
        "parafac": [
            '2024-10-13_11-19-47_bo_parafac_map2_seed0.db',
            '2024-10-13_11-51-05_bo_parafac_map2_seed1.db',
            '2024-10-13_12-49-19_bo_parafac_map2_seed2.db',
            '2024-10-13_13-16-30_bo_parafac_map2_seed3.db',
            '2024-10-13_13-46-07_bo_parafac_map2_seed4.db'
        ],
        # "bruteforce": [
        #     '2024-10-13_11-50-22_bo_benchmark_bruteforce_map2_seed0.db'
        # ]
    }

    color_scale = px.colors.qualitative.Plotly
    save_path = os.path.join(result_path, "images", "best_values_plot.png")
    
    plot_and_save_fig(
        sampler_files, 
        db_folder, 
        color_scale, 
        save_path,
        width=800,
        height=600)
