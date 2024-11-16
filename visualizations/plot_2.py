import os

import numpy as np
import optuna
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

from _src import WarcraftObjective


# Inner function for plotting an individual file
def plot_individual_file(db_file, sampler, db_folder, output_dir, width, height, map_shape=(2, 2)):
    path = os.path.join(db_folder, db_file)
    storage_url = f"sqlite:///{path}"

    # Extract seed number from the filename
    seed = db_file.split('_')[-1].replace('.db', '')

    # Load the study
    study_name = db_file.split('_bo_')[1].replace('.db', '')
    study = optuna.load_study(study_name="bo_" + study_name, storage=storage_url)

    study_df = study.trials_dataframe().copy()

    # direction に基づいて best_value 列を追加する
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        # 最大化の場合
        study_df['best_value'] = study_df['value'].cummax()
    else:
        # 最小化の場合
        study_df['best_value'] = study_df['value'].cummin()

    # WarcraftObjective クラスのインスタンスを作成
    weight_matrix = np.random.rand(*map_shape)  # 任意の weight_matrix (適宜変更してください)
    objective = WarcraftObjective(weight_matrix)

    # 各行に対して penalty_type2 を計算し、新しい列に判定結果を追加
    def check_constraint(row, objective):
        # params から direction_matrix を作成 (前の質問での params 取得方法を使用)
        params_columns = [col for col in row.index if col.startswith('params_')]
        direction_matrix = np.array([row[col] for col in params_columns]).reshape(map_shape)  # 2x2 の例、適宜調整してください

        return objective.check_penalty_type2(direction_matrix)

    # study_df に判定結果の列を追加
    study_df['constraint_1'] = study_df.apply(lambda row: check_constraint(row, objective), axis=1)

    # Create a figure
    fig = go.Figure()

    # プロット：青の点で value 列を表示 (constraint_1 == False の部分)
    fig.add_trace(go.Scatter(
        x=study_df[~study_df['constraint_1']].index,  # 行番号を x 軸に
        y=study_df[~study_df['constraint_1']]['value'],  # value 列を y 軸に
        mode='markers',
        name='Value (constraint_1 == False)',
        marker=dict(color='blue')  # 青の点
    ))

    # プロット：オレンジの点で constraint_1 == True の部分を表示
    fig.add_trace(go.Scatter(
        x=study_df[study_df['constraint_1']].index,  # constraint_1 が True の行を x 軸に
        y=study_df[study_df['constraint_1']]['value'],  # value 列を y 軸に
        mode='markers',
        name='Value (constraint_1 == True)',
        marker=dict(color='orange', symbol='circle')  # オレンジの点
    ))

    # プロット：赤い線で best_value 列を表示
    fig.add_trace(go.Scatter(
        x=study_df.index,  # 行番号を x 軸に
        y=study_df['best_value'],  # best_value 列を y 軸に
        mode='lines',
        name='Best Value',
        line=dict(color='rgba(255, 0, 0, 1)', dash="dash")  # 赤色（透明度50%）
    ))

    # 背景を白に設定
    fig.update_layout(
        template="plotly_white",  # 背景を白にする
        title="Optimization Progress",
        xaxis_title="Iteration",
        yaxis_title="Value",
        width=width,
        height=height,
        yaxis=dict(range=[0.3, 2.2]),  # y 軸の範囲を設定
        legend=dict(
            orientation="h",
            x=0,
            y=1
        )
    )

    # Save the figure as an image
    output_path = os.path.join(output_dir, f"{sampler}_{seed}.png")
    pio.write_image(fig, output_path)

    print(f'Image saved to {output_path}')


# Function to generate and save individual optimization history plots with customizable size
def plot_and_save_individual_histories(sampler_files, db_folder, output_dir, width=800, height=600, map_shape=(2, 2)):
    # Loop over each sampler and its file list
    for sampler, db_files in sampler_files.items():
        # Loop over each file (which corresponds to different seeds)
        for db_file in db_files:
            plot_individual_file(db_file, sampler, db_folder, output_dir, width, height, map_shape)



# Usage example
if __name__ == '__main__':
    result_path = "/Users/keisukeonoue/ws/constrained_BO_v2/results_2024-10-12"
    db_folder = os.path.join(result_path, "dbs")
    output_dir = os.path.join(result_path, "images")

    # sampler_files = {
    #     "random": ['2024-10-12_15-27-35_bo_benchmark_random_seed0.db', '2024-10-12_15-27-44_bo_benchmark_random_seed1.db', '2024-10-12_15-27-53_bo_benchmark_random_seed2.db', '2024-10-12_15-28-02_bo_benchmark_random_seed3.db', '2024-10-12_15-28-10_bo_benchmark_random_seed4.db'],
    #     "tpe": ['2024-10-12_15-29-04_bo_benchmark_tpe_seed0.db', '2024-10-12_15-29-15_bo_benchmark_tpe_seed1.db', '2024-10-12_15-29-26_bo_benchmark_tpe_seed2.db', '2024-10-12_15-29-38_bo_benchmark_tpe_seed3.db', '2024-10-12_15-29-50_bo_benchmark_tpe_seed4.db'],
    #     "gp": ['2024-10-13_10-30-10_bo_benchmark_gp_seed0.db', '2024-10-13_10-31-37_bo_benchmark_gp_seed1.db', '2024-10-13_10-33-12_bo_benchmark_gp_seed2.db', '2024-10-13_10-34-52_bo_benchmark_gp_seed3.db', '2024-10-13_10-36-29_bo_benchmark_gp_seed4.db'],
    #     "parafac": ['2024-10-12_15-32-23_bo_parafac_seed0.db', '2024-10-12_15-33-00_bo_parafac_seed1.db', '2024-10-12_15-33-37_bo_parafac_seed2.db', '2024-10-12_15-34-14_bo_parafac_seed3.db', '2024-10-12_15-34-52_bo_parafac_seed4.db']
    # }

    # sampler_files = {
    #     "random": [
    #         '2024-10-13_11-29-06_bo_benchmark_random_map2_seed0.db',
    #         '2024-10-13_11-29-18_bo_benchmark_random_map2_seed1.db',
    #         '2024-10-13_11-29-31_bo_benchmark_random_map2_seed2.db',
    #         '2024-10-13_11-29-43_bo_benchmark_random_map2_seed3.db',
    #         '2024-10-13_11-29-57_bo_benchmark_random_map2_seed4.db'
    #     ],
    #     "tpe": [
    #         '2024-10-13_11-30-12_bo_benchmark_tpe_map2_seed0.db',
    #         '2024-10-13_11-30-28_bo_benchmark_tpe_map2_seed1.db',
    #         '2024-10-13_11-30-47_bo_benchmark_tpe_map2_seed2.db',
    #         '2024-10-13_11-31-04_bo_benchmark_tpe_map2_seed3.db',
    #         '2024-10-13_11-31-22_bo_benchmark_tpe_map2_seed4.db'
    #     ],
    #     "gp": [
    #         '2024-10-13_11-31-39_bo_benchmark_gp_map2_seed0.db',
    #         '2024-10-13_11-34-55_bo_benchmark_gp_map2_seed1.db',
    #         '2024-10-13_11-38-50_bo_benchmark_gp_map2_seed2.db',
    #         '2024-10-13_11-42-44_bo_benchmark_gp_map2_seed3.db',
    #         '2024-10-13_11-46-30_bo_benchmark_gp_map2_seed4.db'
    #     ],
    #     "parafac": [
    #         '2024-10-13_11-19-47_bo_parafac_map2_seed0.db',
    #         '2024-10-13_11-51-05_bo_parafac_map2_seed1.db',
    #         '2024-10-13_12-49-19_bo_parafac_map2_seed2.db',
    #         '2024-10-13_13-16-30_bo_parafac_map2_seed3.db',
    #         '2024-10-13_13-46-07_bo_parafac_map2_seed4.db'
    #     ],
    #     # "bruteforce": [
    #     #     '2024-10-13_11-50-22_bo_benchmark_bruteforce_map2_seed0.db'
    #     # ]
    # }

    sampler_files = {
        "random": [
            '2024-10-12_15-27-35_bo_benchmark_random_seed0.db',
            '2024-10-12_15-27-44_bo_benchmark_random_seed1.db',
            '2024-10-12_15-27-53_bo_benchmark_random_seed2.db',
            '2024-10-12_15-28-02_bo_benchmark_random_seed3.db',
            '2024-10-12_15-28-10_bo_benchmark_random_seed4.db'
        ],
        "tpe": [
            '2024-10-12_15-29-04_bo_benchmark_tpe_seed0.db',
            '2024-10-12_15-29-15_bo_benchmark_tpe_seed1.db',
            '2024-10-12_15-29-26_bo_benchmark_tpe_seed2.db',
            '2024-10-12_15-29-38_bo_benchmark_tpe_seed3.db',
            '2024-10-12_15-29-50_bo_benchmark_tpe_seed4.db'
        ],
        "gp": [
            '2024-10-13_10-30-10_bo_benchmark_gp_seed0.db',
            '2024-10-13_10-31-37_bo_benchmark_gp_seed1.db',
            '2024-10-13_10-33-12_bo_benchmark_gp_seed2.db',
            '2024-10-13_10-34-52_bo_benchmark_gp_seed3.db',
            '2024-10-13_10-36-29_bo_benchmark_gp_seed4.db'
        ],
        "parafac": [
            '2024-10-12_15-32-23_bo_parafac_seed0.db',
            '2024-10-12_15-33-00_bo_parafac_seed1.db',
            '2024-10-12_15-33-37_bo_parafac_seed2.db',
            '2024-10-12_15-34-14_bo_parafac_seed3.db',
            '2024-10-12_15-34-52_bo_parafac_seed4.db'
        ],
    }



    # Call the function to generate and save plots with custom size
    plot_and_save_individual_histories(
        sampler_files, 
        db_folder, 
        output_dir,
        map_shape=(2, 2),
    )


# if __name__ == '__main__':
#     result_path = "/Users/keisukeonoue/ws/constrained_BO_v2/results_2024-10-12"
#     db_folder = os.path.join(result_path, "dbs")
#     output_dir = "./"

#     # Sample file to debug (you can modify this to any specific file you want to test)
#     db_file = '2024-10-12_15-27-35_bo_benchmark_random_seed0.db'
#     sampler = 'random'

#     # Call the inner function directly to test it with just one file
#     plot_individual_file(
#         db_file, 
#         sampler, 
#         db_folder, 
#         output_dir, 
#         width=1024,  # Custom width for debugging
#         height=768   # Custom height for debugging
#     )
