import os

import numpy as np
import optuna
import plotly.express as px  # For color scale
import plotly.io as pio
import plotly.graph_objects as go

from _src import WarcraftObjective


# Inner function for plotting an individual file using scatter with re-indexed y-axis
def plot_scatter_histogram(db_file, sampler, db_folder, output_dir, width, height):
    path = os.path.join(db_folder, db_file)
    storage_url = f"sqlite:///{path}"

    # Extract seed number from the filename
    seed = db_file.split('_')[-1].replace('.db', '')

    # Load the study
    study_name = db_file.split('_bo_')[1].replace('.db', '')
    study = optuna.load_study(study_name="bo_" + study_name, storage=storage_url)

    study_df = study.trials_dataframe().copy()

    # WarcraftObjective クラスのインスタンスを作成
    weight_matrix = np.random.rand(2, 2)
    objective = WarcraftObjective(weight_matrix)

    # 各行に対して penalty_type2 を計算し、新しい列に判定結果を追加
    def check_constraint(row, objective):
        params_columns = [col for col in row.index if col.startswith('params_')]
        direction_matrix = np.array([row[col] for col in params_columns]).reshape(2, 2)

        return 1 - objective.get_penalty_type2(direction_matrix)  # 0~1の連続値

    # study_df に判定結果の列を追加
    study_df['constraint_1'] = study_df.apply(lambda row: check_constraint(row, objective), axis=1)

    # 各 value ごとにグループ化し、グループ内で y の値を 0 から振り直す
    study_df['y_offset'] = study_df.groupby('value').cumcount()

    # Create a figure
    fig = go.Figure()

    # 点に色のグラデーションを付ける
    fig.add_trace(go.Scatter(
        x=study_df['value'],  # x軸にvalue
        y=study_df['y_offset'],  # グループ内で振り直した y軸の値
        mode='markers',
        marker=dict(
            color=study_df['constraint_1'],  # 0 ~ 1 の値に基づく
            colorscale=px.colors.sequential.Plotly3,  # Plotly3 カラースケール
            size=5,
            opacity=1,
            colorbar=dict(title="Constraint Satisfaction")  # カラーバーの追加
        ),
        name='Value with Penalty Degree'
    ))

    # 背景を白に設定
    fig.update_layout(
        template="plotly_white",  # 背景を白にする
        title="Value Distribution with Constraint Satisfaction (Gradation Color)",
        xaxis_title="Value",
        yaxis_title="Count within each value group",
        width=width,
        height=height,
        showlegend=False  # カラーバーがあるので、レジェンドはオフ
    )

    # Save the figure as an image
    output_path = os.path.join(output_dir, f"{sampler}_{seed}_scatter_gradation.png")
    pio.write_image(fig, output_path)

    print(f'Image saved to {output_path}')


if __name__ == '__main__':
    result_path = "/Users/keisukeonoue/ws/constrained_BO_v2/results_2024-10-12"
    db_folder = os.path.join(result_path, "dbs")
    output_dir = os.path.join(result_path, "images")

    db_file = '2024-10-12_15-30-00_bo_benchmark_bruteforce_seed0.db'
    sampler = 'bruteforce'

    plot_scatter_histogram(
        db_file, 
        sampler, 
        db_folder, 
        output_dir, 
        width=1024,  
        height=768
    )
