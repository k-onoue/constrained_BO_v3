import os
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from optuna.visualization import plot_optimization_history
from _src import search_log_files

# 設定部分
TEST_FUNC = "ackley"
BENCHMARK_PATH_BASE = "/Users/keisukeonoue/ws/constrained_BO_v2/results_benchmark"
PARAFAC_PATH_BASE = "/Users/keisukeonoue/ws/constrained_BO_v2/results"
IMAGE_SAVE_BASE = os.path.join(PARAFAC_PATH_BASE, "images")
PROBLEM_SETTING = {
    "dim": ["dim2", "dim3", "dim5", "dim7"],
    "model": ["random", "tpe", "gp", "parafac"]
}

BENCHMARK_DB_DIR = os.path.join(BENCHMARK_PATH_BASE, TEST_FUNC, "dbs")
PARAFAC_DB_DIR = os.path.join(PARAFAC_PATH_BASE, "dbs")

# ヘルパー関数
def get_color_scale():
    return (px.colors.qualitative.Plotly 
            + px.colors.qualitative.D3 
            + px.colors.qualitative.G10 
            + px.colors.qualitative.T10)

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

def load_study_data(db_file, db_folder, sampler):
    path = os.path.join(db_folder, db_file)
    storage_url = f"sqlite:///{path}"
    study_name = db_file.split('_bo_')[1].replace('.db', '')
    study_name = "bo_" + study_name if "parafac" in sampler else "ackley_bo_" + study_name

    try:
        print(f"Loading study with name: {study_name}, storage: {storage_url}")
        return optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError as e:
        print(f"Error: Study '{study_name}' not found in '{storage_url}'.\n{e}")
        return None  # ここで None を返して後続で処理

def prepare_dataframes(study):
    if study is None:
        return None
    df = study.trials_dataframe().copy()
    df['best_value'] = (df['value'].cummax() if study.direction == optuna.study.StudyDirection.MAXIMIZE 
                        else df['value'].cummin())
    return df[['number', 'best_value']]

def plot_and_save_fig(sampler_files, db_folder, color_scale, save_path, width=800, height=600, title=""):
    fig = go.Figure()
    for idx, (sampler, db_files) in enumerate(sampler_files.items()):
        dataframes = [
            prepare_dataframes(load_study_data(db_file, db_folder, sampler)) 
            for db_file in db_files
        ]
        dataframes = [df for df in dataframes if df is not None]  # Noneを除外

        if not dataframes:
            print(f"No data to plot for {sampler}. Skipping.")
            continue
        
        merged_df = pd.concat(dataframes, axis=1, keys=[f'seed_{i}' for i in range(len(dataframes))])
        best_values = merged_df.xs('best_value', axis=1, level=1)
        mean_best = best_values.mean(axis=1)
        std_best = best_values.std(axis=1)
        
        fig.add_trace(go.Scatter(
            x=best_values.index, y=mean_best, mode='lines',
            name=f'Mean Best Value ({sampler})', line=dict(color=color_scale[idx])
        ))
        fig.add_trace(go.Scatter(
            x=best_values.index, y=mean_best + std_best, mode='lines',
            line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=best_values.index, y=mean_best - std_best, mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor=hex_to_rgba(color_scale[idx]), showlegend=False
        ))

    fig.update_layout(
        title=title, xaxis_title="Iteration", yaxis_title="Best Value",
        template="plotly_white", legend=dict(orientation="h", x=0, y=1),
        width=width, height=height
    )
    fig.write_image(save_path)

def get_log_files(problem_setting, parafac_db_dir, benchmark_db_dir):
    log_files = {dim: {} for dim in problem_setting["dim"]}
    for dim in problem_setting["dim"]:
        for model in problem_setting["model"]:
            if model == "parafac":
                files = search_log_files(parafac_db_dir, [dim, model])
            else:
                files = search_log_files(benchmark_db_dir, [dim, model])
            if len(files) == 5:
                log_files[dim][model] = files
    return log_files

def main():
    log_files = get_log_files(PROBLEM_SETTING, PARAFAC_DB_DIR, BENCHMARK_DB_DIR)
    color_scale = get_color_scale()

    for dim, model_files in log_files.items():
        for model, db_files in model_files.items():
            if db_files:
                sampler_files = {model: db_files}
                save_path = os.path.join(IMAGE_SAVE_BASE, f"best_values_plot_{dim}_{model}.png")
                title = f"Ackley of dimension {dim} with model {model}"
                db_folder = BENCHMARK_DB_DIR if model != "parafac" else PARAFAC_DB_DIR
                
                plot_and_save_fig(
                    sampler_files=sampler_files,
                    db_folder=db_folder,
                    color_scale=color_scale,
                    save_path=save_path,
                    width=600,
                    height=400,
                    title=title
                )

                # 各 study に対して history プロットを保存
                for db_file in db_files:
                    study = load_study_data(db_file, db_folder, model)
                    if study:
                        plot_optimization_history(study).write_image(
                            os.path.join(IMAGE_SAVE_BASE, f"history_plot_{db_file}.png")
                        )

                    

if __name__ == "__main__":
    main()
    print("プロットの保存が完了しました。")
