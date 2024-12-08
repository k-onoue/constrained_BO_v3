import os
import shutil
import optuna
import pandas as pd
import plotly.graph_objects as go
from optuna.visualization import plot_optimization_history
import argparse

# ディレクトリ構造の初期化
def initialize_directories(base_dir):
    logs_dir = os.path.join(base_dir, "logs")
    dbs_dir = os.path.join(base_dir, "dbs")
    images_dir = os.path.join(base_dir, "images")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(dbs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    return logs_dir, dbs_dir, images_dir

# ファイルを整理する関数
def organize_files(base_dir, logs_dir, dbs_dir):
    for filename in os.listdir(base_dir):
        if filename.endswith(".log"):
            shutil.move(os.path.join(base_dir, filename), os.path.join(logs_dir, filename))
        elif filename.endswith(".db"):
            shutil.move(os.path.join(base_dir, filename), os.path.join(dbs_dir, filename))

# Study name を構築する関数
def construct_study_name(common_prefix, db_file):
    return f"{common_prefix}{os.path.splitext(db_file)[0]}"

# プロットを生成して保存する関数
def generate_and_save_plots(dbs_dir, images_dir, common_prefix, max_trials=None):
    task_sampler_best_values = {}

    for db_file in os.listdir(dbs_dir):
        if db_file.endswith(".db"):
            # Study name を構築
            study_name = construct_study_name(common_prefix, db_file)
            db_path = os.path.join(dbs_dir, db_file)

            # Optuna Study の読み込み
            storage = f"sqlite:///{db_path}"
            study = optuna.load_study(study_name=study_name, storage=storage)

            # プロットの生成
            fig = plot_optimization_history(study)

            # ディレクトリとファイル名を生成
            base_name, seed = os.path.splitext(db_file)[0].rsplit("_seed", 1)
            image_subdir = os.path.join(images_dir, base_name)
            os.makedirs(image_subdir, exist_ok=True)

            # プロットを保存
            output_path = os.path.join(image_subdir, f"seed{seed}.png")
            fig.write_image(output_path)

            # Collect best values for mean and error plot
            df = study.trials_dataframe()
            if max_trials is not None:
                df = df.head(max_trials)  # トライアル数を制限

            df['best_value'] = df['value'].cummin() if study.direction == optuna.study.StudyDirection.MINIMIZE else df['value'].cummax()

            task_sampler = base_name
            if task_sampler not in task_sampler_best_values:
                task_sampler_best_values[task_sampler] = []
            task_sampler_best_values[task_sampler].append(df['best_value'])

    # Plot mean with error for each task and sampler pair
    for task_sampler, best_values_list in task_sampler_best_values.items():
        if best_values_list:
            merged_df = pd.concat(best_values_list, axis=1)
            mean_best = merged_df.mean(axis=1)
            std_best = merged_df.std(axis=1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged_df.index, y=mean_best, mode='lines',
                name='Mean Best Value', line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=merged_df.index, y=mean_best + std_best, mode='lines',
                line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=merged_df.index, y=mean_best - std_best, mode='lines',
                line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)', showlegend=False
            ))

            fig.update_layout(
                title=f"Mean Best Value with Error for {task_sampler}",
                xaxis_title="Iteration",
                yaxis_title="Best Value",
                template="plotly_white"
            )

            if max_trials is not None:
                fig_name = f"mean_with_error_{task_sampler}_max_trials_{max_trials}.png"
            else:
                fig_name = f"mean_with_error_{task_sampler}.png"
            
            fig.write_image(os.path.join(images_dir, fig_name))

# メイン処理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process target directory name for organizing and plotting.")
    parser.add_argument("target_dir_name", type=str, help="Name of the target directory to process.")
    parser.add_argument("--max_trials", type=int, default=None, help="Maximum number of trials to include in the plots.")

    args = parser.parse_args()

    # ベースディレクトリと共通部分を設定
    result_dir = "/Users/keisukeonoue/ws/constrained_BO_v3/results_constrained"
    base_dir = os.path.join(result_dir, args.target_dir_name)
    common_prefix = f"{args.target_dir_name}_"

    # ディレクトリの初期化
    logs_dir, dbs_dir, images_dir = initialize_directories(base_dir)

    # ファイルの整理
    organize_files(base_dir, logs_dir, dbs_dir)

    # プロットの生成と保存
    generate_and_save_plots(dbs_dir, images_dir, common_prefix, args.max_trials)

    print("Plots have been generated and saved successfully.")





# import os
# import shutil
# import optuna
# import pandas as pd
# import plotly.graph_objects as go
# from optuna.visualization import plot_optimization_history
# import argparse

# # ディレクトリ構造の初期化
# def initialize_directories(base_dir):
#     logs_dir = os.path.join(base_dir, "logs")
#     dbs_dir = os.path.join(base_dir, "dbs")
#     images_dir = os.path.join(base_dir, "images")

#     os.makedirs(logs_dir, exist_ok=True)
#     os.makedirs(dbs_dir, exist_ok=True)
#     os.makedirs(images_dir, exist_ok=True)

#     return logs_dir, dbs_dir, images_dir

# # ファイルを整理する関数
# def organize_files(base_dir, logs_dir, dbs_dir):
#     for filename in os.listdir(base_dir):
#         if filename.endswith(".log"):
#             shutil.move(os.path.join(base_dir, filename), os.path.join(logs_dir, filename))
#         elif filename.endswith(".db"):
#             shutil.move(os.path.join(base_dir, filename), os.path.join(dbs_dir, filename))

# # Study name を構築する関数
# def construct_study_name(common_prefix, db_file):
#     return f"{common_prefix}{os.path.splitext(db_file)[0]}"

# # プロットを生成して保存する関数
# def generate_and_save_plots(dbs_dir, images_dir, common_prefix):
#     task_sampler_best_values = {}

#     for db_file in os.listdir(dbs_dir):
#         if db_file.endswith(".db"):
#             # Study name を構築
#             study_name = construct_study_name(common_prefix, db_file)
#             db_path = os.path.join(dbs_dir, db_file)

#             # Optuna Study の読み込み
#             storage = f"sqlite:///{db_path}"
#             study = optuna.load_study(study_name=study_name, storage=storage)

#             # プロットの生成
#             fig = plot_optimization_history(study)

#             # ディレクトリとファイル名を生成
#             base_name, seed = os.path.splitext(db_file)[0].rsplit("_seed", 1)
#             image_subdir = os.path.join(images_dir, base_name)
#             os.makedirs(image_subdir, exist_ok=True)

#             # プロットを保存
#             output_path = os.path.join(image_subdir, f"seed{seed}.png")
#             fig.write_image(output_path)

#             # Collect best values for mean and error plot
#             df = study.trials_dataframe()
#             df['best_value'] = df['value'].cummin() if study.direction == optuna.study.StudyDirection.MINIMIZE else df['value'].cummax()

#             task_sampler = base_name
#             if task_sampler not in task_sampler_best_values:
#                 task_sampler_best_values[task_sampler] = []
#             task_sampler_best_values[task_sampler].append(df['best_value'])

#     # Plot mean with error for each task and sampler pair
#     for task_sampler, best_values_list in task_sampler_best_values.items():
#         if best_values_list:
#             merged_df = pd.concat(best_values_list, axis=1)
#             mean_best = merged_df.mean(axis=1)
#             std_best = merged_df.std(axis=1)

#             fig = go.Figure()
#             fig.add_trace(go.Scatter(
#                 x=merged_df.index, y=mean_best, mode='lines',
#                 name='Mean Best Value', line=dict(color='blue')
#             ))
#             fig.add_trace(go.Scatter(
#                 x=merged_df.index, y=mean_best + std_best, mode='lines',
#                 line=dict(width=0), showlegend=False
#             ))
#             fig.add_trace(go.Scatter(
#                 x=merged_df.index, y=mean_best - std_best, mode='lines',
#                 line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)', showlegend=False
#             ))

#             fig.update_layout(
#                 title=f"Mean Best Value with Error for {task_sampler}",
#                 xaxis_title="Iteration",
#                 yaxis_title="Best Value",
#                 template="plotly_white"
#             )

#             fig.write_image(os.path.join(images_dir, f"mean_with_error_{task_sampler}.png"))

# # メイン処理
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process target directory name for organizing and plotting.")
#     parser.add_argument("target_dir_name", type=str, help="Name of the target directory to process.")

#     args = parser.parse_args()

#     # ベースディレクトリと共通部分を設定
#     result_dir = "/Users/keisukeonoue/ws/constrained_BO_v3/results"
#     base_dir = os.path.join(result_dir, args.target_dir_name)
#     common_prefix = f"{args.target_dir_name}_"

#     # ディレクトリの初期化
#     logs_dir, dbs_dir, images_dir = initialize_directories(base_dir)

#     # ファイルの整理
#     organize_files(base_dir, logs_dir, dbs_dir)

#     # プロットの生成と保存
#     generate_and_save_plots(dbs_dir, images_dir, common_prefix)




# import os
# import shutil
# import optuna
# import pandas as pd
# import plotly.graph_objects as go
# from optuna.visualization import plot_optimization_history

# # ベースディレクトリと共通部分を設定
# target_dir_name = "2024-11-17_11-32-46"
# result_dir = "/Users/keisukeonoue/ws/constrained_BO_v3/results"
# base_dir = os.path.join(result_dir, target_dir_name)
# common_prefix = f"{target_dir_name}_"

# # ディレクトリ構造の初期化
# def initialize_directories(base_dir):
#     logs_dir = os.path.join(base_dir, "logs")
#     dbs_dir = os.path.join(base_dir, "dbs")
#     images_dir = os.path.join(base_dir, "images")

#     os.makedirs(logs_dir, exist_ok=True)
#     os.makedirs(dbs_dir, exist_ok=True)
#     os.makedirs(images_dir, exist_ok=True)

#     return logs_dir, dbs_dir, images_dir

# # ファイルを整理する関数
# def organize_files(base_dir, logs_dir, dbs_dir):
#     for filename in os.listdir(base_dir):
#         if filename.endswith(".log"):
#             shutil.move(os.path.join(base_dir, filename), os.path.join(logs_dir, filename))
#         elif filename.endswith(".db"):
#             shutil.move(os.path.join(base_dir, filename), os.path.join(dbs_dir, filename))

# # Study name を構築する関数
# def construct_study_name(common_prefix, db_file):
#     return f"{common_prefix}{os.path.splitext(db_file)[0]}"

# # プロットを生成して保存する関数
# def generate_and_save_plots(dbs_dir, images_dir, common_prefix):
#     task_sampler_best_values = {}

#     for db_file in os.listdir(dbs_dir):
#         if db_file.endswith(".db"):
#             # Study name を構築
#             study_name = construct_study_name(common_prefix, db_file)
#             db_path = os.path.join(dbs_dir, db_file)

#             # Optuna Study の読み込み
#             storage = f"sqlite:///{db_path}"
#             study = optuna.load_study(study_name=study_name, storage=storage)

#             # プロットの生成
#             fig = plot_optimization_history(study)

#             # ディレクトリとファイル名を生成
#             base_name, seed = os.path.splitext(db_file)[0].rsplit("_seed", 1)
#             image_subdir = os.path.join(images_dir, base_name)
#             os.makedirs(image_subdir, exist_ok=True)

#             # プロットを保存
#             output_path = os.path.join(image_subdir, f"seed{seed}.png")
#             fig.write_image(output_path)

#             # Collect best values for mean and error plot
#             df = study.trials_dataframe()
#             df['best_value'] = df['value'].cummin() if study.direction == optuna.study.StudyDirection.MINIMIZE else df['value'].cummax()

#             task_sampler = base_name
#             if task_sampler not in task_sampler_best_values:
#                 task_sampler_best_values[task_sampler] = []
#             task_sampler_best_values[task_sampler].append(df['best_value'])

#     # Plot mean with error for each task and sampler pair
#     for task_sampler, best_values_list in task_sampler_best_values.items():
#         if best_values_list:
#             merged_df = pd.concat(best_values_list, axis=1)
#             mean_best = merged_df.mean(axis=1)
#             std_best = merged_df.std(axis=1)

#             fig = go.Figure()
#             fig.add_trace(go.Scatter(
#                 x=merged_df.index, y=mean_best, mode='lines',
#                 name='Mean Best Value', line=dict(color='blue')
#             ))
#             fig.add_trace(go.Scatter(
#                 x=merged_df.index, y=mean_best + std_best, mode='lines',
#                 line=dict(width=0), showlegend=False
#             ))
#             fig.add_trace(go.Scatter(
#                 x=merged_df.index, y=mean_best - std_best, mode='lines',
#                 line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)', showlegend=False
#             ))

#             fig.update_layout(
#                 title=f"Mean Best Value with Error for {task_sampler}",
#                 xaxis_title="Iteration",
#                 yaxis_title="Best Value",
#                 template="plotly_white"
#             )

#             fig.write_image(os.path.join(images_dir, f"mean_with_error_{task_sampler}.png"))

# # メイン処理
# if __name__ == "__main__":
#     # ディレクトリの初期化
#     logs_dir, dbs_dir, images_dir = initialize_directories(base_dir)

#     # ファイルの整理
#     organize_files(base_dir, logs_dir, dbs_dir)

#     # プロットの生成と保存
#     generate_and_save_plots(dbs_dir, images_dir, common_prefix)
