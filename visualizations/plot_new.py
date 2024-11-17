import os
import shutil
import optuna
from optuna.visualization import plot_optimization_history

# ベースディレクトリと共通部分を設定
target_dir_name = "2024-11-17_11-32-46"
result_dir = "/Users/keisukeonoue/ws/constrained_BO_v3/results"
base_dir = os.path.join(result_dir, target_dir_name)
common_prefix = f"{target_dir_name}_"

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
def generate_and_save_plots(dbs_dir, images_dir, common_prefix):
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

# メイン処理
if __name__ == "__main__":
    # ディレクトリの初期化
    logs_dir, dbs_dir, images_dir = initialize_directories(base_dir)

    # ファイルの整理
    organize_files(base_dir, logs_dir, dbs_dir)

    # プロットの生成と保存
    generate_and_save_plots(dbs_dir, images_dir, common_prefix)