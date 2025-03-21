import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from itertools import combinations

# OpenML からデータ取得
diabetes = fetch_openml(name="diabetes", version=1, as_frame=True)
df = diabetes.frame

# フォーマット
df['Outcome'] = (df['class'] == "tested_positive")
df = df.drop('class', axis=1)
df_features = diabetes.data
df_target = df[['Outcome']]

# 特徴量をビン分割して、Outcomeと組み合わせたデータフレームを作成
def create_binned_dataframe(df_features, df_target):
    """特徴量をビン分割し、Outcomeと一緒にデータフレームとして返す"""
    binned_df = pd.DataFrame(index=df_features.index)
    
    for feature in df_features.columns:
        binned_df[feature] = pd.cut(df_features[feature], bins=5, labels=False)
    
    binned_df['Outcome'] = df_target['Outcome'].values
    
    return binned_df

# 全てのビンを含むクロス集計表を作成する関数
def complete_crosstab(series1, series2):
    """欠けているビンを0埋めしたクロス集計表を作成する"""
    # クロス集計表を作成
    crosstab = pd.crosstab(series1, series2)
    
    # 0-4の全てのインデックスとカラムを確保
    all_bins = list(range(5))  # 5つのビンなので0-4
    
    # 欠けているインデックスを追加
    missing_indices = set(all_bins) - set(crosstab.index)
    for idx in missing_indices:
        crosstab.loc[idx] = 0
    
    # 欠けているカラムを追加
    missing_columns = set(all_bins) - set(crosstab.columns)
    for col in missing_columns:
        crosstab[col] = 0
    
    # インデックスとカラムを整列
    crosstab = crosstab.reindex(index=all_bins, columns=all_bins, fill_value=0)
    
    return crosstab

def create_constraint_tensor(features, constraint_type="count_zero_only", zero_entries_dict=None, zero_positive_rate_dict=None):
    """
    特徴量の制約を表す多次元テンソルを作成します
    
    Parameters:
    -----------
    features : list
        特徴量名のリスト
    constraint_type : str
        制約の種類。"count_zero_only", "positive_rate_zero_only", "both" のいずれか
    zero_entries_dict : dict
        カウントが0のエントリを含む辞書
    zero_positive_rate_dict : dict
        陽性率が0のエントリを含む辞書
    
    Returns:
    --------
    numpy.ndarray
        制約テンソル (値が1なら有効、0なら無効)
    """
    # 各特徴量は5つのビン (0-4) があるので、8次元の5x5x5x5x5x5x5x5テンソルを作成
    n_features = len(features)
    tensor_shape = [5] * n_features  # [5, 5, 5, 5, 5, 5, 5, 5]
    
    # 最初は全て有効 (1) で初期化
    constraint_tensor = np.ones(tensor_shape, dtype=int)
    
    # 特徴量名と対応するインデックスのマッピング
    feature_to_idx = {f: i for i, f in enumerate(features)}
    
    # カウント0のエントリを制約に適用
    if constraint_type in ["count_zero_only", "both"] and zero_entries_dict:
        for (feature1, feature2), invalid_indices in zero_entries_dict.items():
            # 特徴量のインデックスを取得
            idx1 = feature_to_idx[feature1]
            idx2 = feature_to_idx[feature2]
            
            # 各無効なインデックスのペアについて
            for val1, val2, _ in invalid_indices:
                # インデックス選択用のスライスを作成
                idx_selector = [slice(None)] * n_features
                idx_selector[idx1] = val1
                idx_selector[idx2] = val2
                
                # 該当するすべてのエントリを0（無効）に設定
                constraint_tensor[tuple(idx_selector)] = 0
    
    # 陽性率0のエントリを制約に適用
    if constraint_type in ["positive_rate_zero_only", "both"] and zero_positive_rate_dict:
        for (feature1, feature2), invalid_indices in zero_positive_rate_dict.items():
            # 特徴量のインデックスを取得
            idx1 = feature_to_idx[feature1]
            idx2 = feature_to_idx[feature2]
            
            # 各無効なインデックスのペアについて
            for val1, val2, _ in invalid_indices:
                # インデックス選択用のスライスを作成
                idx_selector = [slice(None)] * n_features
                idx_selector[idx1] = val1
                idx_selector[idx2] = val2
                
                # 該当するすべてのエントリを0（無効）に設定
                constraint_tensor[tuple(idx_selector)] = 0
    
    return constraint_tensor

# メイン処理：特徴量ペアの制約抽出と制約テンソルの作成

# ビン分割したデータフレームを作成
binned_df = create_binned_dataframe(df_features, df_target)

# 重複のない特徴量ペアの組み合わせを生成
features = list(df_features.columns)
feature_pairs = list(combinations(features, 2))

# 制約辞書を初期化
zero_entries_dict = {}
zero_positive_rate_dict = {}

# 各特徴量ペアについて、制約エントリを抽出
for feature_1, feature_2 in feature_pairs:
    # 全体の集計表（0埋め）
    pair_table_all = complete_crosstab(binned_df[feature_1], binned_df[feature_2])
    
    # カウント0のエントリを抽出
    temp_zero = []
    for row in pair_table_all.index:
        for col in pair_table_all.columns:
            if pair_table_all.loc[row, col] == 0:
                temp_zero.append((row, col, None))
    
    # 辞書に登録
    zero_entries_dict[(feature_1, feature_2)] = temp_zero
    
    # Outcome = True (陽性)のみの集計表
    diabetic_df = binned_df[binned_df['Outcome'] == True]
    pair_table_true = complete_crosstab(diabetic_df[feature_1], diabetic_df[feature_2])
    
    # 陽性率の計算 (陽性数 ÷ 全体数)
    positive_rate_raw = pair_table_true / pair_table_all
    positive_rate_raw = positive_rate_raw.fillna(0)  # ゼロ除算による NaN を 0 に置き換え
    
    # カウント0ではないが陽性率0のエントリを抽出
    temp_zero_rate = []
    for row in pair_table_all.index:
        for col in pair_table_all.columns:
            if pair_table_all.loc[row, col] > 0 and positive_rate_raw.loc[row, col] == 0:
                count = pair_table_all.loc[row, col]
                temp_zero_rate.append((row, col, int(count)))
    
    # 辞書に登録
    zero_positive_rate_dict[(feature_1, feature_2)] = temp_zero_rate

# 3種類の制約テンソルを作成
constraint_tensor_count_zero = create_constraint_tensor(
    features, constraint_type="count_zero_only", 
    zero_entries_dict=zero_entries_dict
)

constraint_tensor_pos_zero = create_constraint_tensor(
    features, constraint_type="positive_rate_zero_only", 
    zero_positive_rate_dict=zero_positive_rate_dict
)

constraint_tensor_both = create_constraint_tensor(
    features, constraint_type="both", 
    zero_entries_dict=zero_entries_dict, 
    zero_positive_rate_dict=zero_positive_rate_dict
)

# 各テンソルの情報表示
print("\n=== カウント0のエントリのみの制約テンソル ===")
print(f"制約テンソルの形状: {constraint_tensor_count_zero.shape}")
print(f"有効なデータポイント数: {np.sum(constraint_tensor_count_zero)}")
print(f"無効なデータポイント数: {np.prod(constraint_tensor_count_zero.shape) - np.sum(constraint_tensor_count_zero)}")
print(f"有効率: {np.sum(constraint_tensor_count_zero) / np.prod(constraint_tensor_count_zero.shape):.4f}")

print("\n=== 陽性率0のエントリのみの制約テンソル ===")
print(f"制約テンソルの形状: {constraint_tensor_pos_zero.shape}")
print(f"有効なデータポイント数: {np.sum(constraint_tensor_pos_zero)}")
print(f"無効なデータポイント数: {np.prod(constraint_tensor_pos_zero.shape) - np.sum(constraint_tensor_pos_zero)}")
print(f"有効率: {np.sum(constraint_tensor_pos_zero) / np.prod(constraint_tensor_pos_zero.shape):.4f}")

print("\n=== 両方の制約を含むテンソル ===")
print(f"制約テンソルの形状: {constraint_tensor_both.shape}")
print(f"有効なデータポイント数: {np.sum(constraint_tensor_both)}")
print(f"無効なデータポイント数: {np.prod(constraint_tensor_both.shape) - np.sum(constraint_tensor_both)}")
print(f"有効率: {np.sum(constraint_tensor_both) / np.prod(constraint_tensor_both.shape):.4f}")

# 各バージョンを保存
np.savez('diabetes_constraint_count_zero.npz', tensor=constraint_tensor_count_zero, features=features)
np.savez('diabetes_constraint_pos_zero.npz', tensor=constraint_tensor_pos_zero, features=features)
np.savez('diabetes_constraint_both.npz', tensor=constraint_tensor_both, features=features)

# 読み込み確認
print("\n=== 保存データの確認 ===")
for filename in ['diabetes_constraint_count_zero.npz', 'diabetes_constraint_pos_zero.npz', 'diabetes_constraint_both.npz']:
    loaded_data = np.load(filename)
    print(f"{filename}: 有効要素数 {np.sum(loaded_data['tensor'])}")