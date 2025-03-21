
def main():
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

    # ビン分割したデータフレームを作成
    binned_df = create_binned_dataframe(df_features, df_target)

    # 重複のない特徴量ペアの組み合わせを生成
    feature_pairs = list(combinations(df_features.columns, 2))

    # 各特徴量ペアについて、全体とOutcome別にクロス集計表を作成
    for feature_1, feature_2 in feature_pairs:
        print(f"Feature Pair: {feature_1}, {feature_2}")
        print()
        
        # 全体の集計表（0埋め）
        pair_table_all = complete_crosstab(binned_df[feature_1], binned_df[feature_2])
        print("全データ:")
        print(pair_table_all)
        print()
        
        # カウント0のエントリを抽出
        zero_entries = []
        for row in pair_table_all.index:
            for col in pair_table_all.columns:
                if pair_table_all.loc[row, col] == 0:
                    zero_entries.append(f"{feature_1}={row},{feature_2}={col}")
        
        # Outcome = True (陽性)のみの集計表
        diabetic_df = binned_df[binned_df['Outcome'] == True]
        pair_table_true = complete_crosstab(diabetic_df[feature_1], diabetic_df[feature_2])
        
        # 陽性率の計算 (陽性数 ÷ 全体数)
        positive_rate_raw = pair_table_true / pair_table_all
        positive_rate_raw = positive_rate_raw.fillna(0)  # ゼロ除算による NaN を 0 に置き換え
        
        # DataFrameに対して要素ごとにフォーマットを適用
        positive_rate = positive_rate_raw.apply(lambda col: col.map(lambda x: f"{x:.2f}"))
        
        print("陽性率:")
        print(positive_rate)
        print()
        
        # カウント0ではないが陽性率0のエントリを抽出
        zero_positive_rate_entries = []
        for row in pair_table_all.index:
            for col in pair_table_all.columns:
                if pair_table_all.loc[row, col] > 0 and positive_rate_raw.loc[row, col] == 0:
                    count = pair_table_all.loc[row, col]
                    zero_positive_rate_entries.append(f"{feature_1}={row},{feature_2}={col} (サンプル数: {count})")
        
        print("カウント0のエントリ:")
        if zero_entries:
            for entry in zero_entries:
                print(entry)
        else:
            print("なし")

        print()

        print("サンプルはあるが陽性率0のエントリ:")
        if zero_positive_rate_entries:
            for entry in zero_positive_rate_entries:
                print(entry)
        else:
            print("なし")

        print("----------------------------------\n")
            

if __name__ == "__main__":
    main()