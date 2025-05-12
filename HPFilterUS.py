# prompt: FREDのGDPデータに自然対数変換をするコードを書いてください

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr

# FREDからGDPデータを読み込む（例としてCSVファイルを使用）
# 実際のデータファイルパスに置き換えてください
try:
    df = pd.read_csv('GDPC1.csv', index_col='observation_date') # 'DATE'カラムをインデックスに設定
except FileNotFoundError:
    print("Error: 'gdp_data.csv' not found. Please provide the correct file path.")
    exit()

# GDPデータに自然対数変換を適用
# 'GDP'カラムが存在することを想定しています
# GDPのデータはGDPC1と書いてあったので、それを名前に
# カラム名が異なる場合は、適宜変更してください
if 'GDPC1' not in df.columns:
    print("Error: 'GDP' column not found in the DataFrame.")
    exit()

df['GDPC1_ln'] = np.log(df['GDPC1'])

#HPフィルターの適応
cycle, trend = sm.tsa.filters.hpfilter('gdp_ln_data.csv', lamb=1600)

# 結果の表示（最初の5行）
print(df.head())

# 必要に応じてCSVファイルに保存
df.to_csv('gdp_ln_data.csv')