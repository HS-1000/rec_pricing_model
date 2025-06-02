import pandas as pd
import features

df = pd.read_csv('./data/rec.csv')
features.market_features(df)

