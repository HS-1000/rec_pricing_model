import pandas as pd
import features

df = pd.read_csv('./data/demand.csv')
features.demand_features(df)
