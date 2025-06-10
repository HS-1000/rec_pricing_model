import pandas as pd
import numpy as np

"""
사용할 피처들:
	A. 시계열
		1. day-of-year
		2. months-to-now (delete)
	B. REC 현물시장
		1. 거래량
		2. 가격
		3. yoy, mom
	C. 기술지표
		1. 변화량(x)
		2. 이동평균
		3. 표준편차
	D. 시장외부
		1. RPS
		2. 전력수요(mom, yoy, ma)
		3. REC발급량
"""

def date_featrues(date:pd.Series, base_time:pd.Timestamp=None):
	if base_time is None:
		base_time = pd.Timestamp.now()
	date = pd.to_datetime(date)
	day_of_year = date.dt.dayofyear
	norm_day = day_of_year / 366
	day_sin = np.sin(2 * np.pi * norm_day) # 1
	day_cos = np.cos(2 * np.pi * norm_day) # 2
	return pd.DataFrame({
		"date" : date,
		"day_sin" : day_sin,
		"day_cos" : day_cos,
	})

def ema(prices, n):
	weight = 2/(n+1)
	ema_ = [prices[0]]
	tmp = prices[0]
	for p in prices[1:]:
		tmp = weight*p + (1-weight)*tmp
		ema_.append(tmp)
	return ema_

def rolling_min_max(data, window):
	min_ = []
	max_ = []
	for i in range(1, window):
		min_.append(np.min(data[:i]))
		max_.append(np.max(data[:i]))
	for i in range(window, len(data)+1):
		min_.append(np.min(data[i-window:i]))
		max_.append(np.max(data[i-window:i]))
	return np.array(min_), np.array(max_)

def market_features(data:pd.DataFrame):
	"""
	col names: date, volume1, price1
	"""
	data = data.sort_values(by='date', ascending=True).reset_index(drop=True)
	data['date'] = pd.to_datetime(data['date'])
	start_date = data.iloc[0]['date']
	prices = data['price1'].to_numpy()
	volume = data['volume1'].to_numpy()
	price_ma7 = ema(prices, 7)
	vol_ma7 = ema(volume, 7)
	std1 = [np.std(list(prices[:i])) for i in range(1, 20)]
	std2 = [np.std(list(prices[i-20:i])) for i in range(20, len(prices)+1)]
	price_std = np.array(std1 + std2)
	price_min_rol, price_max_rol = rolling_min_max(price_ma7, 30)
	diff = price_max_rol - price_min_rol
	diff[0] = 1 # division by zero
	price_norm = (prices - price_min_rol) / diff # 3
	price_ma_norm = (price_ma7 - price_min_rol) / diff # 4
	vol_min_rol, vol_max_rol = rolling_min_max(vol_ma7, 30)
	diff = vol_max_rol - vol_min_rol
	diff[0] = 1
	vol_norm = (volume - vol_min_rol) / diff # 5
	vol_ma_norm = (vol_ma7 - vol_min_rol) / diff # 6
	price_std_rate = price_std / prices * 4 # 7
	length = len(data)
	prices_mom = np.full(length, np.nan, dtype=np.float64)
	prices_qoq = np.full(length, np.nan, dtype=np.float64)
	prices_yoy = np.full(length, np.nan, dtype=np.float64)
	day_delta = pd.Timedelta(days=1)
	month_delta = pd.Timedelta(days=30)
	quater_delta = pd.Timedelta(days=91)
	year_delta = pd.Timedelta(days=365)
	data = data.set_index('date', drop=True)
	for i, (now_dt, row) in enumerate(data.iterrows()):
		mom_dt = now_dt - month_delta
		while True:
			if mom_dt <= start_date:
				mom_price = data.loc[start_date, 'price1']
				break
			if mom_dt in data.index:
				mom_price = data.loc[mom_dt, 'price1']
				break
			else:
				mom_dt -= day_delta
		prices_mom[i] = (row['price1'] - mom_price) / mom_price
		qoq_dt = now_dt - quater_delta
		while True:
			if qoq_dt <= start_date:
				qoq_price = data.loc[start_date, 'price1']
				break
			if qoq_dt in data.index:
				qoq_price = data.loc[qoq_dt, 'price1']
				break
			else:
				qoq_dt -= day_delta
		prices_qoq[i] = (row['price1'] - qoq_price) / qoq_price
		yoy_dt = now_dt - year_delta
		while True:
			if yoy_dt <= start_date:
				yoy_price = data.loc[start_date, 'price1']
				break
			if yoy_dt in data.index:
				yoy_price = data.loc[yoy_dt, 'price1']
				break
			else:
				yoy_dt -= day_delta
		prices_yoy[i] = (row['price1'] - yoy_price) / yoy_price
	prices_mom *= 7
	prices_qoq *= 4
	prices_yoy *= 2
	prices_mom = np.tanh(prices_mom) # 8
	prices_qoq = np.tanh(prices_qoq) # 9
	prices_yoy = np.tanh(prices_yoy) # 10
	return pd.DataFrame({
		'date' : data.index.to_numpy(),
		'prices' : price_norm,
		'price_ma_norm' : price_ma_norm,
		'volume' : vol_norm,
		'vol_ma_norm' : vol_ma_norm,
		'price_std' : price_std_rate,
		'prices_mom' : prices_mom,
		'prices_qoq' : prices_qoq,
		'prices_yoy' : prices_yoy
	})

def demand_features(data:pd.DataFrame) -> pd.DataFrame:
	"""
	col names: date, peak_demand_mw
	features:
		ma5, ma12, MoM, YoY
	"""
	date = pd.to_datetime(data['date']).to_numpy()
	return_dt = date[365:]
	demand = data['peak_demand_mw'].to_numpy()
	demand_ma5 = ema(demand, 5)[365:]
	dem_ma5_min, dem_ma5_max = rolling_min_max(demand_ma5, 30)
	diff = dem_ma5_max - dem_ma5_min
	diff[0] = 1
	dem_ma5_norm = (demand_ma5 - dem_ma5_min) / diff # 11
	demand_ma12 = ema(demand, 12)[365:]
	dem_ma12_min, dem_ma12_max = rolling_min_max(demand_ma12, 30)
	diff = dem_ma12_max - dem_ma12_min
	diff[0] = 1
	dem_ma12_norm = (demand_ma12 - dem_ma12_min) / diff # 12
	demand_now = demand[365:].copy()
	demand_before_30 = demand[335:-30].copy()
	delta_30 = (demand_now - demand_before_30) / (demand_before_30 * 0.25)
	delta_30_norm = np.tanh(delta_30) # 13
	demand_before_365 = demand[:-365].copy()
	delta_365 = (demand_now - demand_before_365) / (demand_before_365 * 0.25)
	delta_365_norm = np.tanh(delta_365) # 14
	return pd.DataFrame({
	'date' : return_dt,
	'demand_ma5' : dem_ma5_norm,
	'demand_ma12' : dem_ma12_norm,
	'demand_mom' : delta_30_norm,
	'demand_yoy' : delta_365_norm
	})

# rec 발급량 2014 - 2024
# 8338589
# 12140173
# 14599281
# 20108089
# 25862989
# 31966789
# 42952386
# 56027252
# 67510830
# 71820737
# 80014548
# RPS 비율 %
# 2017 4
# 2018 5
# 2019 6
# 2020 7
# 2021 9
# 2022 12.5
# 2023 13
# 2024 13.5
# 2025 14
def year_unit_features(date:pd.Series):
	date = pd.to_datetime(date)
	rec_3y_sum_diff = { # 최근 3년 발행량합의 년 단위 증가율
		2016 : 0.422533037,
		2017 : 0.335523279,
		2018 : 0.292924989,
		2019 : 0.28673279,
		2020 : 0.293109086,
		2021 : 0.299301601,
		2022 : 0.271439564,
		2023 : 0.173393416,
		2024 : 0.122785836
	}
	rps_ratio = {
		2017 : 0.16,
		2018 : 0.2,
		2019 : 0.24,
		2020 : 0.28,
		2021 : 0.36,
		2022 : 0.5,
		2023 : 0.52,
		2024 : 0.54,
		2025 : 0.56
	}
	rec = np.array([None for _ in range(len(date))])
	rps = np.array([None for _ in range(len(date))])
	for i, d in enumerate(date):
		y = d.year
		m = d.month
		# rec
		if m >= 2: # 15
			rec[i] = rec_3y_sum_diff[y-1]
		else:
			rec[i] = rec_3y_sum_diff[y-2]
		# rps
		if m >= 12: # 16
			rps[i] = rps_ratio[y+1]
		else:
			rps[i] = rps_ratio[y]
	return pd.DataFrame({
		'date' : date,
		'rec_count' : rec,
		'rps' : rps
	})

def y_signal(data:pd.DataFrame, mul=4) -> np.ndarray:
	"""
	약 한달뒤(9개의 데이터 포인트)와 가격비교
	"""
	prices = data['price1'].to_numpy()
	now_price = prices[:-27]
	future_price = prices[27:]
	diff = future_price - now_price
	pct_change = diff / now_price
	return_dt = data['date'].iloc[:-27].to_numpy()
	return return_dt, np.tanh(mul * pct_change)

def get_xy(rec_path, demand_path):
	rec_df = pd.read_csv(rec_path)
	rec_df.loc[rec_df.index[-1], 'price1'] = 80000
	demand_df = pd.read_csv(demand_path)

	dt_feat = date_featrues(rec_df["date"])
	rec_feat = market_features(rec_df)
	dem_feat = demand_features(demand_df)
	year_feat = year_unit_features(rec_feat['date'])
	df = pd.merge(rec_feat, dt_feat, on='date', how='inner')
	df = pd.merge(df, dem_feat, on='date', how='inner')
	df = pd.merge(df, year_feat, on='date', how='inner')
	seq_dt = df["date"].iloc[31:-27] # sequences length: 32, predict 27times after rec
	df = df.drop(columns='date').to_numpy()
	seq = np.lib.stride_tricks.sliding_window_view(df, (32, 16))
	seq = np.squeeze(seq, axis=1)
	seq = seq[:-27]
	y_dt, y_ = y_signal(rec_df)
	y = []
	y_dt = pd.to_datetime(y_dt)
	seq_dt = pd.to_datetime(seq_dt)
	y_index = 0
	for t in seq_dt:
		while True:
			if t == y_dt[y_index]:
				y.append(y_[y_index])
				break
			elif t > y_dt[y_index]:
				y_index += 1
			elif y_index == len(y_dt):
				raise RuntimeError("손상된 데이터")
	y = np.array(y)
	return seq, y








