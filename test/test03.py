import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. Load data
rec_df = pd.read_csv('data/rec.csv', parse_dates=['date'])
demand_df = pd.read_csv('data/demand.csv', parse_dates=['date'])

# 정렬 및 인덱싱
rec_df = rec_df.sort_values('date').reset_index(drop=True)
demand_df = demand_df.sort_values('date').set_index('date')

# 2. 리스트 생성
rec_returns = []
demand_changes = []
interval_lengths = []

# 3. 두 REC 가격 데이터 포인트 사이의 수요 변화 계산
for i in range(5, len(rec_df)):
    date_prev = rec_df.loc[i-5, 'date']
    date_curr = rec_df.loc[i, 'date']
    price_prev = rec_df.loc[i-1, 'price1']
    price_curr = rec_df.loc[i, 'price1']
    
    # 기간 중 demand 변화량 계산
    period_demand = demand_df.loc[date_prev:date_curr]
    if len(period_demand) < 2:
        continue  # 너무 짧은 구간은 제외
    
    # 수요 변화량 계산: 누적 변화 or 평균 변화
    demand_change = (period_demand['peak_demand_mw'].iloc[-1] - period_demand['peak_demand_mw'].iloc[0]) / period_demand['peak_demand_mw'].iloc[0]
    
    # REC 가격 변화율
    price_return = (price_curr - price_prev) / price_prev
    
    # 저장
    rec_returns.append(price_return)
    demand_changes.append(demand_change)
    interval_lengths.append((date_curr - date_prev).days)

# 4. DataFrame으로 변환
result_df = pd.DataFrame({
    'rec_return': rec_returns,
    'demand_change': demand_changes,
    'interval_days': interval_lengths
})

# 5. 상관관계 분석
corr, pval = pearsonr(result_df['rec_return'], result_df['demand_change'])
print(f"피어슨 상관계수: {corr:.4f}, p-value: {pval:.4e}")

# 6. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(result_df['demand_change'], result_df['rec_return'], alpha=0.7)
plt.xlabel('Demand Change (기간 내 변화율)')
plt.ylabel('REC Price Return')
plt.title('Demand 변화 vs REC 가격 변화')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.show()
