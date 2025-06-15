import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from build_model import TransformerBlock
import tensorflow as tf
import features

rec_path = "./data/rec.csv"
demand_path = "./data/demand.csv"
model_path = "./rec_pricing.keras"
# model_path = "./checkpoint.keras"

def highlight_background(ax, pred, high_thresh=0.3, low_thresh=-0.3, alpha=0.15):
	"""예측값 기준으로 ax에 배경색 칠하기"""
	for i in range(len(pred)):
		if pred[i] > high_thresh:
			ax.axvspan(i - 0.5, i + 0.5, color='green', alpha=alpha)
		elif pred[i] < low_thresh:
			ax.axvspan(i - 0.5, i + 0.5, color='red', alpha=alpha)

# --- 모델 로드 및 예측 ---
model = tf.keras.models.load_model(
	model_path,
	custom_objects={"TransformerBlock": TransformerBlock},
	compile=False
)

seq, y = features.get_xy(rec_path=rec_path, demand_path=demand_path)
seq = seq.astype('float32')[590:]  # val data
y = y.astype('float32')[590:]
pred = model.predict(seq).reshape(-1)
pred_ma = features.ema(pred, 5)

# --- 가격 데이터 추출 ---
rec_df = pd.read_csv(rec_path)
prices = rec_df['price1'].to_numpy()[-179:-27]       # 길이: 152

# --- 시각화 ---
x = np.arange(len(pred))  # 길이 52

plt.figure(figsize=(12, 6))

# 1. 가격 차트
ax1 = plt.subplot(2, 1, 1)
ax1.plot(x, prices, label='Price', color='blue')
highlight_background(ax1, pred, high_thresh=0.35)
ax1.set_title("REC Price and Detected Local Mins")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True)

# 2. 예측 결과
ax2 = plt.subplot(2, 1, 2)
ax2.plot(x, pred, label='Prediction', color='gray')
ax2.plot(x, pred_ma, label='Prediction Ma', color='green')
ax2.plot(x, y, label='y', color='red')
highlight_background(ax2, pred, high_thresh=0.35)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_title("Model Prediction")
ax2.set_xlabel("Time")
ax2.set_ylabel("Predicted Value")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("rec_model_output.png", dpi=300)
print("저장 완료: rec_model_output.png")




