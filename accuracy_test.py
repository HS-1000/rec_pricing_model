import tensorflow as tf
import features
from build_model import TransformerBlock
import numpy as np
# from tensorflow.keras.callbacks import ModelCheckpoint
#
rec_path = "./data/rec.csv"
demand_path = "./data/demand.csv"
model_path = "./rec_pricing.keras"
# model_path = "./checkpoint.keras"

def custom_accuracy(y_pred, y_true):
	y_pred = y_pred.reshape(-1)
	err = np.abs(y_true - y_pred)
	return err < 0.2

model = tf.keras.models.load_model(
	model_path,
	custom_objects = {
		"TransformerBlock" : TransformerBlock,
	},
	compile = False
)

x, y = features.get_xy(rec_path=rec_path, demand_path=demand_path)
x, y = x.astype('float32'), y.astype('float32')
x_val, y_val = x[690:], y[690:]

pred = model.predict(x_val)
corr = custom_accuracy(pred, y_val)
print(np.mean(corr))
