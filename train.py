import tensorflow as tf
import features
from build_model import TransformerBlock
import numpy as np

rec_path = "./data/rec.csv"
demand_path = "./data/demand.csv"
model_path = "./rec_pricing.keras"
lr_rate = 0.0005
err_threshold = 0.1

def custom_accuracy(y_pred, y_true):
	y_pred = y_pred.reshape(-1)
	err = np.abs(y_true - y_pred)
	return err < err_threshold

class AccSaveCallback(tf.keras.callbacks.Callback):
	def __init__(self, data, save_path, acc_func, init_score=0.4, print_type="all"):
		"""
			data: (x_data, y_data) for validation
			save_path: model path
			acc_func: input (y_pred, y_true) -> output <bool>
			init_score: init high score
			print_type: "all", "high_score", None
		"""
		self.x_data, self.y_data = data
		self.save_path = save_path
		self.acc_func = acc_func
		self.print_type = print_type
		self.high_score = init_score

	def on_epoch_end(self, epoch, logs=None):
		is_high_score = False
		pred = self.model.predict(self.x_data, verbose=0)
		correct = self.acc_func(pred, self.y_data)
		accuracy = np.mean(correct)
		if self.high_score < accuracy:
			is_high_score = True
			self.high_score = accuracy
			self.model.save(self.save_path)
		if self.print_type == "all":
			print(f"\nEpoch {epoch} accuracy: {round(accuracy, 4)}")
		elif self.print_type == "high_score" and is_high_score:
			print(f"\nEpoch {epoch} accuracy: {round(accuracy, 4)}")

def custom_sigmoid_loss(y_true, y_pred):
	y_true = tf.reshape(y_true, [-1]) # [None, 1] -> [None]
	# err = tf.abs(y_true - y_pred)
	err = y_true - y_pred
	err = tf.math.square(err)
	mul = 2 / err_threshold
	loss = tf.sigmoid(mul * (err - err_threshold))
	return tf.reduce_mean(loss)

model = tf.keras.models.load_model(
	model_path,
	custom_objects = {
		"TransformerBlock" : TransformerBlock,
	},
	compile = False
)

adam_opt = tf.keras.optimizers.Adam(learning_rate=lr_rate)

model.compile(
	optimizer = adam_opt,
	loss = "mse",
	# loss = custom_sigmoid_loss
)
# ^^^ 훈련에 두가지 loss 모두 사용됨

model.summary()

x, y = features.get_xy(rec_path=rec_path, demand_path=demand_path)
x, y = x.astype('float32'), y.astype('float32')
x_train, y_train = x[9:690], y[9:690]
x_val, y_val = x[690:], y[690:]

save_cb = AccSaveCallback(
	(x_val, y_val),
	"./checkpoint.keras",
	custom_accuracy,
	init_score = 0.2,
	print_type = "all"
)

model.fit(
	x_train, y_train,
	validation_data = (x_val, y_val),
	epochs = 10,
	batch_size = 32,
	callbacks = [save_cb],
	# verbose = 0
)

model.save(model_path)
