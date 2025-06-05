import tensorflow as tf
import numpy as np
import features
from build_model import TransformerBlock
from tensorflow.keras.callbacks import ModelCheckpoint

rec_path = "./data/rec.csv"
demand_path = "./data/demand.csv"
model_path = "./rec_pricing.keras"

# @tf.keras.saving.register_keras_serializable()
def custom_accuracy(y_true, y_pred):
    err = tf.abs(y_true - y_pred)
    correct = tf.less(err, 0.08)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def lr_scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.05)

lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

model = tf.keras.models.load_model(
	model_path,
	custom_objects = {
		"TransformerBlock" : TransformerBlock,
        "custom_accuracy" : custom_accuracy
	},
    compile = False
)

adam_opt = tf.keras.optimizers.Adam(learning_rate=0.008)

model.compile(
    optimizer = adam_opt,
    loss = 'mae',
    metrics = [custom_accuracy]
)

# model.summary()
# exit()

x, y = features.get_xy(rec_path=rec_path, demand_path=demand_path)
x, y = x.astype('float32'), y.astype('float32')
x_train, y_train = x[9:690], y[9:690]
x_val, y_val = x[690:], y[690:]

# TEST
# pred = model.predict(x[:10])
# print(pred)
# print(y[:10])
# exit()

checkpoint_cb = ModelCheckpoint(
	filepath = "./last_checkpoint.keras",
	monitor = "val_custom_accuracy",
    save_best_only = True,
    save_weights_only = False,
    mode = "max",
    verbose = 1
)


model.fit(
    x_train, y_train,
    validation_data = (x_val, y_val),
    epochs = 100,
    batch_size = 64,
    callbacks = [checkpoint_cb]
)

model.save(model_path)
#TODO 추론값 0으로 수렴, mae로 다시 시도
