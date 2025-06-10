import tensorflow as tf
import numpy as np
import features
from build_model import TransformerBlock
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

rec_path = "./data/rec.csv"
demand_path = "./data/demand.csv"
model_path = "./rec_pricing.keras"
lr_rate = 0.0005

def custom_accuracy(y_true, y_pred):
    # tf.print(y_true.shape)
    # tf.print(y_pred.shape)
    y_true = tf.reshape(y_true, [-1]) # [None, 1] -> [None]
    err = tf.abs(y_true - y_pred)
    correct = tf.less(err, 0.2)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def lr_scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * 0.9997

def custom_sigmoid_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1]) # [None, 1] -> [None]
    err = tf.abs(y_true - y_pred)
    loss = tf.sigmoid(10 * (err - 0.2))
    return tf.reduce_mean(loss)

lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

model = tf.keras.models.load_model(
	model_path,
	custom_objects = {
		"TransformerBlock" : TransformerBlock,
        "custom_accuracy" : custom_accuracy,
	},
    compile = False
)

adam_opt = tf.keras.optimizers.Adam(learning_rate=lr_rate)

model.compile(
    optimizer = adam_opt,
    loss = custom_sigmoid_loss,
    # loss = "mse",
    metrics = [custom_accuracy]
)

model.summary()

x, y = features.get_xy(rec_path=rec_path, demand_path=demand_path)
x, y = x.astype('float32'), y.astype('float32')
x_train, y_train = x[9:690], y[9:690]
x_val, y_val = x[690:], y[690:]


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
    epochs = 200,
    batch_size = 32,
    callbacks = [checkpoint_cb, lr_cb]
)

model.save(model_path)
