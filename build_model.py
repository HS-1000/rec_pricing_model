import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

"""
Input -> LSTM -> Attention -> Dense -> Output
"""

class AdditiveAttention(layers.Layer):
	def __init__(self, units):
		super(AdditiveAttention, self).__init__()
		self.W1 = layers.Dense(units)
		self.W2 = layers.Dense(units)
		self.V = layers.Dense(1)

	def call(self, encoder_outputs):
		# encoder_outputs: (batch_size, time_steps, hidden_size)
		score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(encoder_outputs)))
		attention_weights = tf.nn.softmax(score, axis=1)  # shape: (batch_size, time_steps, 1)
		context_vector = attention_weights * encoder_outputs
		context_vector = tf.reduce_sum(context_vector, axis=1)  # shape: (batch_size, hidden_size)
		return context_vector

def build_lstm_attention_model(input_shape, lstm_units=64, attention_units=32, dense_units=32):
	inputs = layers.Input(shape=input_shape)  # shape: (timesteps, num_features)
	lstm_out = layers.LSTM(lstm_units, return_sequences=True)(inputs)  # (batch_size, timesteps, lstm_units)
	context_vector = AdditiveAttention(attention_units)(lstm_out)  # (batch_size, lstm_units)
	x = layers.Dropout(0.2)(context_vector)
	x = layers.Dense(dense_units, activation='relu')(x)
	output = layers.Dense(1, activation='linear')(x)
	model = Model(inputs=inputs, outputs=output)
	model.compile(optimizer='adam', loss='mse', metrics=['mae']) # 이게 필요할까, 훈련과정에서 컴파일 할텐데
	return model

model = build_lstm_attention_model((32, 16))
model.summary()
model.save("./rec_pricing.keras")
