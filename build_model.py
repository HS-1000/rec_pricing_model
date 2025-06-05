import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

"""
Input -> LSTM -> Attention -> Dense -> Output
"""

class AdditiveAttention(layers.Layer):
	def __init__(self, units, **kwargs):
		super(AdditiveAttention, self).__init__(**kwargs)
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
	output = layers.Dense(1, activation='tanh')(x)
	model = Model(inputs=inputs, outputs=output)
	model.compile(optimizer='adam', loss='mse', metrics=['mae']) # 이게 필요할까, 훈련과정에서 컴파일 할텐데
	return model

class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, embedding_dim, num_heads=8, **kwargs):
		super(MultiHeadAttention, self).__init__(**kwargs)
		self.embedding_dim = embedding_dim # d_model
		self.num_heads = num_heads
		assert embedding_dim % self.num_heads == 0
		self.projection_dim = embedding_dim // num_heads
		self.query_dense = tf.keras.layers.Dense(embedding_dim)
		self.key_dense = tf.keras.layers.Dense(embedding_dim)
		self.value_dense = tf.keras.layers.Dense(embedding_dim)
		self.dense = tf.keras.layers.Dense(embedding_dim)

	def build(self, input_shape):
		self.query_dense.build(input_shape)
		self.key_dense.build(input_shape)
		self.value_dense.build(input_shape)
		self.dense.build(input_shape)
		super().build(input_shape)

	def scaled_dot_product_attention(self, query, key, value, mask=None):
		matmul_qk = tf.matmul(query, key, transpose_b=True)
		depth = tf.cast(tf.shape(key)[-1], tf.float32)
		logits = matmul_qk / tf.math.sqrt(depth)
		if mask is not None:
			pass
		attention_weights = tf.nn.softmax(logits, axis=-1)
		output = tf.matmul(attention_weights, value)
		return output, attention_weights

	def split_heads(self, x, batch_size):
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, inputs, mask=None, self_attention=True):
		if self_attention:
			# x.shape = [batch_size, seq_len, embedding_dim]
			batch_size = tf.shape(inputs)[0]

			# (batch_size, seq_len, embedding_dim)
			query = self.query_dense(inputs)
			key = self.key_dense(inputs)
			value = self.value_dense(inputs)
		else:
			# x.shape = {key, value, query : [batch_size, seq_len, embedding_dim]}
			batch_size = tf.shape(inputs["query"])[0]

			query = self.query_dense(inputs["query"])
			key = self.key_dense(inputs["key"])
			value = self.value_dense(inputs["value"])

		# (batch_size, num_heads, seq_len, projection_dim)
		query = self.split_heads(query, batch_size)
		key = self.split_heads(key, batch_size)
		value = self.split_heads(value, batch_size)

		scaled_attention, _ = self.scaled_dot_product_attention(query, key, value, mask=mask)
		# (batch_size, seq_len, num_heads, projection_dim)
		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

		# (batch_size, seq_len, embedding_dim)
		concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
		outputs = self.dense(concat_attention)
		return outputs
	
	def get_config(self):
		config = super().get_config()
		config.update({
			"embedding_dim" : self.embedding_dim,
			"num_heads" : self.num_heads
		})
		return config

	@classmethod
	def from_config(cls, config=None, **kwargs):
		if config is None:
			config = {}
		elif not isinstance(config, dict):
			raise TypeError(f"Expected dict for 'config', but got {config}")
		config = {**config, **kwargs}
		return cls(**config)

class AddNorm(tf.keras.layers.Layer):
	def __init__(self, epsilon=1e-6, **kwargs):
		super(AddNorm, self).__init__(**kwargs)
		self.epsilon = epsilon
		self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)

	def build(self, input_shape):
		super().build(input_shape)

	def call(self, inputs):
		# inputs는 리스트 형태로 [x, y] 형태입니다.
		x, y = inputs
		# 두 입력을 더합니다.
		output = x + y
		# Layer Normalization을 적용합니다.
		return self.layer_norm(output)
	
	def get_config(self):
		config = super().get_config()
		config.update({
			"epsilon" : self.epsilon
		})
		return config

	@classmethod
	def from_config(cls, config=None, **kwargs):
		if config is None:
			config = {}
		elif not isinstance(config, dict):
			raise TypeError(f"Expected dict for 'config', but got {config}")
		config = {**config, **kwargs}
		return cls(**config)

class FeedForward(tf.keras.layers.Layer):
	def __init__(self, embedding_dim, dff, **kwargs):
		super(FeedForward, self).__init__(**kwargs)
		self.embedding_dim = embedding_dim
		self.dff = dff
		self.ffn = tf.keras.Sequential([
			tf.keras.layers.Dense(dff, activation="relu"),
			tf.keras.layers.Dense(embedding_dim)
		])

	def build(self, input_shape):
		super().build(input_shape) # Dense가 Sequential 이기때문에 자동으로 빌드됨

	def call(self, inputs):
		return self.ffn(inputs)
	
	def get_config(self):
		config = super().get_config()
		config.update({
			"embedding_dim" : self.embedding_dim,
			"dff" : self.dff
		})
		return config

	@classmethod
	def from_config(cls, config=None, **kwargs):
		if config is None:
			config = {}
		elif not isinstance(config, dict):
			raise TypeError(f"Expected dict for 'config', but got {config}")
		config = {**config, **kwargs}
		return cls(**config)

class TransformerBlock(tf.keras.layers.Layer): # Encoder
	# 포지셔널 인코딩을 포함하지 않음
	def __init__(self, embadding_dim, dff, heads, drop1=False, drop2=False, **kwargs):
		super(TransformerBlock, self).__init__(**kwargs)
		self.embadding_dim = embadding_dim
		self.dff = dff
		self.heads = heads
		self.drop1_init = drop1
		self.drop2_init = drop2
		self.mh_att = MultiHeadAttention(embadding_dim, num_heads=heads)
		self.ffn = FeedForward(embadding_dim, dff)
		self.addnorm1 = AddNorm()
		self.addnorm2 = AddNorm()
		if drop1:
			self.drop1 = tf.keras.layers.Dropout(drop1)
		else:
			self.drop1 = False
		if drop2:
			self.drop2 = tf.keras.layers.Dropout(drop2)
		else:
			self.drop2 = drop2

	def build(self, input_shape):
		self.mh_att.build(input_shape)
		if self.drop1:
			self.drop1.build(input_shape)
		self.addnorm1.build(input_shape)
		self.ffn.build(input_shape)
		if self.drop2:
			self.drop2.build(input_shape)
		self.addnorm2.build(input_shape)
		super().build(input_shape)

	def call(self, inputs):
		att_out = self.mh_att(inputs)
		if self.drop1:
			att_out = self.drop1(att_out)
		addnorm1_out = self.addnorm1([inputs, att_out])
		ffn_out = self.ffn(addnorm1_out)
		if self.drop2:
			ffn_out = self.drop2(ffn_out)
		return self.addnorm2([addnorm1_out, ffn_out])

	def get_config(self):
		config = super().get_config()
		config.update({
			"embadding_dim" : self.embadding_dim,
			"dff" : self.dff,
			"heads" : self.heads,
			"drop1" : self.drop1_init,
			"drop2" : self.drop2_init
		})
		return config

	@classmethod
	def from_config(cls, config=None, **kwargs):
		if config is None:
			config = {}
		elif not isinstance(config, dict):
			raise TypeError(f"Expected dict for 'config', but got {config}")
		config = {**config, **kwargs}
		return cls(**config)

if __name__ == "__main__":
	inputs = layers.Input(shape=(32, 16))
	x = TransformerBlock(
		embadding_dim = 16,
		dff = 64,
		heads = 2,
		drop1 = 0.05,
		drop2 = 0.05
	)(inputs)
	x = TransformerBlock(
		embadding_dim = 16,
		dff = 64,
		heads = 2,
		drop1 = 0.05,
		drop2 = 0.05
	)(x)
	x = TransformerBlock(
		embadding_dim = 16,
		dff = 64,
		heads = 2,
		drop1 = 0.05,
		drop2 = 0.05
	)(x)
	x = TransformerBlock(
		embadding_dim = 16,
		dff = 64,
		heads = 2,
		drop1 = 0.05,
		drop2 = 0.05
	)(x)
	x = TransformerBlock(
		embadding_dim = 16,
		dff = 64,
		heads = 2,
		drop1 = 0.05,
		drop2 = 0.05
	)(x)
	x = TransformerBlock(
		embadding_dim = 16,
		dff = 64,
		heads = 2,
		drop1 = 0.05,
		drop2 = 0.05
	)(x)
	x = layers.GlobalAveragePooling1D()(x)
	x = layers.Dense(32, activation="relu")(x)
	outputs = layers.Dense(1, activation="tanh")(x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	model.summary()
	model.save("./rec_pricing.keras")

