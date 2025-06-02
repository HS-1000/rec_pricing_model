import sys
sys.path.append(r"C:\rec_pricing")

import numpy as np
import build_model

model = build_model.build_lstm_attention_model((32, 16))
model.summary()

x_dummy = np.random.rand(5, 32, 16).astype(np.float32)
y_dummy = model.predict(x_dummy)

print(y_dummy)
