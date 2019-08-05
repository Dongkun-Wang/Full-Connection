import numpy as np
import tensorflow as tf
import pandas as pd

print("正在读取数据")
# 训练数据输入
x_train = abs(np.fft.fft(pd.read_csv('DataFile/x_train.csv', header=None)))
x__test = abs(np.fft.fft(pd.read_csv('DataFile/x_test.csv', header=None)))
y_train = np.array(pd.read_csv('DataFile/y_train.csv', header=None))
y__test = np.array(pd.read_csv('DataFile/y_test.csv', header=None))
print("正在训练网络")
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(500, activation=tf.nn.sigmoid),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=20)
print("测试结果")
model.evaluate(x__test, y__test)

