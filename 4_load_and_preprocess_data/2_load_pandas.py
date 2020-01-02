
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf


# load data
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
print(csv_file)
df = pd.read_csv(csv_file)
print(df.head())
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
print(df.head())


# use tf.data
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
    print('feature:{} , target:{}'.format(feat, targ))
train_dataset = dataset.shuffle(len(df)).batch(1)


# build and train model
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model


model = get_compiled_model()
model.fit(train_dataset, epochs=15)

