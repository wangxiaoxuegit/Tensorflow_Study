
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf


# load data
DATA_PATH = 'D:\\study\\tensorflow\\dataset\\mnist\\mnist.npz'
with np.load(DATA_PATH) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']


# use tf.data generate dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))


# shuffle and batch dataset
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset =test_dataset.batch(BATCH_SIZE)


# build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(train_dataset, epochs=5)


# evaluate
model.evaluate(test_dataset)
