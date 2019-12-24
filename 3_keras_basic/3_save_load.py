from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
import time

# load data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# create model
def create_model():
    model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()




# check point solution--1
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
# model.fit(train_images,
#           train_labels,
#           epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])
#
# model1 = create_model()
# loss, acc = model1.evaluate(test_images,  test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
# model1.load_weights(checkpoint_path)
# loss, acc = model1.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))




# check point  solution--2
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# # 创建一个回调，每 5 个 epochs 保存模型的权重
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     period=5)
# # 创建一个新的模型实例
# model = create_model()
# # 使用 `checkpoint_path` 格式保存权重
# model.save_weights(checkpoint_path.format(epoch=0))
# # 使用新的回调*训练*模型
# model.fit(train_images,
#         train_labels,
#         epochs=50,
#         callbacks=[cp_callback],
#         validation_data=(test_images, test_labels),
#         verbose=0)
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# latest
# # 创建一个新的模型实例
# model1 = create_model()
# # 加载以前保存的权重
# model1.load_weights(latest)
# # 重新评估模型
# loss, acc = model1.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))




# save weights
# model.fit(train_images, train_labels, epochs=10, verbose=2)
# # 保存权重
# model.save_weights('./checkpoints/my_checkpoint')
# # 创建模型实例
# model = create_model()
# # Restore the weights
# model.load_weights('./checkpoints/my_checkpoint')
# # Evaluate the model
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))




# save model  solution--1
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('model/my_model.h5')
new_model = keras.models.load_model('model/my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))




# save model  solution--2
# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
# saved_model_path = "./saved_models/{}".format(int(time.time()))
# tf.keras.experimental.export_saved_model(model, saved_model_path)
# saved_model_path
# new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
# new_model.summary()
# new_model.compile(optimizer=model.optimizer,  # 保留已加载的优化程序
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
# loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
