
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd


# load data
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print(train.head())
train_y = train.pop('Species')
test_y = test.pop('Species')
print(train.head())


# define input function
def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


# define feature columns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


# estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=3
)

# training
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000
)

# evaluate
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False)
)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

