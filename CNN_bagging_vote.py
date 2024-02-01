import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
tfrecord_files = [
    '/HOME/scz0c6m/run/dataset/training10_0/training10_0.tfrecords',
    '/HOME/scz0c6m/run/dataset/training10_1/training10_1.tfrecords',
    '/HOME/scz0c6m/run/dataset/training10_2/training10_2.tfrecords',
    '/HOME/scz0c6m/run/dataset/training10_3/training10_3.tfrecords',
    '/HOME/scz0c6m/run/dataset/training10_4/training10_4.tfrecords'
]

# Define a function to parse and create the dataset
def create_dataset(tfrecord_files, batch_size=32, shuffle_buffer=10000, validation_split=0.2):
    def _parse_function(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'label': tf.io.FixedLenFeature([], tf.int64),
                'label_normal': tf.io.FixedLenFeature([], tf.int64),
                'image': tf.io.FixedLenFeature([], tf.string)
            })

        label = features['label']
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [299, 299, 1])
        image = tf.image.resize(image, [224, 224])
        image = tf.image.grayscale_to_rgb(image)
        image = image / 255.0

        return image, label

    dataset = tf.data.Dataset.list_files(tfrecord_files)
    dataset = dataset.interleave(
        lambda file: tf.data.TFRecordDataset(file),
        cycle_length=len(tfrecord_files),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_size = len(list(dataset))
    validation_size = int(dataset_size * validation_split)
    train_size = dataset_size - validation_size

    dataset = dataset.shuffle(shuffle_buffer)
    train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, validation_dataset
train_generator, validation_generator = create_dataset(tfrecord_files)

n_models = 20

def load_all_models(n_models):
    models = []
    for i in range(n_models):
        model = tf.keras.models.load_model(f'/HOME/scz0c6m/run/CNN_models/cnn_model_{i}.h5')
        models.append(model)
    return models


all_models = load_all_models(n_models)

y_true = []
y_pred = []
for images, labels in validation_generator:
    predictions = np.zeros((images.shape[0], 5))
    for model in all_models:
        predictions += model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

with open("/HOME/scz0c6m/run/CNN_models/results.txt", "w") as results_file:
    results_file.write(f"Accuracy: {accuracy}\n")
    results_file.write("Confusion Matrix:\n")
    for row in conf_matrix:
        results_file.write(" ".join(str(x) for x in row) + "\n")

model.summary()