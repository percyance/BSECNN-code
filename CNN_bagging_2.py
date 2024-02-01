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

        label = features['label_normal']
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

class EpochLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"  - loss: {logs.get('loss')}\n")
            f.write(f"  - accuracy: {logs.get('accuracy')}\n")
            f.write(f"  - val_loss: {logs.get('val_loss')}\n")
            f.write(f"  - val_accuracy: {logs.get('val_accuracy')}\n")

def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    return model

model = create_cnn_model()

n_models = 5  
bagging_ratio = 0.8 
epochs = 50

def train_and_save_cnn_model(model_id):
    model = create_cnn_model()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])

    train_subset = train_generator.shuffle(10000).take(int(bagging_ratio * len(list(train_generator))))
    
    log_dir = "/HOME/scz0c6m/run/CNN_models_2/logs"
    log_file = f"{log_dir}/epoch_log_model_{model_id}.txt"

    os.makedirs(log_dir, exist_ok=True)

    epoch_logging_callback = EpochLoggingCallback(log_file)

    history = model.fit(train_subset,
                        epochs=epochs,
                        validation_data=validation_generator,
                        callbacks=[epoch_logging_callback]) 

    model.save(f'/HOME/scz0c6m/run/CNN_models_2/cnn_model_{model_id}.h5')


for i in range(n_models):
    print(f"Training model {i + 1}/{n_models}")
    train_and_save_cnn_model(i)

def load_all_models(n_models):
    models = []
    for i in range(n_models):
        model = tf.keras.models.load_model(f'/HOME/scz0c6m/run/CNN_models_2/cnn_model_{i}.h5')
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

with open("/HOME/scz0c6m/run/CNN_models_2/results.txt", "w") as results_file:
    results_file.write(f"Accuracy: {accuracy}\n")
    results_file.write("Confusion Matrix:\n")
    for row in conf_matrix:
        results_file.write(" ".join(str(x) for x in row) + "\n")

model.summary()