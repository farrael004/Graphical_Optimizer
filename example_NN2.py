import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

from hyperoptimize import GraphicalOptimizer

import pathlib

# Loading data
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = len(class_names)

# Creating model, prediction and performance functions
def model_function(params, X, y):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    epochs = params['epochs']
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    results = {"Accuracy while training": [acc],
               "Validation accuracy": [val_acc],
               "Loss": [loss],
               "Validation loss": [val_loss]}

    return model, results


def prediction_function(model, X):
    y_pred = model.predict(val_ds)
    return y_pred


def performance_function(y_test, y_pred):
    results = {"Accuracy": 1}

    return results


# Creating hyperparameter dictionary
hyperparameters = {"epochs": [5, 10]}


# Performing optimization
opt = GraphicalOptimizer(model_function=model_function,
                         prediction_function=prediction_function,
                         performance_function=performance_function,
                         performance_parameter="Accuracy",
                         hyperparameters=hyperparameters,
                         optimizer="random",
                         max_num_combinations=5,
                         cross_validation=2,
                         max_num_of_parallel_processes=1,
                         parallel_combinations=2,
                         create_GUI=True)

opt.fit(np.zeros(2), np.zeros(2))
