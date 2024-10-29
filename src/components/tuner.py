import tensorflow as tf
from tensorflow.keras import layers as L
import keras_tuner
import numpy as np

from src.config.config import IMG_SHAPE, TUNER_DATA_PATH
from dataclasses import dataclass


@dataclass
class TunerConfig:
    tuner_path:str=TUNER_DATA_PATH


class Tuner:
    def __init__(self):
        self.tuner_config=TunerConfig()

    def init_tuner(self, train_dataset, test_dataset, epochs, max_trials=15):
        tuner=keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            objective='val_loss',
            max_trials=max_trials,
            directory=self.tuner_config.tuner_path,
            overwrite=True
        )
        tuner.search(train_dataset, epochs=epochs, validation_data=test_dataset)

        return tuner

    


def build_model(hp):
    inputs=L.Input(shape=(IMG_SHAPE,IMG_SHAPE, 3))

    layers=hp.Choice('num_con_layers', [4,5,6])
    activation=hp.Choice('activation_fn', 'relu', 'selu')
    for l in range(layers):
        if l == 0:
            x=L.Conv2D(filters=32, kernel_size=3, padding='same', activation=activation)(inputs)
            x=L.MaxPooling2D(pool_size=(4,4))(x)
            continue

        x=L.Conv2D(filters=np.clip(2 ** (6 + l), a_min=32, a_max=512), kernel_size=3, padding='same', activation=activation)(x)
        x=L.MaxPooling2D()(x)

    units=hp.Choice('units', [64, 128,256])

    x=L.Flatten()(x)
    x=L.Dense(units=units, activation=activation)(x)
    x=L.Dense(units=4, activation='softmax')(x)

    model=tf.keras.models.Model(inputs=inputs, outputs=x)


    lr=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.F1Score(threshold=0.5)])
    return model
     