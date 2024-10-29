import os
import time 
import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np
import sys


from dataclasses import dataclass
from src.config.config import MODELS_DATA_PATH, IMG_SHAPE, EPOCHS, TENSORBOARD_DIR
from src.logger import logging
from src.handle_exception import CustomException


@dataclass
class TrainerConfig:
    model_path:str=MODELS_DATA_PATH


class TrainerComponent:
    def __init__(self):
        self.trainer_config=TrainerConfig()
        self.model_id=time.time()
        self.export_dir=os.path.join(self.trainer_config.model_path, str(self.model_id))
        self.train_logs=os.path.join(TENSORBOARD_DIR, str(self.model_id))

    def init_trainer(self, train_dataset, test_dataset, hp, epochs=EPOCHS):
        try:
            logging.info('Initializing Trainer Process ')
            model=build_model(hp)
            callbacks=build_callbacks(log_dir=self.train_logs)
            hist=model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=callbacks)

            tf.saved_model.save(model, self.export_dir)
            return (self.export_dir, self.model_id, hist)
        
        except Exception as e:
            logging.info('Error occurs while training process')
            logging.error(e)
            raise CustomException(e, sys)
        

def build_callbacks(log_dir):
    tensorboard_call=tf.keras.callbacks.TensorBoard(
        log_dir=log_dir)
    early_call=tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True)
    
    return [early_call, tensorboard_call]
        

def build_model(hp):
    layers=hp['num_con_layers']
    activation=hp['activation_fn']
    units=hp['units']
    lr=hp['lr']

    inputs=L.Input(shape=(IMG_SHAPE,IMG_SHAPE, 3))
    for l in range(layers):
        if l == 0:
            x=L.Conv2D(filters=16, kernel_size=7, padding='valid', activation=activation)(inputs)
            x=L.MaxPooling2D(pool_size=(4,4))(x)
            x=L.Dropout(0.3)(x)
            continue

        x=L.Conv2D(filters=np.clip(2 ** (5 + l), a_min=32, a_max=256), kernel_size=3, padding='same', activation=activation)(x)
        x=L.MaxPooling2D()(x)
        x=L.Dropout(0.3)(x)

    x=L.GlobalAveragePooling2D()(x)
    x=L.Dense(units=units, activation=activation)(x)
    x=L.Dropout(0.3)(x)
    output=L.Dense(units=4, activation='softmax')(x)

    model=tf.keras.models.Model(inputs=inputs, outputs=output)


    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])
    return model