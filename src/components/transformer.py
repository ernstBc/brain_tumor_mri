from tensorflow import keras
import tensorflow as tf
import numpy as np

from src.logger import logging
from src.handle_exception import CustomException
from src.config.config import BATCH_SIZE



class TransformerComponent:
    def __init__(self, train_path, valid_path, test_path):
        self.train_path=train_path
        self.valid_path=valid_path
        self.test_path=test_path

    
    def init_transform(self, batch_size=BATCH_SIZE):

        train_dataset=keras.utils.image_dataset_from_directory(self.train_path,
                                                               labels='inferred',
                                                               batch_size=batch_size,
                                                               image_size=(512, 512))
        
        val_dataset=keras.utils.image_dataset_from_directory(self.valid_path,
                                                             labels='inferred',
                                                             batch_size=batch_size,
                                                             image_size=(512, 512))
        
        test_dataset=keras.utils.image_dataset_from_directory(self.test_path,
                                                              labels='inferred',
                                                              batch_size=batch_size,
                                                              image_size=(512, 512))
        
        return train_dataset, val_dataset, test_dataset
    
    
    def preprocessing(img, label=None):
        img=np.array(img)
        img=tf.image.resize(img, size=(512,512))
        img=tf.cast(img, dtype=tf.float32) / 255.0

        if label is not None:
            return img, label
        return img 
        
        
