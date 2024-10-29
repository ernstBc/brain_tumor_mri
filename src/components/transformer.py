from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys

from src.logger import logging
from src.handle_exception import CustomException
from src.logger import logging
from src.config.config import BATCH_SIZE



class TransformerComponent:
    def __init__(self):
        pass
    
    def init_transform(self,train_path, valid_path, test_path, batch_size=BATCH_SIZE):

        try:
            logging.info('Creating datasets with transformation componenent')
            train_dataset=keras.utils.image_dataset_from_directory(train_path,
                                                                labels='inferred',
                                                                batch_size=batch_size,
                                                                image_size=(512, 512))
            
            val_dataset=keras.utils.image_dataset_from_directory(valid_path,
                                                                labels='inferred',
                                                                batch_size=batch_size,
                                                                image_size=(512, 512))
            
            test_dataset=keras.utils.image_dataset_from_directory(test_path,
                                                                labels='inferred',
                                                                batch_size=batch_size,
                                                                image_size=(512, 512))
            return train_dataset, val_dataset, test_dataset
        except Exception as e:
            logging.info('Error while transform process')
            raise CustomException(e, sys)
    
    
    def preprocessing(img, label=None):
        img=np.array(img)
        img=tf.image.resize(img, size=(512,512))
        img=tf.cast(img, dtype=tf.float32) / 255.0

        if label is not None:
            return img, label
        return img 
        
        
