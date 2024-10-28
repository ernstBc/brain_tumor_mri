import os
import shutil
import sys
import random

from dataclasses import dataclass
from src.handle_exception import CustomException
from src.logger import logging
from src.config.config import DATA_PATH, TEST_DATA, TRAIN_DATA, VAL_DATA
from dotenv import load_dotenv
import kagglehub
from typing import List

_=load_dotenv()


@dataclass
class IngestionConfig:
    train_data_path:str = TRAIN_DATA
    test_data_path:str = TEST_DATA
    eval_data_path:str = VAL_DATA
    raw_data_path: str = os.path.join(DATA_PATH, 'data')



class IngestionComponent:
    def __init__(self):
        self.ingestion_config=IngestionConfig()
        

    def init_component(self):
        logging.info('Init Data Ingestion Compoment')
        try:
            if not os.path.exists(self.ingestion_config.raw_data_path):
                assert 'KAGGLE_KEY' in os.environ
                assert 'KAGGLE_USERNAME' in os.environ

                os.makedirs(self.ingestion_config.raw_data_path)
                os.makedirs(self.ingestion_config.train_data_path)
                os.makedirs(self.ingestion_config.eval_data_path)
                os.makedirs(self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info('Error: Kaggle secret key is not in the enviroment.')
            raise CustomException(e, sys)
        
        download_dataset(self.ingestion_config.raw_data_path)
        
        split_dataset(data_path=self.ingestion_config.raw_data_path,
                      train_path=self.ingestion_config.train_data_path,
                      val_path=self.ingestion_config.eval_data_path,
                      test_path=self.ingestion_config.test_data_path,
                      split_size=[0.8, 0.1, 0.1])

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.eval_data_path,
            self.ingestion_config.test_data_path
        )
        


def download_dataset(destination_path:str, dataset_name="rm1000/brain-tumor-mri-scans"):
        '''
        Download the dataset from kaggle and save it in the path folder

        return:
            None
        '''
        try:
            if len(os.listdir(destination_path))==0:
                logging.info('Downloading dataset from kaggle')
                path=kagglehub.dataset_download(dataset_name)
                for folder in os.listdir(path):
                    shutil.move(os.path.join(path, folder), destination_path)
            else:
                logging.info(f'dataset is already downloaded at {destination_path}')

        except Exception as e:
            raise CustomException('Error: Error while trying to download the dataset from kaggle.', sys)
    

def split_dataset(data_path:str,
                      train_path:str, 
                      val_path:str,
                      test_path:str, 
                      split_size:List,
                      shuffle=True):
        '''
        Split the dataset into two subsets with a proportion of (1-split_test), (split_test) and save it
        in train_path directory and test_path directory

        return:
            None
        '''

        assert len(split_size)==3

        categories_folder=os.listdir(data_path)

        for category in categories_folder:
            elements=os.listdir(os.path.join(data_path, category))
            num_elements=len(elements)
            num_train=int(num_elements * split_size[0])
            num_val=int(num_elements * split_size[1])
            num_test=int(num_elements * split_size[2])

            if shuffle:
                random.shuffle(elements)
            
            train=elements[:num_train]
            val=elements[num_train:num_train+num_val]
            test=elements[-num_test:]

            for element in train:
                os.makedirs(os.path.join(train_path, category), exist_ok=True)
                shutil.copy(os.path.join(data_path, category, element), os.path.join(train_path, category, element))

            for element in val:
                os.makedirs(os.path.join(val_path, category), exist_ok=True)
                shutil.copy(os.path.join(data_path, category, element), os.path.join(val_path, category, element))

            for element in test:
                os.makedirs(os.path.join(test_path, category), exist_ok=True)
                shutil.copy(os.path.join(data_path, category, element), os.path.join(test_path, category, element))
        

if __name__=='__main__':
    ingest=IngestionComponent()
    ingest.init_component()


