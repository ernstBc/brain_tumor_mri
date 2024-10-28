from pathlib import Path
import os

MAIN_PATH=Path(os.getcwd())
DATA_PATH=Path.joinpath(MAIN_PATH, 'data')
TRAIN_DATA=Path.joinpath(DATA_PATH, 'train')
VAL_DATA=Path.joinpath(DATA_PATH, 'val')
TEST_DATA=Path.joinpath(DATA_PATH, 'test')


if __name__=='__main__':
    print('main_path------', DATA_PATH)