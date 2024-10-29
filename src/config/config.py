from pathlib import Path
import os

# set directories
MAIN_PATH=Path(os.getcwd())
DATA_PATH=Path.joinpath(MAIN_PATH, 'data')
TRAIN_DATA=Path.joinpath(DATA_PATH, 'train')
VAL_DATA=Path.joinpath(DATA_PATH, 'val')
TEST_DATA=Path.joinpath(DATA_PATH, 'test')
TUNER_DATA_PATH=Path.joinpath(MAIN_PATH, 'artifacts', 'tuner_artifacts')
MODELS_DATA_PATH=Path.joinpath(MAIN_PATH, 'artifacts', 'models')


# hyperparams
BATCH_SIZE=32
IMG_SHAPE=512
