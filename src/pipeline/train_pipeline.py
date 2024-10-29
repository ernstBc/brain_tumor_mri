import sys
from  pathlib import Path

from src.components.ingestion import IngestionComponent
from src.components.transformer import TransformerComponent
from src.components.tuner import TunerComponent
from src.components.trainer import TrainerComponent
from src.handle_exception import CustomException
from src.config.config import EPOCHS
from src.logger import logging
from src.utils import save_artifact
from dotenv import load_dotenv


if __name__=='__main__':
    # initialize components
    _=load_dotenv()
    try:
        logging.info('Starting training pipeline process')
        
        ingestion=IngestionComponent()
        transformer=TransformerComponent()
        tuner=TunerComponent()
        trainer=TrainerComponent()

        # data ingestion
        train_path,valid_path,test_path=ingestion.init_component()
        
        # transform data
        train_dataset, val_dataset, test_dataset=transformer.init_transform(train_path=train_path,valid_path=valid_path,test_path=test_path)

        #tuner component
        hp=tuner.init_tuner(train_dataset=train_dataset, test_dataset=test_dataset, epochs=3, max_trials=5)
        hp=hp.get_best_hyperparameters(1)[0]

        # trainer component
        model_path,model_id,hist=trainer.init_trainer(train_dataset=train_dataset, test_dataset=val_dataset, epochs=EPOCHS, hp=hp)

        metrics_artifacts_path=Path.joinpath(Path(model_path).parent.parent, 'metrics')
        save_artifact(Path.joinpath(metrics_artifacts_path,str(model_id)), hist.history, is_json=True)
        logging.info(f'Best model saved at {model_path}')
        logging.info(f'Model Metrics saved at {metrics_artifacts_path}')

    except Exception as e:
        logging.info('Error in the training pipeline')
        raise CustomException(e, sys)