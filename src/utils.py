import pickle
import sys
import json
import os

from src.logger import logging
from src.handle_exception import CustomException
from src.config.config import IMG_SHAPE



def save_artifact(artifact_path, object, is_json=False):
    try: 
        dir_path=os.path.dirname(artifact_path)
        os.makedirs(dir_path, exist_ok=True)

        if is_json:
            with open(artifact_path, "w") as file: 
                json.dump(object, file, sort_keys=True, indent=4)
        else:
            with open(artifact_path, 'wb') as file:
              pickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info('Artefacto guardado en artifacts')
    except Exception as e:
        logging.info('Error al crear el artifacto')
        raise CustomException(e, sys)
    