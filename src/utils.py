import os
import sys
import pickle
import yaml

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: object)->None:
    try:
        logging.info('Save object fn. start')
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    
    except CustomException as e:
        logging.info("Exception occured in fn. save_object")
        raise CustomException(e,sys)
    

def read_yaml_file(filename:str)->dict:
    try:
        with open(filename,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)
    
