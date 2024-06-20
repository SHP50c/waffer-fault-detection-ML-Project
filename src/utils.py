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
        logging.info("Exception occured in fn:save_object")
        raise CustomException(e,sys)
    

def read_yaml_file(filename:str)->dict:
    logging.info('read_yaml_file fn. start')
    try:
        with open(filename,"rb") as yaml_file:
            obj=yaml.safe_load(yaml_file)
        logging.info('yaml file loaded succesfully')
        return obj
    except Exception as e:
        logging.info('Exception occured in fn:read_yaml_file')
        raise CustomException(e,sys)
    
def load_object(file_path: str)->object:
    logging.info('Load object fn. start')
    try:
        with open(file_path,"rb") as file_obj:
            obj=pickle.load(file_obj)
        logging.info('object was loaded sucessfully')
        return obj
    except CustomException as e:
        logging.info('Exception occured in fn:load_object')
        raise CustomException(e,sys)