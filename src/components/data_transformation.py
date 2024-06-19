import os
import sys

import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    artifact_dir=os.path.join(artifact_folder)
    transformed_train_file_path=os.path.join(artifact_dir,'train.npy')
    transformed_test_file_path=os.path.join(artifact_dir,'test.npy')
    transformed_object_file_path=os.path.join(artifact_dir,'preprocessor.pkl')


class DataTransformation:
    def __init__(self,feature_store_file_path):
        self.feature_store_file_path=feature_store_file_path
        self.data_transformation_config=DataTransformationConfig()



    @staticmethod
    def get_data(feature_store_file_path:str)->pd.DataFrame:
        """
        Method Name :   get_data
        Description :   This method reads all the validated raw data from the feature_store_file_path and returns a pandas DataFrame containing the merged data. 
        
        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            logging.info('Reading the raw data from wafer.csv stored in artifact folder')
            data=pd.read_csv(feature_store_file_path)
            data.rename(columns={"Good/Bad": TARGET_COLUMN},inplace=True)
            
            return data
        
        except CustomException as e:
            logging.info('Exception occured in fn:get_data')
            raise CustomException(e,sys)
        

    def get_data_transformer_object(self):
        try:
            logging.info('Data Transformation initiated')
            imputer_step=('imputer',SimpleImputer(strategy='constant',fill_value=0))
            scaler_step=('scaler',RobustScaler())

            logging.info('Initiating pipeline and creating object of pipeline')
            preprocessor=Pipeline(
                steps=[imputer_step,scaler_step]
            )
            logging.info('Data transformation pipeline complete')
            return preprocessor
        
        except CustomException as e:
            logging.info('Exception occured in fn:get_data_transformer_object')
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self):
        """
            Method Name :   initiate_data_transformation
            Description :   This method initiates the data transformation component for the pipeline 
            
            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        logging.info('Data Transformation has started')
        try:
            dataframe=self.get_data(feature_store_file_path=self.feature_store_file_path)

            logging.info('segregating independent and dependent values')
            X=dataframe.drop(columns='TARGET_COLUMN')
            y=np.where(dataframe[TARGET_COLUMN]==-1,0,1) #replacing the -1 with 0 for model training

            logging.info('splitting the data into training and testing dataset')
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

            logging.info("Obtaining object of preprocessor from fn: get_data_transformation_object")
            preprocessor_obj=self.get_data_transformer_object()

            logging.info('Applying the transformation')
            X_train_scaled=preprocessor_obj.fit_transform(X_train)
            X_test_scaled=preprocessor_obj.transform(X_test)
            
            logging.info('Concatinating train and test input_features_arr with target_feature_arr')
            train_arr=np.c_[X_train_scaled,np.array(y_train)]
            test_arr=np.c_[X_test_scaled,np.array(y_test)]
            
            logging.info('Saving of processor.pkl file started')
            preprocessor_path=self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
            save_object(file_path=preprocessor_path,obj=preprocessor_obj)
            logging.info('preprocessor.pkl is created and saved')

            return(
                train_arr,
                test_arr,
                preprocessor_path
            )
        
        except CustomException as e:
            logging.info('Exception occured in fn:initiate_data_transformation')
            raise CustomException(e,sys)