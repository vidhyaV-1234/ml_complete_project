import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object
@dataclass
class DataTransformatConfig:
    preproccessor_ob_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformatConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_columns=["rollno","std","gender","transport ","place","siblings","income","dropout"]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalsr",StandardScaler())
                ]
            )
            '''for catogorical data
            cat_pipeline=Pipeline( 
                steps=[
                    ("imputer",SimpleImputer(strategy))
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StansdardScalar())
                ]
                )'''
            logging.info("numerical columns standard scaling completed")
            return num_pipeline


        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")
            preproccessor_obj=self.get_data_transformer_object()
            target_column_name="dropout"
            numerical_columns=[
                "rollno","std","gender","transport ","place","siblings","income","dropout"
            ]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preproccessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preproccessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)
            ]
            save_object(

                file_path=self.data_transformation_config.preproccessor_ob_file_path,
                obj=preproccessor_obj

            )

            logging.info(f"saved preprocessing object.")

            return (train_arr,test_arr,self.data_transformation_config.preproccessor_ob_file_path,)

        except Exception as e:
                raise CustomException(e, sys)

