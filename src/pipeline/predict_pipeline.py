import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=r'artifacts\model.pkl'
            preprocessor_path=r'artifacts\preprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        rollno: int ,
        std: int ,
        gender: int,
        sibiling:int,
        place: int,
        transport: int,
        income: int,):

        self.gender = gender

        self.std = std

        self.rollno = rollno

        self.sibiling = sibiling

        self.place = place

        self.transport = transport

        self.income = income

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "rollno": [self.rollno],
                "std": [self.std],
                "gender": [self.gender],
                "sibiling": [self.sibiling],
                "place": [self.place],
                "transport": [self.transport],
                "income": [self.income],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)