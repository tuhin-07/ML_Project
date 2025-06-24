import os,sys
import dill
import pandas as pd
from src.exception import CustomExecption

def save_object(file_path,obj):
    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,'wb') as f:
            dill.dump(obj,f)

    except Exception as e:
        raise CustomExecption(e,sys)
    