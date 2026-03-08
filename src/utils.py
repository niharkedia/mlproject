## will have all the common function that we will use in our project (importing the files, reading the files, saving the files, etc)

import os
import sys
import numpy as np
import pandas as pd
import dill

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)


        os.makedirs(dir_path, exist_ok=True)


        with open(file_path, "wb") as file_obj:

            logging.info("dumping object")


            dill.dump(obj, file_obj)

            logging.info("object dumping completed")

            return file_path

    except Exception as e:
        raise e
