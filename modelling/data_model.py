import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
from utils import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
                 # This method will create the model for data
                 self.embeddings = X
                 
                 # Determine chained mult-output labels
                 y2 = df[Config.TYPE_COLS[0]].astype(str).fillna('')
                 y3 = df[Config.TYPE_COLS[1]].astype(str).fillna('')
                 y4 = df[Config.TYPE_COLS[2]].astype(str).fillna('')
                 
                 # Target 'y' combines the chained sequence
                 self.y = y2 + "_" + y3 + "_" + y4
                 
                 # Create Train/Test Split
                 self.train_df, self.test_df, self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                     df, self.embeddings, self.y, test_size=0.2, random_state=seed
                 )

    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
