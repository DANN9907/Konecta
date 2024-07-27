import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import pandas as pd


class ML_skills():
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def data_exp(self, method: Literal['head', 'info', 'nulls']):

        match method:
            case 'head':
                print(self.data.head(5))
            case 'info':
                print(self.data.info())
            case 'nulls':
                print(self.data.isnull().sum())

    def data_clean(self, '')

a = ML_skills('train.csv')
a.data_exp('nulls')

