import pandas as pd
import numpy as np
from fastai.tabular.all import *


def fastai_neural_regression(path):
    """This function will (train) and return the fastai neural net regressor

    path: str
        This is the path to the csv file of the bike rental



    """
    df = pd.read_csv(path, index_col=[0])
    min_max = pd.read_csv("./data/preprocessed/cnt_min_max.csv")
    cat_names = ['season', 'yr', 'weathersit', "workingday", "holiday"]
    cont_names = ['mnth', 'hr', 'weekday', 'temp', "windspeed", "hum"]
    procs = [Categorify]
    dls = TabularDataLoaders.from_df(
        df, path, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names="cnt", bs=64)
    split = 0.8
    splits = (list(range(0, round(len(df)*split))),
              list(range(round(len(df)*split), len(df))))
    to = TabularPandas(df, procs=[Categorify],
                       cat_names=cat_names,
                       cont_names=cont_names,
                       y_names='cnt',
                       splits=splits)
    dls = to.dataloaders(bs=64)
    learn = tabular_learner(dls,
                            metrics=R2Score(),
                            layers=[500, 250],
                            n_out=1,
                            loss_func=F.mse_loss)
    try:
        learn.load('fastai_learner')
        print("The previous model was loaded!")
    except:
        learn.fit_one_cycle(168)
