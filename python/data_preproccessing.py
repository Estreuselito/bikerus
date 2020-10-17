import pandas as pd

def import_data(path):
    """This function imports the data and does some minor transforming

    path: str
        The (relative) path to the dataset

    Example
    -------
    data = import_data("../data/BikeRental.csv")
    """
    data = pd.read_csv(path)
    return data