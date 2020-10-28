# this file contains all functions, which are used in order to preprocess the data

# imports
import pandas as pd
import bz2
import pickle
import _pickle as cPickle


def import_data(path, **kwargs):
    """This function imports the data and does some minor transforming

    path: str
        The (relative) path to the dataset

    Example
    -------
    data = import_data("../data/BikeRental.csv")
    """
    data = pd.read_csv(path, **kwargs)
    return data

# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
    """loads data and compresses this into a .pbz2 file which reduces it immensly in size

    Parameters
    ----------
    title : str
        the title/path of the file 
    data : dataframe
        the dataframe which shall be compressed
    """
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
      cPickle.dump(data, f)

  # Load any compressed pickle file
def decompress_pickle(file):
    """a function which loads .pbz2 files and decompresses them

    Parameters
    ----------
    file : str
        the path location of the file

    Returns
    -------
    dataframe
        this function returns a dataframe
    """
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data