import pandas as pd
import numpy as np

def categories_to_integer(x) :
    '''
    Assume x is a discrete variable and transform it into integer encoding.
    '''
    x_uni = sorted(list(set(x)))

    mapping = {}  
    for i, category in enumerate(x_uni):  
        mapping[category] = i  

    x = np.asarray(x)
    vfunc = np.vectorize(lambda a: mapping[a])
    x1 = vfunc(x)

    return x1

def zscore_normalize(x) :
    '''
    Assume x is a numeric or encoded as numeric variable, and transform it by zscore normalization.
    '''
    return (x - np.mean(x)) / np.std(x)


