import pandas as pd
import numpy as np

from scipy.stats import entropy
from featureval.preprocess import get_woe_df

def info_value(x, y, bins=20) : 
    x, y = np.asarray(x), np.asarray(y)
    xbins = pd.qcut(x, bins, duplicates='drop')

    N1 = (y==1).sum()
    N0 = y.shape[0] - N1
        
    df = get_woe_df(x, y, xbins)

    iv = ((df['cnt1'] / N1 - df['cnt0'] / N0) * df['woe']).sum()
    
    return iv

def corr_coef(x, y) :
    return ((x-x.mean()) * (y-y.mean())).mean() / (x.std() * y.std())

def abs_corr_coef(x, y) :
    return np.abs(((x-x.mean()) * (y-y.mean())).mean() / (x.std() * y.std()))

def mutual_information(x, y, bins=[20, 20]):  
    """  
    Assume x and y are np arrays for calculation of MI.
    """  
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    xbins, ybins = bins
    # joint distribution
    joint_prob, _, _ = np.histogram2d(x, y, bins=bins)  
    joint_prob /= joint_prob.sum()  
      
    # edge distribution  
    x_prob, _ = np.histogram(x, bins=xbins, density=True)  
    y_prob, _ = np.histogram(y, bins=ybins, density=True)  
      
    # mutual information  
    mi = entropy(x_prob, base=2) + entropy(y_prob, base=2) - entropy(joint_prob.flatten(), base=2)  
    return mi