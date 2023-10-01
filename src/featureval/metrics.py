import pandas as pd
import numpy as np

def info_value(x, y, bins=20) : 
    xbins = pd.qcut(x, 20, duplicates='drop')

    df = np.concatenate(
        [x.reshape(len(x),1), y.reshape(len(y),1), xbins.reshape(len(xbins),1)],
        axis=1,
    )
    df = pd.DataFrame(df, columns=['x', 'y', 'xbins'])
    
    df = df.groupby('xbins').agg({'y': [sum, lambda x: len(x)-sum(x)]}).astype(float)
    df.columns = ['cnt1', 'cnt0']
    
    N1 = (y==1).sum()
    N0 = y.shape[0] - N1
    df = df.assign(
        woe = np.log(df['cnt1'] / N1) - np.log(df['cnt0'] / N0)
    )

    iv = ((df['cnt1'] / N1 - df['cnt0'] / N0) * df['woe']).sum()
    
    return iv

def corr_coef(x, y) :
    return ((x-x.mean()) * (y-y.mean())).mean() / (x.std() * y.std())

def abs_corr_coef(x, y) :
    return np.abs(((x-x.mean()) * (y-y.mean())).mean() / (x.std() * y.std()))

