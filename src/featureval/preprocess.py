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

def detect_variable_type(df, cols=[],) :
    '''
    df is a DataFrame containing the variable columns, if cols is None, use all columns.
    '''
    ccols, ncols = [], []
    if not cols :
        cols = df.columns.tolist()

    for col in cols:  

        dtype = df[col].dtype  
        if dtype in ('int64', 'int32', 'int16', 'uint8', 'uint16', 'uint32', 'uint64'): 
            ncols.append(col) 
            print(f"Column '{col}' is numerical.")  
        elif dtype in ('float64', 'float32', 'float16'):
            ncols.append(col)  
            print(f"Column '{col}' is numerical.")  
        else:  
            ccols.append(col)
            print(f"Column '{col}' is categorical.")  
    
    return ccols, ncols

def get_woe_df(x, y, xbins) :
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'xbins': xbins}, orient='columns')

    df1 = df.groupby('xbins').agg({'y': [sum, lambda x: len(x)-sum(x)]}).astype(float)
    df1.columns = ['cnt1', 'cnt0']
    
    N1 = (y==1).sum()
    N0 = y.shape[0] - N1
    df1 = df1.assign(
        woe = np.log(df1['cnt1'] / N1) - np.log(df1['cnt0'] / N0)
    )
    return df1

def weight_of_evidence(x, y, bins=20) : 
    '''
    Only applicable for y is binary and x is numeric-like.
    '''    
    x, y = np.asarray(x), np.asarray(y)
    xbins = pd.qcut(x, bins, duplicates='drop')

    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'xbins': xbins}, orient='columns')

    df1 = get_woe_df(x, y, xbins).reset_index()

    df2 = df[['x', 'xbins']].merge(df1, on='xbins', how='left')
    woe = df2['woe'].values
    
    return woe