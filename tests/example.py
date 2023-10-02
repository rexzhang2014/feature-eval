
from featureval import mRMR
from featureval.metrics import mutual_information, abs_corr_coef

from featureval.preprocess import categories_to_integer, zscore_normalize

if __name__ == '__main__' :
    import pandas as pd

    df = pd.read_csv('C:/workspace/feature-eval/data/investing_program_prediction_data.csv')
    
    ccols, ncols = [], []

    ycols = ['InvType']

    for col in [ c for c in df.columns if c != 'InvType' ]:  

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
    xcols = ncols + ccols
    df[ccols] = df[ccols].apply(categories_to_integer,axis=0)
    df[ncols] = df[ncols].apply(zscore_normalize, axis=0)
    df[ycols] = df[ycols].apply(categories_to_integer, axis=0)
    
    print(df.head())

    mrmr = mRMR(10, df, xcols, ycols, metric_func=lambda x, y: mutual_information(x,y,[20,20]))
    mrmr = mRMR(10, df, xcols, ycols, metric_func=abs_corr_coef)
    mrmr.run()

    