
from featureval.selection import mRMR
from featureval.metrics import abs_corr_coef

if __name__ == '__main__' :
    import pandas as pd

    df = pd.DataFrame(
        [[1,1,1,1],[2,1,3,1],[3,1,2,1], [3,3,2,0], [2,1,3,0], [3,4,5,0]],
        columns=['x1', 'x2', 'x3', 'y']
    )
    xcols = ['x1', 'x2', 'x3',]
    ycols = ['y']
    mrmr = mRMR(2, df, xcols, ycols, metric_func=abs_corr_coef)
    mrmr.run()

    