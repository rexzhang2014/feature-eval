
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import product

class MetricBase() :
    def __init__(self, data:pd.DataFrame, xcols=[], ycols=[], *args, **kwargs) :
        self.data=data
        self.xcols=xcols
        self.ycols=ycols
        
    def calculate(self) :
        self.result = self._calc()
        return self.result
    
    def _calc(self) :
        pass
    
class Relevance(MetricBase):
    '''
    The redundency is defined as the relationship between one feature and the prediction target. In this version, only binary target is involved and only 0-1 values are supported.
    
    Some common used metrics are linear correlation, information value, ks stats, mutual information etc. 
    
    Parameters
    ----------
    data : pd.DataFrame Holding columns, each of a feature.
    
    xcols : list The feature set column names.
    
    ycols : list The target column names. Only 1 y-col is supported in this version.
    '''
    def __init__(self, data, xcols=[], ycols=[], func=None, *args, **kwargs) :
        self.relevance = pd.DataFrame(columns=['x'])
        self.relevance['x'] = xcols
        self.name = kwargs.get('name', self.__class__.__name__)
        
        self.relevance[self.name] = 0
        self._func_ = func
        super().__init__(data, xcols, ycols, *args, **kwargs)
    
    def _calc(self) :
        
        vals = []
        for x, y in product(self.xcols, self.ycols) :
            val = self._func_(self.data[x].values, self.data[y].values)
            vals.append(val)
            
        self.relevance[self.name] = vals
        return self.relevance
    
    
class Redundancy(MetricBase):
    '''
    The redundency is defined as the extent of overlap of the two objects.  
    
    The *object* can be individual feature or a set of features. 
    
    In feature selection tasks, the X usually is a set of features, representing the selected candidate feature set, whereas Y is a single feature and we evaluate if Y has high redundency with the X set and then we decide whether to keep the Y feature. 
    
    In more complicated cases, Y is also a feature set, then we would evaluate every y in Y to decide wheter to keep them.
    
    Some common used metrics are linear correlation, mutual information etc. 
    
    Parameters
    ----------
    data : pd.DataFrame Holding columns, each of a feature.
    
    xcols : list The feature set X.
    
    ycols : list The feature set Y.
    '''
    def __init__(self, data, xcols=[], ycols=[], func=None, *args, **kwargs) :
        
        # self.pairwise = pd.DataFrame(columns=['x', 'y'])
        
        xy = [ (x,y) for x, y in product(xcols, ycols)]
        self.pairwise = pd.DataFrame.from_records(xy, columns=['x', 'y'])
        # print(self.pairwise.head())
        self.name = kwargs.get('name', self.__class__.__name__)
        
        
        self.redundancy = pd.DataFrame()
        self.redundancy[self.name] = 0
        
        self._func_ = func
        
        self.xcols = xcols
        self.ycols = ycols
        
        super().__init__(data, xcols, ycols, *args, **kwargs)
        
    def _calc_pairwise(self) :
        
        vals = []
        for x, y in product(self.xcols, self.ycols) :
            val = self._func_(self.data[x].values, self.data[y].values)
            vals.append(val)
        
        # print(len(vals), self.pairwise.shape)
            
        self.pairwise[self.name] = vals
    
        return self.pairwise
    
    def _calc(self, agg_method='mean') :
        
        pairwise = self._calc_pairwise()
        
        if isinstance(agg_method, str) :
            agg_method = eval('np.'+agg_method)
            
        fsetwise = pairwise.groupby('y')[self.name].agg(agg_method)
        
        self.redundancy = fsetwise.reset_index()
        return self.redundancy

class PearsonCorrelation(Relevance) :

    def __init__(self, data, xcols=[], ycols=[], *args, **kwargs) :
        super().__init__(data, xcols, ycols, func=corr_coef,*args, **kwargs)
    

class InformationValue(Relevance) :

    def __init__(self, data, xcols=[], ycols=[], bins=5, *args, **kwargs) :
        
        super().__init__(data, xcols, ycols, func=info_value, *args, **kwargs)
    

# class MutualInformation(Relevance, Redundency) :
#     pass

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

