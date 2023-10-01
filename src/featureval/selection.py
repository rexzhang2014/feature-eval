
from featureval.redrel import Redundancy, Relevance
from featureval.metrics import abs_corr_coef

class SelectionStrategy() :
    def __init__(self, data, xcols, ycols, init_set=[], rel_class=Relevance, red_class=Redundancy, metric_func=None, *args, **kwargs) :
        self.data = data
        self.xcols = xcols
        self.ycols = ycols
        self.selected = init_set
        self.metric_func = metric_func
        self.rel_class = rel_class # (data, self.xcols, self.ycols, metric_func)
        self.red_class = red_class # (data, self.xcols, self.ycols, metric_func)
    
    def _run(self) : 
        pass
    
    def run(self) :
        
        return self._run()
    
class mRMR(SelectionStrategy) :
    def __init__(self, n_features=-1, *args, **kwargs) :
        self.n_features = n_features
        super().__init__(*args, **kwargs)
        
    def _run(self) :
        relevance = self.rel_class(self.data, self.xcols, self.ycols, self.metric_func)
        rel_rlt = relevance.calculate().sort_values(relevance.name, ascending=False)
    
        records = rel_rlt.to_records(index=None)
        print(records)
        
        self.selected = []
        
        for i in range(0, self.n_features) :
            print(f'iteration:{i}')
            f_rec = records[i]
            
            if i == 0 :
                self.selected.append(f_rec[0])
                continue
                
            # At iteration i, the x is the selected feature, y is all the other features out of x.
            candidates = [ f for f in self.xcols if f not in self.selected]
            print(self.selected, candidates)
            redundancy = self.red_class(self.data, self.selected, candidates, func=self.metric_func)
            red_rlt = redundancy.calculate()
            print(red_rlt)

            rel_vs_red = rel_rlt.merge(red_rlt, left_on='x', right_on='y') #- red_rlt.sort_values('y') #- red_rlt
            
            rel_vs_red['score'] = rel_vs_red[relevance.name] - rel_vs_red[redundancy.name]
            print(rel_vs_red)

            best = rel_vs_red.sort_values('score', ascending=False).head(1)
            self.selected.append(best['y'].values[0])
            print(self.selected)
        
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

    