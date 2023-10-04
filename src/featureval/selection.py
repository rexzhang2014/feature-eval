
from featureval.redrel import Redundancy, Relevance 
from featureval.metrics import mutual_information

# from redrel import Redundancy, Relevance
# from metrics import mutual_information, abs_corr_coef, info_value

class SelectionStrategy() :
    def __init__(self, data, xcols, ycols, init_set=[], 
                 rel_class=Relevance, red_class=Redundancy, metric_func=None, 
                 rel_func=None, red_func=None,
                 *args, **kwargs) :
        self.data = data
        self.xcols = xcols
        self.ycols = ycols
        self.selected = init_set

        if metric_func :
            self.rel_func = metric_func
            self.red_func = metric_func
        elif rel_func and red_func :
            self.rel_func = rel_func
            self.red_func = red_func
        else :
            raise ValueError(f"Invalid Arguments:{metric_func}")
        
        self.rel_class = rel_class # (data, self.xcols, self.ycols, metric_func)
        self.red_class = red_class # (data, self.xcols, self.ycols, metric_func)

        self.verbose = kwargs.get('verbose', 0)
    
    def _vprint(self, *args, **kwargs) :
        if self.verbose > 0 :
            print(*args, **kwargs)

    def _run(self) : 
        pass
    
    def run(self) :
        
        return self._run()
    
class mRMR(SelectionStrategy) :
    def __init__(self, n_features=-1, *args, **kwargs) :
        self.n_features = n_features
        super().__init__(*args, **kwargs)
        
    def _run(self) :
        relevance = self.rel_class(self.data, self.xcols, self.ycols, self.rel_func)
        rel_rlt = relevance.calculate().sort_values(relevance.name, ascending=False)
    
        records = rel_rlt.to_records(index=None)
        self._vprint(records)
        
        self.selected = []
        
        for i in range(0, self.n_features) :
            self._vprint(f'iteration:{i}')
            f_rec = records[i]
            
            # Select the top relevant feature at 1st iteration.
            if i == 0 :
                self.selected.append(f_rec[0])
                continue
                
            # At iteration i(i>0) the x is the selected feature, y is all the other features out of x.
            candidates = [ f for f in self.xcols if f not in self.selected]
            self._vprint(self.selected, candidates)
            redundancy = self.red_class(self.data, self.selected, candidates, func=self.red_func)
            red_rlt = redundancy.calculate()
            self._vprint(red_rlt.head())

            # Calculate the mean relevance if a candidate feature is selected.
            curr_rele_mean = rel_rlt.loc[rel_rlt['x'].isin(self.selected), relevance.name].mean()
            cand_rele = rel_rlt[rel_rlt['x'].isin(candidates)]
            cand_rele = cand_rele.assign(
                cand_rele_mean = (cand_rele[relevance.name] + len(self.selected)*curr_rele_mean) / (len(self.selected)+1)
            )

            # Find the best feature by mean_relevance - redundancy
            rel_vs_red = cand_rele.merge(red_rlt, left_on='x', right_on='y') #- red_rlt.sort_values('y') #- red_rlt
            
            rel_vs_red['score'] = rel_vs_red['cand_rele_mean'] - rel_vs_red[redundancy.name]
            self._vprint(rel_vs_red.head())

            best = rel_vs_red.sort_values('score', ascending=False).head(1)
            self.selected.append(best['y'].values[0])
            self._vprint(self.selected)

        return (self.selected, rel_vs_red,)
        
if __name__ == '__main__' :
    import pandas as pd

    df = pd.read_csv('C:/workspace/feature-eval/data/investing_program_prediction_data.csv')
    
    from preprocess import categories_to_integer, zscore_normalize, detect_variable_type, weight_of_evidence

    ycols = ['InvType']
    xcols = [ col for col in df.columns if col not in ycols]
    ccols, ncols = detect_variable_type(df, xcols)

    # convert categorical y into integer-based y
    df[ycols] = df[ycols].apply(categories_to_integer, axis=0)
    
    # preprocess x
    df[ccols] = df[ccols].apply(categories_to_integer,axis=0)
    df[ncols] = df[ncols].apply(zscore_normalize, axis=0)
    df[ncols] = df[ncols].apply(lambda x: weight_of_evidence(x, df[ycols].values.flatten()), axis=0)
    
    mrmr = mRMR(3, df, 
                xcols, ycols, 
                # metric_func=lambda x, y: mutual_information(x,y,[10,10]),
                rel_func=mutual_information,
                red_func=mutual_information,
                verbose=0)
    # mrmr = mRMR(3, df, xcols, ycols, metric_func=abs_corr_coef)
    selected, score = mrmr.run()

    print(selected)


    