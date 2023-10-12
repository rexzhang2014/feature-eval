import pandas as pd
from redrel import MetricBase
from preprocess import get_woe_df
class BinningBase(MetricBase) :
    def fit(self) :
        pass
    def transform(self, new_data, *args, **kwargs) :
        pass


class WOEBin(BinningBase) :
    '''
    Re-order the value bins so to let the bins have monotonic metric value.
    Eg, the woe value increase along the bins increase.   
    '''
    def __init__(self, data, xcols=..., ycols=..., bins=10, init_bins=20, merge_thres=0.01, *args, **kwargs):
        self.bins = bins
        self.init_bins = init_bins
        self.merge_thres = merge_thres
        super().__init__(data, xcols, ycols, *args, **kwargs)

    def _calc(self) :
        # df = self.data[self.xcols].apply(lambda x: pd.cut(x, self.init_bins), axis=0)
        res_df = self.data[self.xcols].copy()

        for i, col in enumerate(self.xcols) :
            n_uniq_val = self.data[col].nunique()
            if n_uniq_val < self.bins :
                print(f'Skipped. {col} has less unique value than expected bins:{self.bins}.')
                continue

            x = self.data[col].values
            y = self.data[self.ycols].values.flatten()

            # Initial binning by any simple method. 
            # It is possible to split the data into small bins becase they are to be merged in next steps
            xbins = pd.qcut(x, self.init_bins, duplicates='drop')
            df = pd.DataFrame.from_dict({'x': x, 'y': y, 'xbins': xbins}, orient='columns')

            woe_df = get_woe_df(x, y, xbins).reset_index()
             
            woe_df['woe_bin'] = pd.qcut(woe_df['woe'], self.bins, duplicates='drop')
            
            woe_df = df[['x', 'xbins']].merge(woe_df, on='xbins', how='left')
            
            woe_bin = get_woe_df(x, y, woe_df['woe_bin']).reset_index()
            
            woe_bin = woe_df[['x', 'woe_bin']].merge(
                woe_bin, left_on='woe_bin', right_on='xbins', how='left'
            )

            woe_bin = woe_bin.rename(columns={'x': col})
            
            res_df[col+'_woe_bin'] = woe_bin['woe_bin']
            res_df[col+'_woe'] = woe_bin['woe']
            print(f"{'.'*i}", end='\r')
        self.result = res_df
        return res_df

if __name__ == '__main__' :
    
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
    # df[ncols] = df[ncols].apply(lambda x: weight_of_evidence(x, df[ycols].values.flatten()), axis=0)
    
    woebin = WOEBin(df, xcols, ycols, 5)
    # mrmr = mRMR(3, df, xcols, ycols, metric_func=abs_corr_coef)
    result = woebin.calculate()
    
    result.to_csv('woebin.csv', index=None)

    print(result)


    
