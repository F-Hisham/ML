import pandas as pd
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor


class Analysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.folder_path = r''

    def VIF(self):
        self.df.columns = self.df.columns.str.replace(' ', '')
        self.df.columns = self.df.columns.str.replace(r'[)(-]', '_', regex=True)
        Y, X = dmatrices('LoanDelinquencyDefaulted~OriginalInterestRate+OriginalUPB+CurrentActualUPB+OriginalLoanTerm+'
                         'NumberofUnits+OriginalLoantoValueRatio_LTV_+NumberofBorrowers+BorrowerCreditScoreatOrigination+'
                         'CurrentInterestRate+LoanAge+RemainingMonthstoLegalMaturity+RemainingMonthsToMaturity+Debt_To_Income_DTI_+'
                         'OriginalCombinedLoantoValueRatio_CLTV_+Co_BorrowerCreditScoreatOrigination+MortgageInsurancePercentage',
                         data=self.df, return_type='dataframe')
        # calculate VIF for each explanatory variable
        vif = pd.DataFrame()
        vif['variable'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif.to_csv(self.folder_path + 'vif_1m.csv')

    def generate_quantitative_statistics(self, name='sample_quantitative') -> None:
        df = self.df.select_dtypes(include='float64')
        df_len = len(df)
        stat = df.describe(percentiles=[0.05, 0.95])
        stat = stat.append(pd.Series(data=df.nunique(), name='nunique'), ignore_index=False)
        stat = stat.append(pd.Series(data=df.skew(axis=0), name='Skewness'), ignore_index=False)
        stat = stat.append(pd.Series(data=df.kurt(axis=0), name='Kurtosis'), ignore_index=False)
        stat = stat.append(pd.Series(data=df.median(axis=0), name='Median'), ignore_index=False)
        stat = stat.append(pd.Series(data=df_len - stat.loc['count'], name='n missing'), ignore_index=False)
        stat = stat.append(pd.Series(data=1 - (stat.loc['count'] / df_len), name='pc missing'), ignore_index=False)
        stat.transpose().to_csv(self.folder_path + name + '.csv')

    def generate_qualitative_statistics(self, name='sample_qualitative') -> None:
        dff = self.df.select_dtypes(include='object').astype('object')
        tmp = {}
        for key in dff.columns:
            tmp[key] = dff.groupby(key)[key].count()

        df = pd.DataFrame(columns=['Variable', 'Value', 'Frequency', 'Count', 'N_missing', 'Pct_missing'])
        for key in tmp.keys():
            if tmp[key].count() < 1000:
                df = df.append(pd.DataFrame(
                    {'Variable': key, 'Value': tmp[key].keys(), 'Frequency': tmp[key][:], 'Count': dff[key].count(),
                     'N_missing': len(dff) - dff[key].count(), 'Pct_missing': (len(dff) - dff[key].count()) / len(dff)}),
                               ignore_index=True)
            else:
                df = df.append(pd.DataFrame(
                    {'Variable': key, 'Value': 'skipped', 'Frequency': 'skipped', 'Count': dff[key].count(),
                     'N_missing': len(dff) - dff[key].count(), 'Pct_missing': (len(dff) - dff[key].count()) / len(dff)},
                    index=[0]), ignore_index=True)
        df.to_csv(self.folder_path + name + '.csv')

    def generate_data_summary(self) -> None:
        data_summary = pd.DataFrame(self.df.nunique(), columns={'Nunique'}).merge(self.df.count().rename('Count'), left_index=True, right_index=True)
        data_summary['n_Missing'] = len(self.df) - data_summary['Count']
        data_summary['pc_Missing'] = (len(self.df) - data_summary['Count']) / len(self.df)
        data_summary = data_summary.merge(self.df.dtypes.rename('Type'), left_index=True, right_index=True)
        data_summary.to_csv(self.folder_path + 'Data_Summary.csv')
