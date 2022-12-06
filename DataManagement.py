import os
import pandas as pd
import dask.dataframe as dd
import numpy as np
from abc import ABC, abstractmethod


class DataManagement(ABC):
    def __init__(self):
        self.folder_path = r'C:\tmp\Banking Book Campaign'
        self.headers: list
        self.dtypes: dict
        self.dates_fields: list
        self.to_include: list
        self.df: None

    @staticmethod
    def generate_headers_dtypes(self, header_file_path = 'Variables_Info.csv'):
        self.headers_types = pd.read_csv(rf'{header_file_path}', sep=',').replace('\n', ' ', regex=True)
        self.headers = list(self.headers_types['Field'])
        self.dates_fields = list(self.headers_types[(self.headers_types['To Include'] == True) & (self.headers_types['From definition'] == 'DATE')]['Field'])
        self.to_include = list(self.headers_types[self.headers_types['To Include']==True]['Field'])
        self.dtypes = self.headers_types[['Field', 'Type']].set_index('Field').to_dict()['Type']

    @abstractmethod
    def read_csv_file(self):
        pass

    @abstractmethod
    def export_data(self):
        pass

    @abstractmethod
    def clean_data(self):
        pass


class PandasDataManagement(DataManagement):
    def read_csv_file(self, path='', filename='1m_random_sample') -> pd.DataFrame:
        return pd.read_csv(filename + '.csv', dtype=self.dtypes, header=0)

    def export_data(self,path='', filename='Data_cleaned') -> None:
        self.df.to_csv(path+filename+'.csv')

    def clean_data(self) -> pd.DataFrame:
        self.df = self.df[self.to_include]
        self.dropna(axis=1, how='all', inplace=True) # remove empty columns
        self.df.drop_duplicates(inplace=True) # remove duplicated rows
        return self.df

    def transform_pd(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transforming Current Loan Delinquency Status to a flag
        self.df['Current Loan Delinquency Status'].where(~(self.df['Current Loan Delinquency Status'] == 'XX'), other='100', inplace=True)
        self.df = self.df.astype({'Current Loan Delinquency Status':float}, copy=False)
        self. df['Loan Delinquency Defaulted'] = np.select([(3 < self.df['Current Loan Delinquency Status']) & (self.df['Current Loan Delinquency Status'] < 100),
                                                      self.df['Current Loan Delinquency Status'] <= 3,
                                                      self.df['Current Loan Delinquency Status'] == 100], [1, 0, 0])
        self.df = self.df.astype({'Loan Delinquency Defaulted': bool}, copy=False)

        # Transforming date fields format
        self.df[self.dates_fields] = pd.to_datetime(self.df[self.dates_fields].stack(), format='%m%Y').unstack()
        # df.to_csv(folder_path + '1m Random Sample (2000Y) - Cleaned v2.csv')
        return self.df


class DaskDataManagement(DataManagement):
    def read_csv_file(self, path=r'C:\tmp\Banking Book Campaign\2000') -> dd.DataFrame:
        return dd.read_csv(urlpath=os.path.join(path, "*.csv"), sep="|", header=None, blocksize=25e6, names=self.headers, dtype=self.dtypes)

    def export_data(self, path='', filename='Data_cleaned') -> None:
        pd.DataFrame(data=self.df, columns=self.headers).to_csv(path +filename+'.csv')

    def clean_data(self, ddf: dd.DataFrame) -> dd.DataFrame:
        self.df = self.df.drop(list(self.headers_types[self.headers_types ['Empty'] == True]['Field']), axis=1)  # remove empty columns
        self.df = self.df.drop_duplicates()  # Remove duplicated rows
        return self.df

    def random_rows(self, ddf: dd.DataFrame, n=1000005 / 36634659) -> dd.DataFrame:
        return self.df.sample(frac=n, replace=True, random_state=True)