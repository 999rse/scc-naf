import pandas as pd
import torch
import os
import re
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from naf.forests import ForestKind, TaskType
from naf.naf_model import NeuralAttentionForest, NAFParams
from sklearn.metrics import r2_score, mean_squared_error


class SKPredModel:
    compiled_regexps = None
    tdf2predict = None
    loaded_model = None
    feature_names = None
    # train_df = None

    def __init__(self, model_path):
        """
        Here you initialize your model
        """
        regexp_time_list = [
            r'(?P<days>.+?(?=-))-(?P<hours>.+?(?=:)):(?P<minutes>.+?(?=:)):(?P<seconds>\d+)',
            r'(?P<hours>.+?(?=:)):(?P<minutes>.+?(?=:)):(?P<seconds>\d+)'
        ]
        self.compiled_regexps = [re.compile(regexp) for regexp in regexp_time_list]
        self.tdf2predict = pd.DataFrame()
        self.loaded_model = torch.load(model_path)
        # self.train_df = pd.DataFrame()

    
    def convert_time_to_seconds(self, element):
        for rcompile in self.compiled_regexps:
            rsearch = rcompile.search(element)
            if rsearch:
                try:
                    return (int(rsearch.group('days')) * 24 + int(rsearch.group('hours'))) * 3600 + int(rsearch.group('minutes')) * 60 + int(rsearch.group('seconds'))
                except:
                    return int(rsearch.group('hours')) * 3600 + int(rsearch.group('minutes')) * 60 + int(rsearch.group('seconds'))
    
    @property
    def model_keys(self) -> list():
        return ((self.train_df).columns).to_list

    def prepare_df(self, test_df: pd.DataFrame, train_df: pd.DataFrame()) -> pd.DataFrame:
        """
        Here you put any feature generation algorithms that you use in your model

        :param 
            test_df - датафрейм, который необходимо обработать
            train_df - датафрейм, с которого надо получить данные
        :return: test_df extended with generated features
        """
        
        train_df.dropna(inplace=True)
        try:
            train_df.drop(columns=['Start', 'JobName'], inplace=True)
            test_df.drop(columns=['Start', 'JobName'], inplace=True)
        except:
            pass
        train_df['Timelimit'] = train_df['Timelimit'].apply(self.convert_time_to_seconds)
        train_df['Elapsed'] = train_df['Elapsed'].apply(self.convert_time_to_seconds)
        train_df['Area'].replace({'geophys': 0,
            'radiophys': 1,
            'phys': 2,
            'bioinf': 4,
            'mach': 5,
            'biophys': 6,
            'it': 7,
            'mech': 8,
            'energ': 9,
            'astrophys': 10}, inplace=True)
        train_df['Area'].astype(int)

        train_df['Partition'].replace({'tornado': 0,
            'g2': 1,
            'cascade': 2,
            'tornado-k40': 3,
            'nv': 4}, inplace=True)
        train_df['Partition'].astype(int)
        train_df['State'].replace({
            'COMPLETED': 0,
            'FAILED': 1,
            'TIMEOUT': 2,
            'NODE_FAIL': 3,
            'OUT_OF_MEMORY': 4
        }, inplace=True)
        train_df['State'].replace(r'(CANCELLED.+)|(CANCELLED)', 5, regex=True, inplace=True)
        train_df['State'].astype(int)
        train_df['ExitCode'].replace(r':','', regex=True, inplace=True)
        train_df.reset_index(inplace=True, drop=True)

        tc = pd.DataFrame(train_df[train_df['State'] == 5].groupby('UID').count()['State'])
        ts = pd.DataFrame(train_df[train_df['State'] == 0].groupby('UID').count()['State'])
        tf = pd.DataFrame(train_df[train_df['State'] == 1].groupby('UID').count()['State'])
        tt = pd.DataFrame(train_df[train_df['State'] == 2].groupby('UID').count()['State'])

        tc.columns = ['task_cancel']
        ts.columns = ['task_success']
        tf.columns = ['task_fail']
        tt.columns = ['task_timeout']

        tc.reset_index(inplace=True)
        ts.reset_index(inplace=True)
        tf.reset_index(inplace=True)
        tt.reset_index(inplace=True)

        task_type = tc.merge(ts, on='UID', how='left')
        task_type = task_type.merge(tf, on='UID', how='left')
        task_type = task_type.merge(tt, on='UID', how='left')
        task_type.fillna({'task_cancel': 0, 'task_success': 0, 'task_fail': 0, 'task_timeout': 0}, inplace=True)
        # доля успешно выполненных задач к провальным помноженная на разность между временем заявленным и конечным в успешных задачах
        div_time_success = pd.DataFrame(train_df[train_df['State'] == 0].groupby('UID').mean()['Timelimit'] - train_df[train_df['State'] == 0].groupby('UID').median()['Elapsed'])
        div_time_success.columns = ['div_time_success']
        div_time_success.reset_index(inplace=True)
        task_type['ts2tf'] = task_type['task_success'] / (task_type['task_success'] + task_type['task_fail']) * div_time_success['div_time_success']

        # доля успешных задач ко всем задачам
        task_type['ts2all'] = task_type['task_success'] / (task_type['task_success'] + task_type['task_fail'] + task_type['task_cancel'] + task_type['task_timeout'])

        train_df = train_df.merge(task_type, on='UID', how='left')
        # Медианное значение целевой переменной
        median_elapsed = pd.DataFrame(train_df[train_df['State'] == 0].groupby('UID').median()['Elapsed'])
        median_elapsed.columns = ['median_elapsed']
        median_elapsed.reset_index(inplace=True)
        train_df = train_df.merge(median_elapsed, on='UID', how='left')
        # степень доверия
        trust_degree = pd.DataFrame(train_df[train_df['State'] == 0].groupby('UID').mean()['Elapsed'] / train_df[train_df['State'] == 0].groupby('UID').mean()['Timelimit'])
        trust_degree.columns = ['trust_degree']
        trust_degree.reset_index(inplace=True)
        train_df = train_df.merge(trust_degree, on='UID', how='left')
        # Submit
        train_df['Submit'] = pd.to_datetime(train_df['Submit'])
        train_df['Month'] = train_df['Submit'].dt.month

        train_df.drop(columns=['Submit'], inplace=True)
        # Процентиль 20 и 90
        percentile_20_elapsed = pd.DataFrame(train_df[train_df['State'] == 0].groupby('UID').quantile(.20)['Elapsed'])
        percentile_90_elapsed = pd.DataFrame(train_df[train_df['State'] == 0].groupby('UID').quantile(.90)['Elapsed'])

        percentile_20_elapsed.columns = ['percentile_20_elapsed']
        percentile_90_elapsed.columns = ['percentile_90_elapsed']
        percentile_20_elapsed.reset_index(inplace=True)
        percentile_90_elapsed.reset_index(inplace=True)

        perec_2090 = percentile_20_elapsed.merge(percentile_90_elapsed, on='UID', how='left')
        train_df = train_df.merge(perec_2090, on='UID', how='left')

        test_df['Area'].fillna(0, inplace=True)
        test_df = test_df.merge(task_type, on='UID', how='left')
        test_df = test_df.merge(median_elapsed, on='UID', how='left')
        test_df = test_df.merge(trust_degree, on='UID', how='left')
        test_df['Submit'] = pd.to_datetime(test_df['Submit'])
        test_df['Month'] = test_df['Submit'].dt.month
        test_df.drop(columns=['Submit'], inplace=True)
        test_df = test_df.merge(perec_2090, on='UID', how='left')

        train_df['trust_degree'].fillna(0, inplace=True)
        train_df.fillna(-1, inplace=True)

        test_df['trust_degree'].fillna(0, inplace=True)
        test_df.fillna(-1, inplace=True)

        test_df['Timelimit'] = test_df['Timelimit'].apply(self.convert_time_to_seconds)
        test_df['Area'].replace({'geophys': 0,
            'radiophys': 1,
            'phys': 2,
            'bioinf': 4,
            'mach': 5,
            'biophys': 6,
            'it': 7,
            'mech': 8,
            'energ': 9,
            'astrophys': 10}, inplace=True)
        test_df['Area'].astype(int)

        test_df['Partition'].replace({'tornado': 0,
            'g2': 1,
            'cascade': 2,
            'tornado-k40': 3,
            'nv': 4}, inplace=True)
        test_df['Partition'].astype(int)
        test_df['State'].replace({
            'COMPLETED': 0,
            'FAILED': 1,
            'TIMEOUT': 2,
            'NODE_FAIL': 3,
            'OUT_OF_MEMORY': 4
        }, inplace=True)
        test_df['State'].replace(r'(CANCELLED.+)|(CANCELLED)', 5, regex=True, inplace=True)
        test_df['State'].astype(int)
        test_df['ExitCode'].replace(r':','', regex=True, inplace=True)
        test_df.reset_index(inplace=True, drop=True)

        train_df = train_df.loc[:, ['UID', 'GID', 'Area', 'Partition', 'ReqNodes', 'ReqCPUS',
       'Timelimit', 'Priority', 'State', 'ExitCode', 'task_cancel',
       'task_success', 'task_fail', 'task_timeout', 'ts2tf', 'ts2all',
       'median_elapsed', 'trust_degree', 'Month', 'percentile_20_elapsed',
       'percentile_90_elapsed']]
       
        self.train_df = train_df

        self.tdf2predict = test_df
        self.tdf2predict = self.tdf2predict.loc[:, ['UID', 'GID', 'Area', 'Partition', 'ReqNodes', 'ReqCPUS',
       'Timelimit', 'Priority', 'State', 'ExitCode', 'task_cancel',
       'task_success', 'task_fail', 'task_timeout', 'ts2tf', 'ts2all',
       'median_elapsed', 'trust_degree', 'Month', 'percentile_20_elapsed',
       'percentile_90_elapsed']]

        scaler = StandardScaler(copy=False)
        test_df = scaler.fit(train_df)
        self.tdf2predict = scaler.transform(self.tdf2predict)

        return self.tdf2predict



    def predict(self) -> pd.Series:
        """
        Here you implement inference for your model

        :param test_df: dataframe to predict
        :return: vector of estimated times in milliseconds
        """

        params = NAFParams(
            kind=ForestKind.RANDOM,
            task=TaskType.REGRESSION,
            mode='end_to_end',
            n_epochs=100,
            lr=0.01,
            lam=1.0,
            target_loss_weight=1.0,
            hidden_size=16,
            n_layers=1,
            forest=dict(
                n_estimators=16,
                min_samples_leaf=1
            ),
            random_state=22,
        )
        pred_test_df = self.loaded_model.predict(self.tdf2predict)

        return pd.Series(pred_test_df)
