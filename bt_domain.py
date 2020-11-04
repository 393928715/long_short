import pandas as pd
from sklearn.model_selection import GroupTimeSeriesSplit
from zgtools.data_tool import *
from xgboost import XGBClassifier, XGBRegressor
from lightgbm  import LGBMRegressor, LGBMClassifier, plot_importance
from sklearn.metrics import confusion_matrix

def get_qcut_ret(df_test, cut_num=10):
    seri_date = df_test.index.get_level_values('date').drop_duplicates()
    df_test['ret_qcut'] = pd.qcut(df_test['up_proba'], cut_num, duplicates='drop', labels=False)
    df_ret = pd.DataFrame()
    for i in range(10):
        date_list = []
        ret_list = []
        for date in seri_date:
            df_tmp = df_test[df_test.index.get_level_values('date') == date]
            df_buy = df_tmp[df_tmp.ret_qcut == i]
            ret = df_buy[label_col].sum().mean()
            date_list.append(date)
            ret_list.append(ret)
        df_ret[i] = ret_list

    df_ret['date'] = seri_date
    df_ret.set_index('date', inplace=True)
    df_ret = df_ret.cumsum()
    df_ret.plot();plt.show()
    group = df_test.groupby(level=0).apply(lambda x: x.groupby('ret_qcut').mean())
    group.groupby(level=1).mean()['ret_60'].plot(kind='bar');
    plt.show()

    # group.mean()[label_col].plot(kind='bar', title='mean');plt.show()
    # group.std()[label_col].plot(kind='bar', title='std');plt.show()
    return df_ret

def get_ret_no_cost(df_test, upper_quantile, lower_quantile):
    seri_date = df_test.index.get_level_values('date').drop_duplicates()
    date_list = []
    long_ret_list = []
    short_ret_list = []
    for date in seri_date:
        df_tmp = df_test[df_test.index.get_level_values('date') == date]

        if isinstance(upper_quantile, float):
            df_buy = df_tmp[df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(upper_quantile)]
        else:
            df_buy = df_tmp[(df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(upper_quantile[0])) &
                            (df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(upper_quantile[1]))]

        if isinstance(lower_quantile, float):
            df_sell = df_tmp[df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(lower_quantile)]
        else:
            df_sell = df_tmp[(df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(lower_quantile[1])) &
                             (df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(lower_quantile[0]))]



        buy_return = df_buy[label_col].sum() / len(df_buy)

        sell_return = df_sell[label_col].sum() / len(df_sell)

        ret = (buy_return - sell_return) / 2
        date_list.append(date)
        long_ret_list.append(buy_return)
        short_ret_list.append(sell_return)

    df_ret = pd.DataFrame({'date': date_list, 'long_ret': long_ret_list, 'short_ret': short_ret_list})
    df_ret.set_index('date', inplace=True)

    df_ret['long_ret'] = df_ret['long_ret'].cumsum()
    df_ret['short_ret'] = df_ret['short_ret'].cumsum()
    df_ret['ret'] = df_ret['long_ret'] - df_ret['short_ret']
    df_ret['ret'] = df_ret['ret'] / 2
    df_ret[['long_ret', 'short_ret']].plot();
    plt.show()
    df_ret['ret'].plot();
    plt.show()
    return df_ret

def get_real_ret(df_test, upper_quantile, lower_quantile, direction=0, reverse=None):
    '''
    :param df_test: 包含收益和交易方向的dataframe
    :param upper_quantile: 买入的分位数
    :param lower_quantile: 卖出的分位数
    :param direction: 交易方向设置，-1：只做空，0：多空对冲，1：只做多
    :param reverse: 反转设置，-1：空头反转，1：多头反转
    :return:
    '''
    seri_date = df_test.index.get_level_values('date').drop_duplicates()
    # 构造买卖方向矩阵
    df_order_flag = df_test.copy()
    df_order_flag['order_flag'] = None
    for date in seri_date:
        df_tmp = df_test[df_test.index.get_level_values('date') == date]

        if isinstance(upper_quantile, float):
            df_buy = df_tmp[df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(upper_quantile)]
        elif isinstance(upper_quantile, int):
            df_buy = df_tmp[df_tmp['ret_qcut'] >= upper_quantile]
        else:
            if isinstance(upper_quantile[0], float):
                df_buy = df_tmp[(df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(upper_quantile[0])) &
                                (df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(upper_quantile[1]))]
            elif isinstance(upper_quantile[0], int):
                df_buy = df_tmp[(df_tmp['ret_qcut'] <= upper_quantile[0]) &
                                (df_tmp['ret_qcut'] >= upper_quantile[1])]

        if isinstance(lower_quantile, float):
            df_sell = df_tmp[df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(lower_quantile)]
        elif isinstance(lower_quantile, int):
            df_sell = df_tmp[df_tmp['ret_qcut'] <= lower_quantile]
        else:
            if isinstance(lower_quantile[0], float):
                df_sell = df_tmp[(df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(lower_quantile[1])) &
                                 (df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(lower_quantile[0]))]
            elif isinstance(lower_quantile[0], int):
                df_sell = df_tmp[(df_tmp['ret_qcut'] >= lower_quantile[1]) &
                                 (df_tmp['ret_qcut'] <= lower_quantile[0])]

        # df_buy = df_tmp[df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(0.7)]
        # df_sell = df_tmp[df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(0.3)]

        if direction==0:
            df_order_flag.loc[df_buy.index, 'order_flag'] = 1
            df_order_flag.loc[df_sell.index, 'order_flag'] = -1
        elif direction==1:
            df_order_flag.loc[df_buy.index, 'order_flag'] = 1
            df_order_flag.loc[df_sell.index, 'order_flag'] = 0
        elif direction==-1:
            df_order_flag.loc[df_buy.index, 'order_flag'] = 0
            df_order_flag.loc[df_sell.index, 'order_flag'] = -1

        if reverse == 1:
            df_order_flag.loc[df_buy.index, 'order_flag'] *= -1
        elif reverse == -1:
            df_order_flag.loc[df_sell.index, 'order_flag'] *= -1

    df_order_flag['order_flag'].fillna(0, inplace=True)
    df_order_flag = df_order_flag['order_flag'].unstack()

    # 构造交易权重矩阵
    df_weight = df_order_flag.groupby(axis=0, level=0).apply(lambda x: x/(np.abs(x).sum().sum()))

    # 构造换手矩阵
    df_turnover = df_weight.diff()
    df_turnover.fillna(0, inplace=True)
    df_turnover = np.abs(df_turnover)

    # # 计算交易费率
    df_fee = df_turnover * 0.001

    # 构造收益率矩阵
    df_next_open2open = df_test[label_col].unstack()

    # 计算收益
    df_ret = df_next_open2open * df_weight
    # df_ret = df_ret - df_fee

    # 计算每日净值
    df_pnl = df_ret.sum(axis=1)
    #df_pnl = df_pnl - 0.002

    return df_pnl

def get_domain_data(df, index_domain):
    df = df[df.index.isin(index_domain)]
    return df

# 用XGBoost建模
def get_domain_model(x_train, y_train, seri_date_domain):
    model = XGBClassifier(seed=1, tree_method='gpu_hist')
    index_domain = x_train.index[x_train.index.get_level_values('date').isin(seri_date_domain)]
    x_train_domain = get_domain_data(x_train, index_domain)
    y_train_domain = get_domain_data(y_train, index_domain)
    model.fit(x_train_domain, y_train_domain)
    return model

def get_domain_pred(x_test, df_test, seri_date_domain, model):
    index_domain = df_test.index[df_test.index.get_level_values('date').isin(seri_date_domain)]
    df_test = get_domain_data(df_test, index_domain)
    x_test = get_domain_data(x_test, index_domain)
    pred_proba = model.predict_proba(x_test)[:, 1]  # 只取预测为1的概率
    df_test['up_proba'] = pred_proba
    return df_test

def data_process(df_factor, label_col, time=None):
    train_date = '2020-07-01'
    df_train = df_factor[df_factor.index.get_level_values('date') < train_date]
    df_train['bin'] = None
    df_train.loc[df_train[label_col] <= df_train[label_col].quantile(0.3), 'bin'] = 0
    df_train.loc[df_train[label_col] >= df_train[label_col].quantile(0.7), 'bin'] = 1
    df_train.dropna(inplace=True)
    if time:
        df_test = df_factor[(df_factor.index.get_level_values('date') >= train_date)
                            & (df_factor['time'] == time)]
    else:
        df_test = df_factor[(df_factor.index.get_level_values('date') >= train_date)]
    no_x_cols = ['next_open2open', 'bin', 'ret_60', 'ret_30', 'ret_15', 'ts', 'date', 'time', 'datetime']
    factor_cols = df_train.columns.difference(no_x_cols)
    y_col = 'bin'
    # df_train, scaler = data_scale(df_train, factor_cols=factor_cols)
    # #df_val, _ = data_scale(df_val, factor_cols=factor_cols, scaler=scaler)
    # df_test,_ = data_scale(df_test, factor_cols=factor_cols, scaler=scaler)
    x_train, y_train = get_x_y(df=df_train, no_x_cols=no_x_cols, label_col=y_col)
    #x_val, y_val = get_x_y(df=df_val, no_x_cols=no_x_cols, label_col=label_col)
    x_test, y_test = get_x_y(df=df_test, no_x_cols=no_x_cols, label_col=label_col)
    return x_train, y_train, x_test, y_test

def get_model(x_train, y_train):
    # 用XGBoost建模
    model = XGBClassifier(seed=1, tree_method='gpu_hist')
    model.fit(x_train, y_train)
    return model

def get_df_weight(df_test, upper_quantile, lower_quantile):
    seri_date = df_test.index.get_level_values('date').drop_duplicates()
    # 构造买卖方向矩阵
    df_order_flag = df_test.copy()
    df_order_flag['order_flag'] = None
    for date in seri_date:
        df_tmp = df_test[df_test.index.get_level_values('date') == date]

        if isinstance(upper_quantile, float):
            df_buy = df_tmp[df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(upper_quantile)]
        else:
            df_buy = df_tmp[(df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(upper_quantile[0])) &
                            (df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(upper_quantile[1]))]

        if isinstance(lower_quantile, float):
            df_sell = df_tmp[df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(lower_quantile)]
        else:
            df_sell = df_tmp[(df_tmp['up_proba'] >= df_tmp['up_proba'].quantile(lower_quantile[1])) &
                             (df_tmp['up_proba'] <= df_tmp['up_proba'].quantile(lower_quantile[0]))]

        df_order_flag.loc[df_buy.index, 'order_flag'] = 1
        df_order_flag.loc[df_sell.index, 'order_flag'] = -1

    df_order_flag['order_flag'].fillna(0, inplace=True)
    df_order_flag = df_order_flag['order_flag'].unstack()

    # 构造交易权重矩阵
    df_weight = df_order_flag.groupby(axis=0, level=0).apply(lambda x: x/(np.abs(x).sum().sum()))

    return df_weight

# 分域数据
domain_bin_col = 'bin'
df_domain = pd.read_csv('D:\work\GTJA\section\data\k-means_test_cy50.csv')
df_domain.columns = ['trade_date'] + df_domain.columns[1:].tolist()
df_domain['trade_date'] = df_domain['trade_date'].str.replace('-','')

#df_domain = pd.read_csv('ma_score.csv')
df_domain['trade_date'] = pd.to_datetime(df_domain['trade_date'], format='%Y%m%d').astype(str)
seri_date_domain1 = df_domain[df_domain[domain_bin_col]==0]['trade_date']
seri_date_domain2 = df_domain[df_domain[domain_bin_col]==1]['trade_date']
seri_date_domain3 = df_domain[df_domain[domain_bin_col]==2]['trade_date']
seri_date_domain4 = df_domain[df_domain[domain_bin_col]==3]['trade_date']

label_col = 'ret_60'
def strategy_cy50_kmeans():
    '''
    10:00 s0,s3 做空
    14:30 s1,s2 做多
    '''
    df_factor = pd.read_csv('data/df_factor_introday_0955-1005_20201030.csv', index_col=['date', 'code'])
    x_train, y_train, x_test, y_test = data_process(df_factor, label_col='ret_60', time='10:00:00')
    model_1000 = get_model(x_train, y_train)
    df_test_1000 = y_test.to_frame()
    df_test_1000['up_proba'] = model_1000.predict_proba(x_test)[:,1] # 只取预测为1的概率

    # 10:00在分域0、3上空
    df_test1_1000 = df_test_1000[df_test_1000.index.get_level_values('date').isin(seri_date_domain1)]
    df_pnl_1000_1 = get_real_ret(df_test1_1000, upper_quantile=0.9, lower_quantile=0.1,direction=-1)  # 获取真实收益

    df_test4_1000 = df_test_1000[df_test_1000.index.get_level_values('date').isin(seri_date_domain4)]
    df_pnl_1000_4 = get_real_ret(df_test4_1000, upper_quantile=0.9, lower_quantile=0.1,direction=-1)  # 获取真实收益

    # 14:30 在分域1、2上多
    df_factor = pd.read_csv('data/df_factor_introday_1425-1435_20201030.csv', index_col=['date', 'code'])
    x_train, y_train, x_test, y_test = data_process(df_factor, label_col='ret_60', time='14:30:00')
    model_1430 = get_model(x_train, y_train)
    df_test_1430 = y_test.to_frame()
    df_test_1430['up_proba'] = model_1430.predict_proba(x_test)[:,1] # 只取预测为1的概率


    # 在分域1做多
    df_test2_1430 = df_test_1430[df_test_1430.index.get_level_values('date').isin(seri_date_domain2)]
    df_pnl_1430_2 = get_real_ret(df_test2_1430, upper_quantile=0.9, lower_quantile=0.1,direction=1)

    # 在分域4做多
    df_test3_1430 = df_test_1430[df_test_1430.index.get_level_values('date').isin(seri_date_domain3)]
    df_pnl_1430_3 = get_real_ret(df_test3_1430, upper_quantile=0.7, lower_quantile=0.1,direction=1)  # 获取真实收益


    df_pnl = pd.concat([df_pnl_1000_1, df_pnl_1000_4, df_pnl_1430_2, df_pnl_1430_3], axis=0)
    df_pnl.sort_index(inplace=True)

    df_domain.set_index('trade_date', inplace=True)

    df_pnl_cla=pd.concat([df_pnl, df_domain['bin']], axis=1)
    df_pnl_cla.loc[df_pnl_cla.bin.isin([1,2]), 'cla']=1
    df_pnl_cla.loc[df_pnl_cla.bin.isin([0,3]), 'cla']=0
    df_pnl_cla.groupby('cla').mean()[0].plot(kind='bar');plt.show()

    df_pnl = df_pnl.reindex(x_test.index.get_level_values('date').drop_duplicates(), fill_value=0)
    # df_pnl = df_pnl - 0.002
    df_pnl = df_pnl.cumsum()
    df_pnl.plot()
    plt.show()
    return df_pnl

strategy_cy50_kmeans()



# 读取数据
# df_factor = pd.read_csv('df_factor_introday_1425-1435.csv', index_col=['date', 'code'])
# x_train, y_train, x_test, y_test = data_process(df_factor, label_col='ret_60', time='14:30:00')
# model_1430 = get_model(x_train, y_train)
# df_test = y_test.to_frame()
# df_test['up_proba'] = model_1430.predict_proba(x_test)[:,1]    # 只取预测为1的概率

'''
统计k-means新分域表现
'''
# df_factor = pd.read_csv('data/df_factor_introday_0955-1005_20201030.csv', index_col=['date', 'code'])
# x_train, y_train, x_test, y_test = data_process(df_factor, label_col='ret_60', time='10:00:00')
# model_1000 = get_model(x_train, y_train)
# df_test_1000 = y_test.to_frame()
# df_test_1000['up_proba'] = model_1000.predict_proba(x_test)[:,1] # 只取预测为1的概率

# 分域1表现
# df_test1_1000 = df_test_1000[df_test_1000.index.get_level_values('date').isin(seri_date_domain1)]
# get_qcut_ret(df_test1_1000)
#
# # 分域2表现
# df_test2_1000 = df_test_1000[df_test_1000.index.get_level_values('date').isin(seri_date_domain2)]
# get_qcut_ret(df_test2_1000)
#
# # 分域3表现
# df_test3_1000 = df_test_1000[df_test_1000.index.get_level_values('date').isin(seri_date_domain3)]
# get_qcut_ret(df_test3_1000)
#
# df_test4_1000 = df_test_1000[df_test_1000.index.get_level_values('date').isin(seri_date_domain4)]
# get_qcut_ret(df_test4_1000)


# df_ret_1000 = get_ret_no_cost(df_test1_1000, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
#df_pnl_1000_1 = get_real_ret(df_test1_1000, upper_quantile=0.9, lower_quantile=0.1,direction=-1, reverse=-1)  # 获取真实收益

# df_ret = get_qcut_ret(df_test, 10)
# df_ret = get_ret_no_cost(df_test, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
# df_pnl = get_real_ret(df_test, upper_quantile=0.9, lower_quantile=0.1)  # 获取真实收益

# df_test4_1430 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain4)]
# df_ret_1430 = get_ret_no_cost(df_test4_1430, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
# df_pnl_1430 = get_real_ret(df_test4_1430, upper_quantile=0.9, lower_quantile=0.1,direction=1)  # 获取真实收益

# df_test1 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain1)]
# df_test2 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain2)]
# df_test3 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain3)]
# df_test4 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain4)]
#
# get_qcut_ret(df_test4)

# '''
# 10:00和14:30在分域4做多
# '''
# df_factor = pd.read_csv('df_factor_introday_0955-1005.csv', index_col=['date', 'code'])
# x_train, y_train, x_test, y_test = data_process(df_factor, label_col='ret_60', time='10:00:00')
# model_1000 = get_model(x_train, y_train)
# df_test_1000 = y_test.to_frame()
# df_test_1000['up_proba'] = model_1000.predict_proba(x_test)[:,1] # 只取预测为1的概率
#
# # 在分域1上空头反转
# df_test1_1000 = df_test_1000[df_test_1000.index.get_level_values('date').isin(seri_date_domain1)]
# get_qcut_ret(df_test1_1000)
# # df_ret_1000 = get_ret_no_cost(df_test1_1000, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
# df_pnl_1000_1 = get_real_ret(df_test1_1000, upper_quantile=0.9, lower_quantile=0.1,direction=-1, reverse=-1)  # 获取真实收益
#
# # 在分域4上做多
# df_test4_1000 = df_test_1000[df_test_1000.index.get_level_values('date').isin(seri_date_domain4)]
# get_qcut_ret(df_test4_1000)
# #df_ret_1000 = get_ret_no_cost(df_test4_1000, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
# df_pnl_1000_2 = get_real_ret(df_test4_1000, upper_quantile=0.9, lower_quantile=0.1,direction=1)  # 获取真实收益
#
# df_factor = pd.read_csv('df_factor_introday_1425-1435.csv', index_col=['date', 'code'])
# x_train, y_train, x_test, y_test = data_process(df_factor, label_col='ret_60', time='14:30:00')
# model_1430 = get_model(x_train, y_train)
# df_test_1430 = y_test.to_frame()
# df_test_1430['up_proba'] = model_1430.predict_proba(x_test)[:,1] # 只取预测为1的概率
#
# #get_qcut_ret(df_test_1430)
#
# # 在分域1做空头反转
# df_test1_1430 = df_test_1430[df_test_1430.index.get_level_values('date').isin(seri_date_domain1)]
# # df_ret_1430 = get_ret_no_cost(df_test4_1430, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
# df_pnl_1430_1 = get_real_ret(df_test1_1430, upper_quantile=0.9, lower_quantile=0.1,direction=-1, reverse=-1)
# # get_qcut_ret(df_test1_1430)
#
# # 在分域4做多
# df_test4_1430 = df_test_1430[df_test_1430.index.get_level_values('date').isin(seri_date_domain4)]
# #get_qcut_ret(df_test4_1430)
# # df_ret_1430 = get_ret_no_cost(df_test4_1430, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
# df_pnl_1430_2 = get_real_ret(df_test4_1430, upper_quantile=0.7, lower_quantile=0.1,direction=1)  # 获取真实收益

# df_pnl = pd.concat([df_pnl_1000_1, df_pnl_1000_4, df_pnl_1430_2, df_pnl_1430_3], axis=0)
# df_pnl.sort_index(inplace=True)
#
# df_domain.set_index('trade_date', inplace=True)
#
# df_pnl_cla=pd.concat([df_pnl, df_domain['bin']], axis=1)
# df_pnl_cla.loc[df_pnl_cla.bin.isin([1,2]), 'cla']=1
# df_pnl_cla.loc[df_pnl_cla.bin.isin([0,3]), 'cla']=0
# df_pnl_cla.groupby('cla').mean()[0].plot(kind='bar');plt.show()
#
# df_pnl = df_pnl.reindex(x_test.index.get_level_values('date').drop_duplicates(), fill_value=0)
# # df_pnl = df_pnl - 0.002
# df_pnl = df_pnl.cumsum()
# df_pnl.plot()
# plt.show()

# import empyrical
# yieldRate = empyrical.annual_return(df_pnl, period='daily', annualization=None)
# max_draw_down = empyrical.max_drawdown(df_pnl)
# calmar = empyrical.calmar_ratio(df_pnl, period='daily', annualization=None)
# sharpe = empyrical.sharpe_ratio(df_pnl, risk_free=0, period='daily', annualization=None)

# df_pnl_1430 = df_pnl_1430.reindex(x_test.index.get_level_values('date').drop
# _duplicates(), method='ffill')
# df_pnl_1430.plot()
# plt.show()
#
# df_pnl_1000 = df_pnl_1000.reindex(x_test.index.get_level_values('date').drop_duplicates(), method='ffill')
# df_pnl_1000.plot()
# plt.show()

# df_factor = pd.read_csv('df_factor_introday_1425-1435.csv', index_col=['date', 'code'])
# x_train, y_train, x_test, y_test = data_process(df_factor, label_col='ret_60')
# model_1000 = get_model(x_train, y_train)
# df_train, scaler = data_scale(df_train, factor_cols=factor_cols)
# #df_val, _ = data_scale(df_val, factor_cols=factor_cols, scaler=scaler)
# df_test,_ = data_scale(df_test, factor_cols=factor_cols, scaler=scaler)

# 用XGBoost建模
# model = XGBClassifier(seed=1, tree_method='gpu_hist')
#
# model.fit(x_train, y_train.astype(int))
# df_test['up_proba'] = model.predict_proba(x_test)[:,1]

# get_qcut_ret(df_test)
# get_ret_no_cost(df_test, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
# get_real_ret(df_test, upper_quantile=0.9, lower_quantile=0.1)  # 获取真实收益

# df_test['up_proba'] = pred_proba
#
# df_test1 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain1)]
# df_test2 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain2)]
# df_test3 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain3)]
# df_test4 = df_test[df_test.index.get_level_values('date').isin(seri_date_domain4)]

# get_qcut_ret(df_test1)

# model1 = get_domain_model(x_train, y_train, seri_date_domain1)
# model2 = get_domain_model(x_train, y_train, seri_date_domain2)
# model3 = get_domain_model(x_train, y_train, seri_date_domain3)
# model4 = get_domain_model(x_train, y_train, seri_date_domain4)
#
# pred_proba1 = get_domain_pred(x_test, df_test, seri_date_domain1, model1)
# pred_proba2 = get_domain_pred(x_test, df_test, seri_date_domain2, model2)
# pred_proba3 = get_domain_pred(x_test, df_test, seri_date_domain3, model3)
# pred_proba4 = get_domain_pred(x_test, df_test, seri_date_domain4, model4)
#
# df_test = pd.concat([pred_proba1, pred_proba2, pred_proba3, pred_proba4], axis=0)
# df_test.sort_index(inplace=True)

# df_ret = get_qcut_ret(df_test) # 获取分位数收益
# df_ret = get_ret_no_cost(df_test4, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
# df_pnl = get_real_ret(df_test4, upper_quantile=0.9, lower_quantile=0.1)  # 获取真实收益
# #
# # 对不同分域做统计
# get_qcut_ret(pred_proba4)