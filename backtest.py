import pandas as pd
from sklearn.model_selection import GroupTimeSeriesSplit
from zgtools.data_tool import *
from xgboost import XGBClassifier, XGBRegressor
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
    group = df_test.groupby('ret_qcut')
    group.mean()[label_col].plot(kind='bar', title='mean');plt.show()
    group.std()[label_col].plot(kind='bar', title='std');plt.show()
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

# 构造收益率矩阵
def get_df_chg(df_test, label_col):
    return df_test[label_col].unstack()

# def get_real_ret(df_weight, df_chg, cost_rate=0.001):
#     # 构造换手矩阵
#     df_turnover = df_weight.diff()
#     df_turnover.fillna(0, inplace=True)
#     df_turnover = np.abs(df_turnover)
#
#     # 计算交易费率
#     df_fee = df_turnover * cost_rate
#
#     # 计算收益
#     df_ret = df_chg * df_weight
#     df_ret = df_ret - df_fee
#
#     # 计算净值
#     df_pnl = df_ret.sum(axis=1)
#     df_pnl = df_pnl.cumsum()
#
#     df_pnl.plot(figsize=(15, 8))
#     plt.show()
#
#     return df_pnl

def get_model(x_train, y_train):
    # 用XGBoost建模
    model = XGBClassifier(seed=1, tree_method='gpu_hist')
    model.fit(x_train, y_train)
    return model

def data_process(df_factor, label_col):
    train_date = '2020-04-30'
    df_train = df_factor[df_factor.index.get_level_values('date') < train_date]
    df_train['bin'] = None
    df_train.loc[df_train[label_col] <= df_train[label_col].quantile(0.3), 'bin'] = 0
    df_train.loc[df_train[label_col] >= df_train[label_col].quantile(0.7), 'bin'] = 1
    df_train.dropna(inplace=True)
    df_test = df_factor[(df_factor.index.get_level_values('date') >= train_date)]
                        #& (df_factor['time'] == '10:00:00')]
    no_x_cols = ['next_open2open', 'bin', 'ret_60', 'ret_30', 'ret_15', 'ts', 'date', 'time']
    factor_cols = df_train.columns.difference(no_x_cols)
    y_col = 'bin'
    df_train, scaler = data_scale(df_train, factor_cols=factor_cols)
    #df_val, _ = data_scale(df_val, factor_cols=factor_cols, scaler=scaler)
    df_test,_ = data_scale(df_test, factor_cols=factor_cols, scaler=scaler)
    x_train, y_train = get_x_y(df=df_train, no_x_cols=no_x_cols, label_col=y_col)
    #x_val, y_val = get_x_y(df=df_val, no_x_cols=no_x_cols, label_col=label_col)
    x_test, y_test = get_x_y(df=df_test, no_x_cols=no_x_cols, label_col=label_col)
    return x_train, y_train, x_test, y_test

# 读取数据
df_factor_1000 = pd.read_csv('df_factor_introday_0950-1000-mean.csv')
df_factor_1000['date'] = df_factor_1000['date'] + ' 10:00'
df_factor_1000.set_index(['date', 'code'], inplace=True)
x_train_1000, y_train_1000, x_test_1000, y_test_1000 = data_process(df_factor_1000, label_col='ret_60')
model_1000 = get_model(x_train_1000, y_train_1000)
df_test_1000 = y_test_1000.to_frame()
df_test_1000['up_proba'] = model_1000.predict_proba(x_test_1000)[:,1] # 只取预测为1的概率
df_weight_1000 = get_df_weight(df_test_1000, upper_quantile=0.9, lower_quantile=0.1)

df_factor_1100 = pd.read_csv('df_factor_introday_1050-1100-mean.csv')
df_factor_1100['date'] = df_factor_1100['date'] + ' 11:00'
df_factor_1100.set_index(['date', 'code'], inplace=True)
x_train_1100, y_train_1100, x_test_1100, y_test_1100 = data_process(df_factor_1100, label_col='ret_60')
model_1100 = get_model(x_train_1100, y_train_1100)
df_test_1100 = y_test_1100.to_frame()
df_test_1100['up_proba'] = model_1100.predict_proba(x_test_1100)[:,1] # 只取预测为1的概率
df_weight_1100 = get_df_weight(df_test_1100, upper_quantile=0.9, lower_quantile=0.1)

df_factor_1330 = pd.read_csv('df_factor_introday_1320-1330-mean.csv')
df_factor_1330['date'] = df_factor_1330['date'] + ' 13:30'
df_factor_1330.set_index(['date', 'code'], inplace=True)
x_train_1330, y_train_1330, x_test_1330, y_test_1330 = data_process(df_factor_1330, label_col='ret_60')
model_1330 = get_model(x_train_1330, y_train_1330)
df_test_1330 = y_test_1330.to_frame()
df_test_1330['up_proba'] = model_1330.predict_proba(x_test_1330)[:,1] # 只取预测为1的概率
df_weight_1330 = get_df_weight(df_test_1330, upper_quantile=0.9, lower_quantile=0.1)

df_factor_1430 = pd.read_csv('df_factor_introday_1420-1430-mean.csv')
df_factor_1430['date'] = df_factor_1430['date'] + ' 14:30'
df_factor_1430.set_index(['date', 'code'], inplace=True)
x_train_1430, y_train_1430, x_test_1430, y_test_1430 = data_process(df_factor_1430, label_col='ret_30')
model_1430 = get_model(x_train_1430, y_train_1430)
df_test_1430 = y_test_1430.to_frame()
df_test_1430['up_proba'] = model_1430.predict_proba(x_test_1430)[:,1] # 只取预测为1的概率
df_weight_1430 = get_df_weight(df_test_1430, upper_quantile=0.9, lower_quantile=0.1)

df_weight = pd.concat([df_weight_1000, df_weight_1100, df_weight_1330, df_weight_1430], axis=0)
df_weight.sort_index(inplace=True)
df_weight = df_weight/4

df_chg_1000 = get_df_chg(df_test_1000, label_col='ret_60')
df_chg_1100 = get_df_chg(df_test_1100, label_col='ret_60')
df_chg_1330 = get_df_chg(df_test_1330, label_col='ret_60')
df_chg_1430 = get_df_chg(df_test_1430, label_col='ret_30')
df_chg = pd.concat([df_chg_1000, df_chg_1330, df_chg_1100, df_chg_1430], axis=0)
df_chg.sort_index(inplace=True)

# 构造换手矩阵
df_turnover = df_weight.diff()
df_turnover.fillna(0, inplace=True)
df_turnover = np.abs(df_turnover)

# 计算交易费率
df_fee = df_turnover * 0.001

# 计算收益
df_ret = df_chg * df_weight
#df_ret = df_ret - df_fee

# 计算净值
df_pnl = df_ret.sum(axis=1)
df_pnl = df_pnl.cumsum()

df_pnl.plot(figsize=(15, 8))
plt.show()

#df_pnl = get_real_ret(df_weight, df_chg)

# pred = model.predict(x_test)
# pred_proba = model.predict_proba(x_test)[:,1] # 只取预测为1的概率
# df_test['up_proba'] = pred_proba
#
# df_ret = get_qcut_ret(df_test) # 获取分位数收益
# df_ret = get_ret_no_cost(df_test, upper_quantile=0.9, lower_quantile=0.1) # 获取无成本收益
#df_pnl = get_real_ret(df_test, upper_quantile=0.9, lower_quantile=0.1)  # 获取真实收益

# df_test['buy_flag'] = pred
# df_buy = df_test['buy_flag'].to_frame().unstack()['buy_flag']
# df_buy.columns = map(lambda x:str(x).zfill(6), df_buy.columns)
# df_buy.columns = map(lambda x:x+'.SH' if x[0]=='6' else x+'.SZ', df_buy.columns)
# df_buy.to_csv('buy_flag.csv')

# positive = y_test[y_test>0].index
# negative = y_test[y_test<=0].index
# y_test[positive] = 1
# y_test[negative] = 0
#
# cm = confusion_matrix(y_test.values, list(pred))
# tn, fp, fn, tp = cm.ravel()
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = tp/(tp+fn)
# # Specificity or true negative rate
# TNR = tn/(tn+fp)
# # Precision or positive predictive value
# PPV = tp/(tp+fp)
# # Negative predictive value
# NPV = tn/(tn+fn)
# # Fall out or false positive rate
# FPR = fp/(fp+tn)
# # False negative rate
# FNR = fn/(tp+fn)
# # False discovery rate
# FDR = fp/(tp+fp)
#
# df_score = pd.DataFrame(
#     {'importance': list(model.feature_importances_), 'name': factor_cols}).astype(str)
#
# def factor_selection(df_corr, df_score, corr_threshold=0.7):
#     '''
#     :param df_corr: 相关系数df
#     :param df_score: 因子打分df，拥有name列和score列
#     :param corr_threshold: 相关系数筛选阈值
#     :return:包含有效列和有效列得分的dict
#     '''
#
#     # %% 因子筛选
#     drop_factor_list = []
#     df_corr = df_factor.corr()
#     df_corr = df_corr[df_corr>corr_threshold]
#     for factor in df_score.name:
#         if factor in drop_factor_list or factor not in df_corr.columns:
#             continue
#
#         df_corr_tmp = df_corr[factor]
#         drop_factor = df_corr_tmp.dropna().index.difference([factor]).tolist()
#         drop_factor_list += drop_factor
#     drop_factor_list = list(set(drop_factor_list))
#     rest_cols = list(set(df_score.name.unique()).difference(drop_factor_list))
#     df_score = df_score[df_score.name.isin(rest_cols)]
#     dict_ = {'rest_cols':rest_cols, 'score':df_score, 'drop_cols':drop_factor_list}
#     return dict_
#
# dict = factor_selection(df_corr=x_train.corr(), df_score=df_score)
# use_cols = dict['rest_cols']
#
# model.fit(x_train[use_cols], y_train)
# pred = model.predict(x_test[use_cols])
# cm = confusion_matrix(y_test.values, list(pred))
# tn, fp, fn, tp = cm.ravel()
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = tp/(tp+fn)
# # Specificity or true negative rate
# TNR = tn/(tn+fp)
# # Precision or positive predictive value
# PPV = tp/(tp+fp)
# # Negative predictive value
# NPV = tn/(tn+fn)
# # Fall out or false positive rate
# FPR = fp/(fp+tn)
# # False negative rate
# FNR = fn/(tp+fn)
# # False discovery rate
# FDR = fp/(tp+fp)