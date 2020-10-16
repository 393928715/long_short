import pandas as pd
import os
import datetime

root_dir = 'factors/factors'
flist = os.listdir(root_dir)
f = flist[0]
df_list = []
no_x_cols = ['ret_15', 'ret_30', 'ret_60', 'code']
for f in flist:
    fpath = os.path.join(root_dir, f)
    df = pd.read_csv(fpath, parse_dates=['ts'])
    code = f.split('.')[0]
    df['date'] = df.ts.dt.date
    df['time'] = df.ts.dt.time
    df = df[df['time'] > pd.Timestamp('14:30').time()] # 取每天最后30分钟的数据
    df = df[df.columns.difference(no_x_cols)].groupby('date').mean()
    df['code'] = code
    df_list.append(df)
    print(code)
df_factor = pd.concat(df_list, axis=0)
df_factor['code'] = df_factor['code'].apply(lambda x: x[-6:])
#df_factor.set_index('code', inplace=True, append=True)
df_factor.to_csv('df_factor_day.csv')

df_day = pd.read_csv(r'D:\work\GTJA\stk_mkt.csv', parse_dates=['date'])
df_day.code = df_day.code.apply(lambda x: x[:6])
df_day = df_day[df_day.code.isin(df_factor.code)]

df_day.set_index(['date', 'code'], inplace=True)
df_factor.set_index('code', inplace=True, append=True)

df_factor['next_open2open'] = df_day['pct_chg']

