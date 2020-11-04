import pandas as pd
import os
import datetime

root_dir = '../factors/factors'
root_dir = r'D:\work\GTJA\factor_data\cyb_factors_out_sample'
flist = os.listdir(root_dir)
f = flist[0]
df_list = []

label_col = 'ret_60'
no_x_cols = ['ret_15', 'ret_30', 'code', 'ret_60']#'ret_60',
for f in flist:
    fpath = os.path.join(root_dir, f)
    df = pd.read_csv(fpath, parse_dates=['datetime'])
    code = f.split('.')[0]
    df['date'] = df.datetime.dt.date
    df['time'] = df.datetime.dt.time
    # df = df[(df['time']==pd.Timestamp('09:31').time())|(df['time']==pd.Timestamp('09:45').time())
    #         |(df['time']==pd.Timestamp('10:00').time())|(df['time']==pd.Timestamp('10:15').time())
    #         |(df['time']==pd.Timestamp('10:30').time())]

    # df = df[(df['time']>=pd.Timestamp('09:31').time())&(df['time']==pd.Timestamp('09:45').time())]

    # df = df[(df['time'] == pd.Timestamp('10:00').time())]
    # seri_ret60 = df[df.time == pd.Timestamp('10:00').time()][label_col]
    df = df[(df['time']>pd.Timestamp('09:55').time())&(df['time']<=pd.Timestamp('10:05').time())]
    # df = df[df.columns.difference(no_x_cols)].groupby('date').mean()
    # df[label_col] = seri_ret60.tolist()

    df['code'] = code
    df_list.append(df)
    print(code)

df_factor = pd.concat(df_list, axis=0)
df_factor['code'] = df_factor['code'].apply(lambda x: x[-6:])
fname = 'data/df_factor_introday_0955-1005_outofsample.csv'
if 'date' in df_factor.columns:
    df_factor.to_csv(fname, index=None)
else:
    df_factor.to_csv(fname)
