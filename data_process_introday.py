import pandas as pd
import os
import datetime

root_dir = 'factors/factors'
flist = os.listdir(root_dir)
f = flist[0]
df_list = []

no_x_cols = ['ret_15', 'ret_60', 'code']#'ret_60',
for f in flist:
    fpath = os.path.join(root_dir, f)
    df = pd.read_csv(fpath, parse_dates=['ts'])
    code = f.split('.')[0]
    df['date'] = df.ts.dt.date
    df['time'] = df.ts.dt.time
    df = df[(df['time'] <= pd.Timestamp('14:30').time())&
            (df['time'] > pd.Timestamp('14:20').time())]

    # df = df[df['time'] == pd.Timestamp('10:00').time()]
    df = df[df.columns.difference(no_x_cols)].groupby('date').mean()
    df['code'] = code
    df_list.append(df)
    print(code)

df_factor = pd.concat(df_list, axis=0)
df_factor['code'] = df_factor['code'].apply(lambda x: x[-6:])
df_factor.to_csv('df_factor_introday_1420-1430-mean.csv')
