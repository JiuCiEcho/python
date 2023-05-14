import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据表
df = pd.read_excel('user.xlsx')

# 数据清洗
for col in df.columns:
    print(col, 'has', df[col].isnull().sum(), 'missing values')
    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
        print(col, 'has', df[df[col] < 0].shape[0], 'negative values')
    elif df[col].dtype == 'O':
        print(col, 'is of object type')
        if col == 'last_login_time' or col == 'addtime':
            df[col] = pd.to_datetime(df[col])

# 年度注册用户分析
df['year'] = pd.DatetimeIndex(df['addtime']).year
df['month'] = pd.DatetimeIndex(df['addtime']).month
year_month_group = df.groupby(['year', 'month']).agg({'username': 'count'}).reset_index()
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=year_month_group, x='month', y='username', hue='year')
plt.xlabel('Month')
plt.ylabel('Number of registered users')
plt.title('Annual growth of registered users')
plt.show()

# 新注册用户分析
start_date = pd.Timestamp('2017-01-01')
end_date = pd.Timestamp('2017-06-30')
mask = (df['addtime'] >= start_date) & (df['addtime'] <= end_date)
new_reg_users = df.loc[mask]
new_reg_users_day = new_reg_users.groupby(['addtime']).agg({'username': 'count'}).reset_index()

# 将日期作为行，月份作为列，统计每个月每天新注册用户数量，生成透视表
new_reg_users_day['day'] = pd.DatetimeIndex(new_reg_users_day['addtime']).day
new_reg_users_pivot = pd.pivot_table(new_reg_users_day, values='username', index='day', columns='addtime').fillna(0)

plt.figure(figsize=(12, 6))
sns.heatmap(data=new_reg_users_pivot, cmap='coolwarm')
plt.xlabel('Date')
plt.ylabel('Day of month')
plt.title('New registered users during Jan-Jun 2017')
plt.show()

