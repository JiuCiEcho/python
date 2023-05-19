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
start_date = pd.Timestamp('2017-01-01 00:00:00')
end_date = pd.Timestamp('2017-06-30 00:00:00')
mask = (df['addtime'] >= start_date) & (df['addtime'] <= end_date)
new_reg_users = df.loc[mask]
new_reg_users['week'] = new_reg_users['addtime'].dt.strftime('%U')

weekly_reg_users = new_reg_users.groupby('week')['username'].count().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_reg_users, x='week', y='username', color='blue', label='Registered Users')
plt.fill_between(weekly_reg_users['week'], weekly_reg_users['username'], alpha=0.2, color='blue')
plt.xlabel('Week')
plt.ylabel('Number of registered users')
plt.title('Weekly new registered users (Jan-Jun 2017)')
plt.xticks(rotation=45)
plt.legend()
plt.show()