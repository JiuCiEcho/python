import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 读取数据
df = pd.read_csv('data.csv', index_col=0, encoding='utf-8')
print(df.head())

# 数据清洗
df.dropna(inplace=True)  # 删除空值
df['单价'] = df['单价'].str.extract(r'(\d+\.?\d*)').astype(float)
df['总价'] = df['总价'].str.extract(r'(\d+\.?\d*)').astype(float)
df['建筑面积'] = df['建筑面积'].str.extract(r'(\d+\.?\d*)').astype(float)


# 区域二手房均价分析
mean_price = df.groupby('区域')['总价'].mean().sort_values(ascending=False)
mean_price.plot(kind='bar', figsize=(10, 6))
plt.title('不同区域二手房均价对比', fontproperties='SimHei', fontsize=16)
plt.xlabel('区域', fontproperties='SimHei', fontsize=12)
plt.ylabel('均价', fontproperties='SimHei', fontsize=12)
plt.xticks(fontproperties='SimHei', fontsize=10)
plt.show()

# 热门户型分析
plt.figure(figsize=(10, 6))
sns.countplot(x='户型', data=df, order=df['户型'].value_counts().index)
plt.title('热门户型分析', fontproperties='SimHei', fontsize=16)
plt.xlabel('户型', fontproperties='SimHei', fontsize=12)
plt.ylabel('数量', fontproperties='SimHei', fontsize=12)
plt.xticks(rotation=45, fontproperties='SimHei', fontsize=10)
plt.show()

# 二手房价预测
X = df[['建筑面积', '单价']]
y = df['总价']
reg = LinearRegression().fit(X, y)

new_X = [[100, 10000], [150, 8000], [200, 12000]]
new_y = reg.predict(new_X)
print(f'新数据预测结果为：{new_y}')
