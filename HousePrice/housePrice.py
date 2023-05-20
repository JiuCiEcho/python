import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv('data.csv', index_col=0, encoding='utf-8')

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

region_groups = df.groupby('区域')
# 热门户型分析
popular_types = {}

for region, region_data in region_groups:
    popular_types[region] = region_data['户型'].value_counts().head(5)

plt.figure(figsize=(10, 6))
for region, types in popular_types.items():
    plt.bar(types.index, types.values, label=region)

plt.title('各区域热门户型分析', fontproperties='SimHei', fontsize=16)
plt.xlabel('户型', fontproperties='SimHei', fontsize=12)
plt.ylabel('数量', fontproperties='SimHei', fontsize=12)
plt.legend()
plt.xticks(rotation=45, fontproperties='SimHei', fontsize=10)
plt.show()

# 分区域拟合房价走势模型
models = {}

for region, region_data in region_groups:
    X = region_data[['建筑面积', '单价']]
    y = region_data['总价']
    # 限制建筑面积在600以内
    mask = X['建筑面积'] <= 600
    X = X[mask]
    y = y[mask]
    reg = LinearRegression().fit(X, y)
    models[region] = reg

# 绘制建筑面积和单价对总价的影响图
plt.figure(figsize=(10, 6))
for region, reg in models.items():
    region_data = region_groups.get_group(region)
    X = region_data[['建筑面积', '单价']]
    y = region_data['总价']
    # 限制建筑面积在600以内
    mask = X['建筑面积'] <= 600
    X = X[mask]
    y = y[mask]
    plt.scatter(X['建筑面积'], X['单价'], c=y, cmap='viridis', label=region)
    plt.plot(X['建筑面积'], reg.predict(X), label=f'{region}拟合线', linewidth=2)

plt.xlabel('建筑面积',fontproperties='SimHei',fontsize=12)
plt.ylabel('单价',fontproperties='SimHei',fontsize=12)
plt.title('建筑面积和单价对总价的影响',fontproperties='SimHei',fontsize=12)
plt.legend()
plt.colorbar(label='总价')
plt.show()

