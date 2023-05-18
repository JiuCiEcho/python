t1=('周杰伦',11,['football','music'])

#查询其年龄在其下标位置
num=t1.index(11)
print(f'年龄的下标为{num}')

#查询学生的姓名
name=t1[0]
print(f"学生的姓名为：{name}")

#删除学生的爱好football
t1[2].remove('football')
print(t1)

#增加爱好
t1[2].append("coding")
print(t1)


