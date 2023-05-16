"""
用for循环输出九九乘法表
"""

i = 1
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{i}*{j}={i * j}\t", end='')
    print()

m=1
while m<=9:
    n=1
    while n<=m:
        print(f"{m}*{n}={m * n}\t", end='')
        n+=1
    print()
    m+=1






