"""
发工资
"""
import random

sum = 10000
for i in range(1, 21):
    print(f"开始给第{i}位员工发工资")
    j = random.randint(1, 10)
    if j > 5 and sum>1000:
        print(f"你的绩效分为{j}分,请领取工资1000元", end='')
        sum -= 1000
        print(f"账户余额为{sum}")
    elif sum<1000:
        print("余额不足，不能发工资，请下个月领取")
        break
    else:
        print(f"你的绩效分为{j}分,不能领取工资")
