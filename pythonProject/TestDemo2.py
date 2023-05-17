money = 5000000
name = input("请输入你的名字：")


def check_money():
    """
    查询余额函数
    :return: None
    """
    print(f"您的余额是{money}")


def add_money(money):
    inMoney = int(input("请输入你要存钱的金额:"))
    money += inMoney
    return money


def withinDraw(money):
    outMoney = int(input("请输入你要取款的金额:"))
    money -= outMoney
    return money


def menu():
    print("1. 存款     2. 取款")
    print("3. 查询余额     4. 退出")
    return int(input("请输入你的选择:"))


while True:
    choose=menu()
    if choose == 1:
        money = add_money(money)
        check_money()
    elif choose == 2:
        money = withinDraw(money)
        check_money()
    elif choose==3:
        check_money()
    elif choose==4:
        print("退出成功~~~")
        break;
    else:
        print("对不起，你输入错误~~~")
