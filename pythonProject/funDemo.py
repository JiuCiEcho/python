#函数的两种调用方式

#调用函数，函数名作为参数，这种方式注重函数逻辑，不注重函数如何实现
def test_fun(computer):
    result=computer(1,2)
    print(f'函数的结果是{result}')


def computer(x,y):
    return x+y

test_fun(computer)

#使用lambda的方式调用函数
test_fun(lambda x,y:x+y)