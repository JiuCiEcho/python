my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
my_list_1 = []
i=0
for i in my_list:
    if my_list[i] % 2 == 0:
        my_list_1.append(my_list[i])
print(my_list_1)

my_list_2=[]
while i<len(my_list):
    if my_list[i] % 2 == 0:
        my_list_1.append(my_list[i])
    i+=1
print(my_list_2)
if my_list[i] % 2 == 0:
    my_list_1.append(my_list[i])
