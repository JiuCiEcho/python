my_list=[21,25,21,23,22,20]

my_list.append(31)
print(my_list)

my_list.extend([19,33,30])
print(my_list)

list_0=my_list.pop(0)
print(list_0)

list_1=my_list.pop(len(my_list)-1)
print(list_1)

index=my_list.index(31)
print(index)

count=my_list.count(21)
print(count)

my_list.insert(0,21)
print(my_list)

del my_list[0]
print(my_list)

print(len(my_list))

my_list.clear()

print([[1,2,3,4,5],[2,58,6,142,4]])


