import os

def gen_stuff(x):
	for i in x:
		
		yield(i)
		yield('yes')

a = [1,2,3,4,5]
b = gen_stuff(a)
for i in range(10):
	print(next(b))
