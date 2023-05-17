
"""
https://stackoverflow.com/questions/11585793/are-numpy-arrays-passed-by-reference
https://stackoverflow.com/questions/9047111/vs-operators-with-numpy

https://blog.finxter.com/python-__isub__-magic-method/

"""
import numpy as np

def foo(arr):
    arr = arr - 3
    # return arr

def bar(arr):
    arr -= 3
    # return arr

a = np.array([3, 4, 5])
a = np.array([10])
# https://stackoverflow.com/questions/4088731/python-int-doesnt-have-iadd-method
# a = 10  # not be effected., Integer has no __iadd__()

foo(a)
print(a) # prints [3, 4, 5]

b = a
foo(b)
print(b) # prints [3, 4, 5]

bar(b)
print(b) # prints [0, 1, 2]
print(a)

b = a
bar(b)
print(b, a) # prints [3, 4, 5]

b = a*1
bar(b)
print(b, a) # prints [3, 4, 5]

