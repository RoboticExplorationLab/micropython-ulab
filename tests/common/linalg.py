import ulab
from ulab import linalg

a = ulab.array([[1, 2], [3, 4]])
print(linalg.inv(a))

b = ulab.array([[1, 2, 3], [4, 5, 6], [7, 8, 7]])
print(linalg.inv(b))

c = ulab.array([[1, 2, 0, 0], [0, 6, 7, 0], [0, 0, 8, 9], [0, 0, 15, 13]])
print(linalg.inv(c))

print(linalg.det(a))
print(linalg.det(b))
print(linalg.det(c))

## test matrix vector multiplication
A = ulab.array([[1, 2, 3], [4, 5, 6], [7, 8, 7]])
x = ulab.array([1,2,3])
b = ulab.linalg.dot(A,x)

# A is a 2d array (3, 3)
print(A.shape())

# x is a 1d array (3,)
print(x.shape())

# b is a 1d array (3,)
print(b.shape())

# A*x = b
print(b)
