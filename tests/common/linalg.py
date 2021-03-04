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

## test determinant of a matrix
A = ulab.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
det = ulab.linalg.det(A)

# det is a float containing the determinant
print(det)

## test 2-norm of a vector
x = ulab.array([5, 2, 3])
norm = ulab.linalg.norm(x)

# norm is a float containing the 2-norm of the vector
print(norm)

## test 2-norm of a 2*2 matrix
A = ulab.array([[-2, 5], [8, 7]])
norm = ulab.linalg.norm(A)

# norm is a float containing the 2-norm of the matrix
print(norm)

## test 2-norm of a 3*3 matrix
A = ulab.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
norm = ulab.linalg.norm(A)

# norm is a float containing the 2-norm of the matrix
print(norm)

## test trace of a 3*3 square matrix
A = ulab.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
trace = ulab.linalg.trace(A)

# trace is a float containing the trace of the matrix
print(trace)

## test trace of a 1*1 square matrix
A = ulab.array([[4]])
trace = ulab.linalg.trace(A)

# trace is a float containing the trace of the matrix
print(trace)
