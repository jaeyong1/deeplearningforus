

import numpy as np
s = 483
print("Rank 0. Scalar = ", s) 

v = np.array([1.1, 2.2 , 3.3]) #Vector
print("Rank 1. Shape = ", v.shape)
print("Rank 1. Items = \n", v)

m = np.array([[1,2,3], [4,5,6]]) #Matrix
print("Rank 2. Shape = ", m.shape)
print("Rank 2. Items = \n", m)

t = np.array([[[2],[4],[6]], [[8],[10],[12]], [[14],[16],[18]], [[20],[22],[24]]]) #3-Tensor
print("Rank 3. Shape = ", t.shape)
print("Rank 3. Items = \n", t)

t = np.array([[[2,1],[4,1],[6,1]], [[8,1],[10,1],[12,1]], [[14,1],[16,1],[18,1]], [[20,1],[22,1],[24,1]]]) #3-Tensor
print("Rank 3.(2nd) Shape = ", t.shape)
print("Rank 3.(2nd) Items = \n", t)


print("----")

# array생성과 기본값 채우기
import numpy as np

# 0으로 채운 배열생성
a = np.zeros((3,2))
print(a)

# 1로 채운 배열생성
b = np.ones((4,2))
print(b)

# 지정값으로 채운 배열생성
c = np.full((2,3), 7)
print(c)

# 대각선만 1인 배열생성
d = np.eye(5)
print(d)

#랜덤값 채운 배열생성
e = np.random.random((2,2))
print(e)

print("----")

# Array 연산
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
#a,b의 shape = (2,2)
# 덧셈
print (a+b)

# 뺄셈
print (b-a)

# 곱셈
print (a*b)

# 나눗셈
print (a/b)

# 행렬곱셈
print (a@b)
