import numpy as np

M=np.array([[90, 4, 2, 4],\
           [7, 86, 1, 6],\
           [8, 7, 80, 5],\
           [10, 2, 3, 85]], dtype=float)/100.

print("Transform Matrix:")
print(M)

V1=np.array([19,14,56,11],dtype=float)/100.
print("第一次:")
print(V1)

V2=np.dot(V1,M)
print("第二次:")
print(V2)

V3=np.dot(V2,M)
print("第二次:")
print(V3)


V_init=np.array([19,14,56,11],dtype=float)/100.
print("原初向量:")
print(V_init)
for i in range(10000):
    V_init=np.dot(V_init,M)

print("原初始化的向量最后结果:")
print(V_init)

V_init2=np.array([100,0,0,0],dtype=float)/100.
print("原初向量:")
print(V_init2)
for i in range(10000):
    V_init2=np.dot(V_init2,M)

print("一开始全部坐bus最后结果:")
print(V_init2)