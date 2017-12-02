import numpy as np

#交通工具转移矩阵
Hidden=np.array([[90, 4, 2, 4],\
           [7, 86, 1, 6],\
           [8, 7, 80, 5],\
           [10, 2, 3, 85]], dtype=float)/100.

#交通工具-地点转移矩阵
Seen=np.array([[10, 60, 40],\
           [60, 20, 20],\
           [30, 10, 60],\
           [30, 30, 40]], dtype=float)/100.

print("交通工具转移矩阵:")
print(Hidden)
print("交通工具-地点转移矩阵:")
print(Seen)
print("次数 餐馆 博物馆 公园")
#bus出门
Init_State=np.array([100,0,0,0],dtype=float)/100.
#第一次去的地方
print("1 ",np.dot(Init_State,Seen))

#第二次换乘
Hidden_state=np.dot(Init_State,Hidden)
#第二次去的地方
print("2 ",np.dot(Hidden_state,Seen))

#第三次换乘
Hidden_state=np.dot(Hidden_state,Hidden)
#第三次去的地方
print("3 ",np.dot(Hidden_state,Seen))