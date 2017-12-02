def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield a,b
        # 先计算后面
        a, b = b, a + b
        n = n + 1

for n in fab(5):
    print(n)