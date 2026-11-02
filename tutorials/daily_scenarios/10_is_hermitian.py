# 检查是否为厄米矩阵
import guan

matrix1 = [
    [2, 1.00001-1j],
    [1+1j, 1]
]

print(guan.is_hermitian(matrix1))

matrix2 = [
    [2, 1.00002-1j],
    [1+1j, 1]
]

print(guan.is_hermitian(matrix2))
