import guan
import numpy as np

x_array = np.array([1, 2, 3])
y_array = np.array([5, 6, 7])
guan.write_one_dimensional_data(x_array, y_array, filename='one_dimensional_data')
matrix = np.zeros((3, 3))
matrix[0, 1] = 11
guan.write_two_dimensional_data(x_array, y_array, matrix, filename='two_dimensional_data')
x_read, y_read = guan.read_one_dimensional_data('one_dimensional_data')
print(x_read, '\n')
print(y_read, '\n\n')
x_read, y_read, matrix_read = guan.read_two_dimensional_data('two_dimensional_data')
print(x_read, '\n')
print(y_read, '\n')
print(matrix_read)