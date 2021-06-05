import gjh
import numpy as np

x = np.array([1, 2, 3])
y = np.array([5, 6, 7])
gjh.write_one_dimensional_data(x, y, filename='one_dimensional_data')

matrix = np.zeros((3, 3))
matrix[0, 1] = 11
gjh.write_two_dimensional_data(x, y, matrix, filename='two_dimensional_data')


x_read, y_read = gjh.read_one_dimensional_data('one_dimensional_data')
print(x_read, '\n')
print(y_read, '\n\n')

x_read, y_read, matrix_read = gjh.read_two_dimensional_data('two_dimensional_data')
print(x_read, '\n')
print(y_read, '\n')
print(matrix_read)