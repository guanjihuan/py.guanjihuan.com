import numpy as np
import cmath
import guan

# Example 1
vector = np.array([np.sqrt(0.5), np.sqrt(0.5)])*cmath.exp(np.random.uniform(0, 1)*1j)
print('\nExample 1\n', vector)
print(np.dot(vector.transpose().conj(), vector), '\n')

vector = guan.find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.001)
print(vector)
print(np.dot(vector.transpose().conj(), vector), '\n')


# Example 2
vector = np.array([1, 0])*cmath.exp(np.random.uniform(0, 1)*1j)
print('\nExample 2\n', vector)
print(np.dot(vector.transpose().conj(), vector), '\n')

vector = guan.find_vector_with_fixed_gauge_by_making_one_component_real(vector, precision=0.001)
print(vector)
print(np.dot(vector.transpose().conj(), vector), '\n')