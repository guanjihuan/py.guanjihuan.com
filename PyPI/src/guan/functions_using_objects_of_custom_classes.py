# functions_using_objects_of_custom_classes

# 从原子对象列表中获取 (x, y) 坐标数组
def get_coordinate_array_from_atom_object_list(atom_object_list):
    coordinate_array = []
    for atom in atom_object_list:
        x = atom.x
        y = atom.y
        coordinate_array.append([x, y])
    return coordinate_array

# 从原子对象列表中获取 x 和 y 的最大值和最小值
def get_max_min_x_y_from_atom_object_list(atom_object_list):
    import guan
    coordinate_array = guan.get_coordinate_array_from_atom_object_list(atom_object_list)
    x_array = []
    for coordinate in coordinate_array:
        x_array.append(coordinate[0])
    y_array = []
    for coordinate in coordinate_array:
        y_array.append(coordinate[1])
    max_x = max(x_array)
    min_x = min(x_array)
    max_y = max(y_array)
    min_y = min(y_array)
    return max_x, min_x, max_y, min_y

# 根据原子对象列表来初始化哈密顿量
def initialize_hamiltonian_from_atom_object_list(atom_object_list):
    import numpy as np
    import guan
    dim = guan.dimension_of_array(atom_object_list[0].energy)
    num = len(atom_object_list)
    hamiltonian = np.zeros((dim*num, dim*num))
    for i0 in range(num):
        hamiltonian[i0*dim+0:i0*dim+dim, i0*dim+0:i0*dim+dim] = atom_object_list[i0].energy
    return hamiltonian