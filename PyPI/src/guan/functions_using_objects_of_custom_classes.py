# functions_using_objects_of_custom_classes

# 将原子对象列表转出原子字典列表
def convert_atom_object_list_to_atom_dict_list(atom_object_list):
    atom_dict_list = []
    for atom_object in atom_object_list:
        atom_dict = {
            'name': atom_object.name,
            'index': atom_object.index, 
            'x': atom_object.x,
            'y': atom_object.y,
            'z': atom_object.z,
            'energy': atom_object.energy,
        }
        atom_dict_list.append(atom_dict)
    return atom_dict_list

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

# 从原子对象列表中获取满足坐标条件的索引
def get_index_via_coordinate_from_atom_object_list(atom_object_list, x=0, y=0, z=0, eta=1e-3):
    for atom in atom_object_list:
        x_i = atom.x
        y_i = atom.y
        z_i = atom.z
        index = atom.index
        if abs(x-x_i)<eta and abs(y-y_i)<eta and abs(z-z_i)<eta:
            return index

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