# functions_using_objects_of_custom_classes

# 从原子对象列表中获取 (x, y) 坐标数组
def get_coordinate_array_from_atom_object_list(atom_object_list):
    coordinate_array = []
    for atom in atom_object_list:
        x = atom.x
        y = atom.y
        coordinate_array.append([x, y])
    return coordinate_array

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