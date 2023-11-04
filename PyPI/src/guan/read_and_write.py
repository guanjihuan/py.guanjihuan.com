# Module: read_and_write

# 将数据存到文件
def dump_data(data, filename, file_format='.txt'):
    import pickle
    with open(filename+file_format, 'wb') as f:
        pickle.dump(data, f)
    import guan
    guan.statistics_of_guan_package()

# 从文件中恢复数据到变量
def load_data(filename, file_format='.txt'):
    import pickle
    with open(filename+file_format, 'rb') as f:
        data = pickle.load(f)
    import guan
    guan.statistics_of_guan_package()
    return data

# 读取文件中的一维数据（每一行一组x和y）
def read_one_dimensional_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    for row in row_list:
        column = np.array(row.split()) 
        if column.shape[0] != 0:  
            x_array = np.append(x_array, [float(column[0])], axis=0)  
            y_row = np.zeros(dim_column-1)
            for dim0 in range(dim_column-1):
                y_row[dim0] = float(column[dim0+1])
            if np.array(y_array).shape[0] == 0:
                y_array = [y_row]
            else:
                y_array = np.append(y_array, [y_row], axis=0)
    import guan
    guan.statistics_of_guan_package()
    return x_array, y_array

# 读取文件中的一维数据（每一行一组x和y）（支持复数形式）
def read_one_dimensional_complex_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    for row in row_list:
        column = np.array(row.split()) 
        if column.shape[0] != 0:  
            x_array = np.append(x_array, [complex(column[0])], axis=0)  
            y_row = np.zeros(dim_column-1, dtype=complex)
            for dim0 in range(dim_column-1):
                y_row[dim0] = complex(column[dim0+1])
            if np.array(y_array).shape[0] == 0:
                y_array = [y_row]
            else:
                y_array = np.append(y_array, [y_row], axis=0)
    import guan
    guan.statistics_of_guan_package()
    return x_array, y_array

# 读取文件中的二维数据（第一行和列分别为横纵坐标）
def read_two_dimensional_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    matrix = np.array([])
    for i0 in range(row_list.shape[0]):
        column = np.array(row_list[i0].split()) 
        if i0 == 0:
            x_str = column[1::] 
            x_array = np.zeros(x_str.shape[0])
            for i00 in range(x_str.shape[0]):
                x_array[i00] = float(x_str[i00]) 
        elif column.shape[0] != 0: 
            y_array = np.append(y_array, [float(column[0])], axis=0)  
            matrix_row = np.zeros(dim_column-1)
            for dim0 in range(dim_column-1):
                matrix_row[dim0] = float(column[dim0+1])
            if np.array(matrix).shape[0] == 0:
                matrix = [matrix_row]
            else:
                matrix = np.append(matrix, [matrix_row], axis=0)
    import guan
    guan.statistics_of_guan_package()
    return x_array, y_array, matrix

# 读取文件中的二维数据（第一行和列分别为横纵坐标）（支持复数形式）
def read_two_dimensional_complex_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    matrix = np.array([])
    for i0 in range(row_list.shape[0]):
        column = np.array(row_list[i0].split()) 
        if i0 == 0:
            x_str = column[1::] 
            x_array = np.zeros(x_str.shape[0], dtype=complex)
            for i00 in range(x_str.shape[0]):
                x_array[i00] = complex(x_str[i00]) 
        elif column.shape[0] != 0: 
            y_array = np.append(y_array, [complex(column[0])], axis=0)  
            matrix_row = np.zeros(dim_column-1, dtype=complex)
            for dim0 in range(dim_column-1):
                matrix_row[dim0] = complex(column[dim0+1])
            if np.array(matrix).shape[0] == 0:
                matrix = [matrix_row]
            else:
                matrix = np.append(matrix, [matrix_row], axis=0)
    import guan
    guan.statistics_of_guan_package()
    return x_array, y_array, matrix

# 读取文件中的二维数据（不包括x和y）
def read_two_dimensional_data_without_xy_array(filename='a', file_format='.txt'):
    import numpy as np
    matrix = np.loadtxt(filename+file_format)
    import guan
    guan.statistics_of_guan_package()
    return matrix

# 打开文件用于新增内容
def open_file(filename='a', file_format='.txt'):
    f = open(filename+file_format, 'a', encoding='UTF-8')
    import guan
    guan.statistics_of_guan_package()
    return f

# 在文件中写入一维数据（每一行一组x和y）
def write_one_dimensional_data(x_array, y_array, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_one_dimensional_data_without_opening_file(x_array, y_array, f)
    import guan
    guan.statistics_of_guan_package()

# 在文件中写入一维数据（每一行一组x和y）（需要输入文件）
def write_one_dimensional_data_without_opening_file(x_array, y_array, f):
    import numpy as np
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    i0 = 0
    for x0 in x_array:
        f.write(str(x0)+'   ')
        if len(y_array.shape) == 1:
            f.write(str(y_array[i0])+'\n')
        elif len(y_array.shape) == 2:
            for j0 in range(y_array.shape[1]):
                f.write(str(y_array[i0, j0])+'   ')
            f.write('\n')
        i0 += 1
    import guan
    guan.statistics_of_guan_package()

# 在文件中写入二维数据（第一行和列分别为横纵坐标）
def write_two_dimensional_data(x_array, y_array, matrix, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_two_dimensional_data_without_opening_file(x_array, y_array, matrix, f)
    guan.statistics_of_guan_package()

# 在文件中写入二维数据（第一行和列分别为横纵坐标）（需要输入文件）
def write_two_dimensional_data_without_opening_file(x_array, y_array, matrix, f):
    import numpy as np
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    matrix = np.array(matrix)
    f.write('0   ')
    for x0 in x_array:
        f.write(str(x0)+'   ')
    f.write('\n')
    i0 = 0
    for y0 in y_array:
        f.write(str(y0))
        j0 = 0
        for x0 in x_array:
            f.write('   '+str(matrix[i0, j0])+'   ')
            j0 += 1
        f.write('\n')
        i0 += 1
    import guan
    guan.statistics_of_guan_package()

# 在文件中写入二维数据（不包括x和y）
def write_two_dimensional_data_without_xy_array(matrix, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f)
    guan.statistics_of_guan_package()

# 在文件中写入二维数据（不包括x和y）（需要输入文件）
def write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f):
    for row in matrix:
        for element in row:
            f.write(str(element)+'   ')
        f.write('\n')
    import guan
    guan.statistics_of_guan_package()

# 以显示编号的样式，打印数组
def print_array_with_index(array, show_index=1, index_type=0):
    if show_index==0:
        for i0 in array:
            print(i0)
    else:
        if index_type==0:
            index = 0
            for i0 in array:
                print(index, i0)
                index += 1
        else:
            index = 0
            for i0 in array:
                index += 1
                print(index, i0)
    import guan
    guan.statistics_of_guan_package()
