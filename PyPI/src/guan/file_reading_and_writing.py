# Module: file_reading_and_writing

# 使用pickle将变量保存到文件（支持几乎所有对象类型）
def dump_data(data, filename, file_format='.txt'):
    import pickle
    with open(filename+file_format, 'wb') as f:
        pickle.dump(data, f)

# 使用pickle从文件中恢复数据到变量（支持几乎所有对象类型）
def load_data(filename, file_format='.txt'):
    import pickle
    with open(filename+file_format, 'rb') as f:
        data = pickle.load(f)
    return data

# 使用NumPy保存数组变量到npy文件（二进制文件）
def save_npy_data(data, filename):
    import numpy as np
    np.save(filename+'.npy', data)

# 使用NumPy从npy文件恢复数据到数组变量（二进制文件）
def load_npy_data(filename):
    import numpy as np
    data = np.load(filename+'.npy')
    return data

# 使用NumPy保存数组变量到TXT文件（文本文件）
def save_txt_data(data, filename):
    import numpy as np
    np.savetxt(filename+'.txt', data)

# 使用NumPy从TXT文件恢复数据到数组变量（文本文件）
def load_txt_data(filename):
    import numpy as np
    data = np.loadtxt(filename+'.txt')
    return data

# 如果不存在文件夹，则新建文件夹
def make_directory(directory='./test'):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

# 如果不存在文件，则新建空文件
def make_file(file_path='./a.txt'):
    import os
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass

# 打开文件用于写入，默认为新增内容
def open_file(filename='a', file_format='.txt', mode='add'):
    if mode == 'add':
        f = open(filename+file_format, 'a', encoding='UTF-8')
    elif mode == 'overwrite':
        f = open(filename+file_format, 'w', encoding='UTF-8')
    return f

# 读取文本文件内容。如果文件不存在，返回空字符串
def read_text_file(file_path='./a.txt', make_file=None):
    import os
    if not os.path.exists(file_path):
        if make_file != None:
            with open(file_path, 'w') as f:
                pass
        return ''
    else:
        with open(file_path, 'r') as f:
            content = f.read()
        return content

# 获取目录中的所有文件名
def get_all_filenames_in_directory(directory='./', file_format=None, show_root_path=0, sort=1, include_subdirectory=1):
    import os
    file_list = []
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if file_format == None:
                if show_root_path == 0:
                    file_list.append(files[i0])
                else:
                    file_list.append(root+'/'+files[i0])
            else:
                if file_format in files[i0]:
                    if show_root_path == 0:
                        file_list.append(files[i0])
                    else:
                        file_list.append(root+'/'+files[i0])
        if include_subdirectory != 1:
            break
    if sort == 1:
        sorted(file_list)
    return file_list

# 获取文件夹中某种文本类型的文件以及读取所有内容
def read_text_files_in_directory(directory='./', file_format='.md'):
    import os
    file_list = []
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if file_format in files[i0]:
                file_list.append(root+'/'+files[i0])
    content_array = []
    for file in file_list:
        with open(file, 'r') as f:
            content_array.append(f.read())
    return file_list, content_array

# 在多个文本文件中查找关键词
def find_words_in_multiple_files(words, directory='./', file_format='.md'):
    import guan
    file_list, content_array = guan.read_text_files_in_directory(directory=directory, file_format=file_format)
    num_files = len(file_list)
    file_list_with_words = []
    for i0 in range(num_files):
        if words in content_array[i0]:
            file_list_with_words.append(file_list[i0])
    return file_list_with_words

# 复制一份文件
def copy_file(file1='./a.txt', file2='./b.txt'):
    import shutil
    shutil.copy(file1, file2)

# 打开文件，替代某字符串
def open_file_and_replace_str(file_path='./a.txt', old_str='', new_str=''):
    import guan
    content = guan.read_text_file(file_path=file_path)
    content = content.replace(old_str, new_str)
    f = guan.open_file(filename=file_path, file_format='', mode='overwrite')
    f.write(content)
    f.close()

# 复制一份文件，然后再替代某字符串
def copy_file_and_replace_str(old_file='./a.txt', new_file='./b.txt', old_str='', new_str=''):
    import guan
    guan.copy_file(file1=old_file, file2=new_file)
    content = guan.read_text_file(file_path=new_file)
    content = content.replace(old_str, new_str)
    f = guan.open_file(filename=new_file, file_format='', mode='overwrite')
    f.write(content)
    f.close()

# 拼接两个PDF文件
def combine_two_pdf_files(input_file_1='a.pdf', input_file_2='b.pdf', output_file='combined_file.pdf'):
    import PyPDF2
    output_pdf = PyPDF2.PdfWriter()
    with open(input_file_1, 'rb') as file1:
        pdf1 = PyPDF2.PdfReader(file1)
        for page in range(len(pdf1.pages)):
            output_pdf.add_page(pdf1.pages[page])
    with open(input_file_2, 'rb') as file2:
        pdf2 = PyPDF2.PdfReader(file2)
        for page in range(len(pdf2.pages)):
            output_pdf.add_page(pdf2.pages[page])
    with open(output_file, 'wb') as combined_file:
        output_pdf.write(combined_file)

# 读取文件中的一维数据（一行一组x和y）
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
    return x_array, y_array

# 读取文件中的一维数据（一行一组x和y）（支持复数形式）
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
    return x_array, y_array

# 读取文件中的二维数据（第一行和第一列分别为横纵坐标）
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
    return x_array, y_array, matrix

# 读取文件中的二维数据（第一行和第一列分别为横纵坐标）（支持复数形式）
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
    return x_array, y_array, matrix

# 读取文件中的二维数据（不包括x和y）
def read_two_dimensional_data_without_xy_array(filename='a', file_format='.txt'):
    import numpy as np
    matrix = np.loadtxt(filename+file_format)
    return matrix

# 在文件中写入一维数据（一行一组x和y）
def write_one_dimensional_data(x_array, y_array, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_one_dimensional_data_without_opening_file(x_array, y_array, f)

# 在文件中写入一维数据（一行一组x和y）（需要输入已打开的文件）
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

# 在文件中写入二维数据（第一行和第一列分别为横纵坐标）
def write_two_dimensional_data(x_array, y_array, matrix, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_two_dimensional_data_without_opening_file(x_array, y_array, matrix, f)

# 在文件中写入二维数据（第一行和第一列分别为横纵坐标）（需要输入已打开的文件）
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

# 在文件中写入二维数据（不包括x和y）
def write_two_dimensional_data_without_xy_array(matrix, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f)

# 在文件中写入二维数据（不包括x和y）（需要输入已打开的文件）
def write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f):
    for row in matrix:
        for element in row:
            f.write(str(element)+'   ')
        f.write('\n')