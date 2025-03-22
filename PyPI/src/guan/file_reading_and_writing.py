# Module: file_reading_and_writing

# 使用pickle将变量保存到文件（支持几乎所有对象类型）
def dump_data(data, filename, file_format='.pkl'):
    import pickle
    with open(filename+file_format, 'wb') as f:
        pickle.dump(data, f)

# 使用pickle从文件中恢复数据到变量（支持几乎所有对象类型）
def load_data(filename, file_format='.pkl'):
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

# 读取文件中的一维数据（一行一组x和y）
def read_one_dimensional_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r', encoding='UTF-8')
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
    f = open(filename+file_format, 'r', encoding='UTF-8')
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

# 读取文件中的XYZ数据（一行一组x, y, z）
def read_xyz_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r', encoding='UTF-8')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    x_array = np.array([])
    y_array = np.array([])
    z_array = np.array([])
    for row in row_list:
        column = np.array(row.split()) 
        if column.shape[0] != 0:  
            x_array = np.append(x_array, [float(column[0])], axis=0)
            y_array = np.append(y_array, [float(column[1])], axis=0)
            z_array = np.append(z_array, [float(column[2])], axis=0)
    return x_array, y_array, z_array

# 读取文件中的二维数据（第一行和第一列分别为横纵坐标）
def read_two_dimensional_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r', encoding='UTF-8')
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
    f = open(filename+file_format, 'r', encoding='UTF-8')
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

# 读取文件中的二维数据（不包括横纵坐标）
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

# 在文件中写入XYZ数据（一行一组x, y, z）
def write_xyz_data(x_array, y_array, z_array, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        write_xyz_data_without_opening_file(x_array, y_array, z_array, f)

# 在文件中写入XYZ数据（一行一组x, y, z）（需要输入已打开的文件）
def write_xyz_data_without_opening_file(x_array, y_array, z_array, f):
    import numpy as np
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    z_array = np.array(z_array)
    i0 = 0
    for x0 in x_array:
        f.write(str(x0)+'   ')
        f.write(str(y_array[i0])+'   ')
        f.write(str(z_array[i0])+'\n')
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

# 在文件中写入二维数据（不包括横纵坐标）
def write_two_dimensional_data_without_xy_array(matrix, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f)

# 在文件中写入二维数据（不包括横纵坐标）（需要输入已打开的文件）
def write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f):
    for row in matrix:
        for element in row:
            f.write(str(element)+'   ')
        f.write('\n')

# 创建一个sh文件用于提交任务（PBS）
def make_sh_file_for_qsub(sh_filename='a', command_line='python a.py', cpu_num=1, task_name='task', cd_dir=0):
    sh_content = \
        '#!/bin/sh\n' \
        +'#PBS -N '+task_name+'\n' \
        +'#PBS -l nodes=1:ppn='+str(cpu_num)+'\n'
    if cd_dir==1:
        sh_content += 'cd $PBS_O_WORKDIR\n'
    sh_content += command_line
    with open(sh_filename+'.sh', 'w') as f:
        f.write(sh_content)

# 创建一个sh文件用于提交任务（Slurm）
def make_sh_file_for_sbatch(sh_filename='a', command_line='python a.py', cpu_num=1, task_name='task', cd_dir=0):
    sh_content = \
        '#!/bin/sh\n' \
        +'#SBATCH --job-name='+task_name+'\n' \
        +'#SBATCH --cpus-per-task='+str(cpu_num)+'\n'
    if cd_dir==1:
        sh_content += 'cd $PBS_O_WORKDIR\n'
    sh_content += command_line
    with open(sh_filename+'.sh', 'w') as f:
        f.write(sh_content)

# 创建一个sh文件用于提交任务（LSF）
def make_sh_file_for_bsub(sh_filename='a', command_line='python a.py', cpu_num=1, task_name='task', cd_dir=0, bsub_q=0, queue_name='score'):
    sh_content = \
        '#!/bin/sh\n' \
        +'#BSUB -J '+task_name+'\n' \
        +'#BSUB -n '+str(cpu_num)+'\n'
    if bsub_q==1:
        sh_content += '#BSUB -q '+queue_name+'\n'
    if cd_dir==1:
        sh_content += 'cd $PBS_O_WORKDIR\n'
    sh_content += command_line
    with open(sh_filename+'.sh', 'w') as f:
        f.write(sh_content)

# qsub 提交任务（PBS）
def qsub_task(filename='a', file_format='.sh'):
    import os
    os.system('qsub '+filename+file_format)

# sbatch 提交任务（Slurm）
def sbatch_task(filename='a', file_format='.sh'):
    import os
    os.system('sbatch '+filename+file_format)

# bsub 提交任务（LSF）
def bsub_task(filename='a', file_format='.sh'):
    import os
    os.system('bsub < '+filename+file_format)

# 复制.py和.sh文件，然后提交任务，实现半手动并行（PBS）
def copy_py_sh_file_and_qsub_task(parameter_array, py_filename='a', old_str_in_py='parameter = 0', new_str_in_py='parameter = ', sh_filename='a', task_name='task'):
    import os
    parameter_str_array = []
    for i0 in parameter_array:
        parameter_str_array.append(str(i0))
    index = 0
    for parameter_str in parameter_str_array:
        index += 1
        # copy python file
        old_file = py_filename+'.py'
        new_file = py_filename+'_'+str(index)+'.py'
        os.system('cp '+old_file+' '+new_file)
        with open(new_file, 'r') as f:
            content  = f.read()
        old_str = old_str_in_py
        new_str = new_str_in_py+parameter_str
        content = content.replace(old_str, new_str)
        with open(py_filename+'_'+str(index)+'.py', 'w') as f:
            f.write(content)
        # copy sh file
        old_file = sh_filename+'.sh'
        new_file = sh_filename+'_'+str(index)+'.sh'
        os.system('cp '+old_file+' '+new_file)
        with open(new_file, 'r') as f:
            content  = f.read()
        old_str = 'python '+py_filename+'.py'
        new_str = 'python '+py_filename+'_'+str(index)+'.py'
        content = content.replace(old_str, new_str)
        old_str = '#PBS -N '+task_name
        new_str = '#PBS -N '+task_name+'_'+str(index)
        content = content.replace(old_str, new_str)
        with open(sh_filename+'_'+str(index)+'.sh', 'w') as f: 
            f.write(content)
        # qsub task
        os.system('qsub '+new_file)

# 复制.py和.sh文件，然后提交任务，实现半手动并行（Slurm）
def copy_py_sh_file_and_sbatch_task(parameter_array, py_filename='a', old_str_in_py='parameter = 0', new_str_in_py='parameter = ', sh_filename='a', task_name='task'):
    import os
    parameter_str_array = []
    for i0 in parameter_array:
        parameter_str_array.append(str(i0))
    index = 0
    for parameter_str in parameter_str_array:
        index += 1
        # copy python file
        old_file = py_filename+'.py'
        new_file = py_filename+'_'+str(index)+'.py'
        os.system('cp '+old_file+' '+new_file)
        with open(new_file, 'r') as f:
            content  = f.read()
        old_str = old_str_in_py
        new_str = new_str_in_py+parameter_str
        content = content.replace(old_str, new_str)
        with open(py_filename+'_'+str(index)+'.py', 'w') as f:
            f.write(content)
        # copy sh file
        old_file = sh_filename+'.sh'
        new_file = sh_filename+'_'+str(index)+'.sh'
        os.system('cp '+old_file+' '+new_file)
        with open(new_file, 'r') as f:
            content  = f.read()
        old_str = 'python '+py_filename+'.py'
        new_str = 'python '+py_filename+'_'+str(index)+'.py'
        content = content.replace(old_str, new_str)
        old_str = '#SBATCH --job-name='+task_name
        new_str = '#SBATCH --job-name='+task_name+'_'+str(index)
        content = content.replace(old_str, new_str)
        with open(sh_filename+'_'+str(index)+'.sh', 'w') as f: 
            f.write(content)
        # sbatch task
        os.system('sbatch '+new_file)

# 复制.py和.sh文件，然后提交任务，实现半手动并行（LSF）
def copy_py_sh_file_and_bsub_task(parameter_array, py_filename='a', old_str_in_py='parameter = 0', new_str_in_py='parameter = ', sh_filename='a', task_name='task'):
    import os
    parameter_str_array = []
    for i0 in parameter_array:
        parameter_str_array.append(str(i0))
    index = 0
    for parameter_str in parameter_str_array:
        index += 1
        # copy python file
        old_file = py_filename+'.py'
        new_file = py_filename+'_'+str(index)+'.py'
        os.system('cp '+old_file+' '+new_file)
        with open(new_file, 'r') as f:
            content  = f.read()
        old_str = old_str_in_py
        new_str = new_str_in_py+parameter_str
        content = content.replace(old_str, new_str)
        with open(py_filename+'_'+str(index)+'.py', 'w') as f:
            f.write(content)
        # copy sh file
        old_file = sh_filename+'.sh'
        new_file = sh_filename+'_'+str(index)+'.sh'
        os.system('cp '+old_file+' '+new_file)
        with open(new_file, 'r') as f:
            content  = f.read()
        old_str = 'python '+py_filename+'.py'
        new_str = 'python '+py_filename+'_'+str(index)+'.py'
        content = content.replace(old_str, new_str)
        old_str = '#BSUB -J '+task_name
        new_str = '#BSUB -J '+task_name+'_'+str(index)
        content = content.replace(old_str, new_str)
        with open(sh_filename+'_'+str(index)+'.sh', 'w') as f: 
            f.write(content)
        # bsub task
        os.system('bsub < '+new_file)

# 把矩阵写入.md文件（Markdown表格形式）
def write_matrix_in_markdown_format(matrix, filename='a'):
    import numpy as np
    matrix = np.array(matrix)
    dim_0 = matrix.shape[0]
    dim_1 = matrix.shape[1]
    with open(filename+'.md', 'w', encoding='UTF-8') as f:
        for i1 in range(dim_1):
            f.write(f'| column {i1+1} ')
        f.write('|\n')
        for i1 in range(dim_1):
            f.write('| :---: ')
        f.write('|\n')
        for i0 in range(dim_0):
            for i1 in range(dim_1):
                f.write(f'| {matrix[i0, i1]} ')
            f.write('|\n')

# 把矩阵写入.md文件（Latex形式）
def write_matrix_in_latex_format(matrix, filename='a', format='bmatrix'):
    import numpy as np
    matrix = np.array(matrix)
    dim_0 = matrix.shape[0]
    dim_1 = matrix.shape[1]
    with open(filename+'.md', 'w', encoding='UTF-8') as f:
        f.write(f'$$\\begin{{{format}}}\n')
        for i0 in range(dim_0):
            for i1 in range(dim_1):
                if i1 != dim_1-1:
                    f.write(f'{matrix[i0, i1]} & ')
                else:
                    f.write(f'{matrix[i0, i1]} \\\\\n')
        f.write(f'\\end{{{format}}}$$')

# 如果不存在文件夹，则新建文件夹
def make_directory(directory='./test'):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

# 如果不存在文件，则新建空文件
def make_file(file_path='./a.txt'):
    import os
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='UTF-8') as f:
            pass

# 打开文件用于写入，默认为新增内容
def open_file(filename='a', file_format='.txt', mode='add'):
    if mode == 'add':
        f = open(filename+file_format, 'a', encoding='UTF-8')
    elif mode == 'overwrite':
        f = open(filename+file_format, 'w', encoding='UTF-8')
    return f

# 打印到TXT文件
def print_to_file(*args, filename='print_result', file_format='.txt', print_on=True):
    if print_on==True:
        for arg in args:
            print(arg, end=' ')
        print()
    f = open(filename+file_format, 'a', encoding='UTF-8')
    for arg in args:
        f.write(str(arg)+' ')
    f.write('\n')
    f.close()

# 读取文本文件内容。如果文件不存在，返回空字符串
def read_text_file(file_path='./a.txt', make_file=None):
    import os
    if not os.path.exists(file_path):
        if make_file != None:
            with open(file_path, 'w', encoding='UTF-8') as f:
                pass
        return ''
    else:
        with open(file_path, 'r', encoding='UTF-8') as f:
            content = f.read()
        return content

# 获取当前文件夹中的所有子文件夹名
def get_all_directories_in_current_directory(current_directory='./'):
    import os
    all_items = os.listdir(current_directory)
    directories = [item for item in all_items if os.path.isdir(os.path.join(current_directory, item))]
    return directories

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
        file_list = sorted(file_list)
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
        with open(file, 'r', encoding='UTF-8') as f:
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
def copy_file(old_file='./a.txt', new_file='./b.txt'):
    import shutil
    shutil.copy(old_file, new_file)

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
    guan.copy_file(old_file=old_file, new_file=new_file)
    content = guan.read_text_file(file_path=new_file)
    content = content.replace(old_str, new_str)
    f = guan.open_file(filename=new_file, file_format='', mode='overwrite')
    f.write(content)
    f.close()

# 改变当前的目录位置
def change_directory_by_replacement(current_key_word='code', new_key_word='data'):
    import os
    code_path = os.getcwd()
    data_path = code_path.replace('\\', '/') 
    data_path = data_path.replace(current_key_word, new_key_word) 
    if os.path.exists(data_path) == False:
        os.makedirs(data_path)
    os.chdir(data_path)

# 查找文件名相同的文件
def find_repeated_file_with_same_filename(directory='./', ignored_directory_with_words=[], ignored_file_with_words=[], num=1000):
    import os
    from collections import Counter
    file_list = []
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            file_list.append(files[i0])
            for word in ignored_directory_with_words:
                if word in root:
                    file_list.remove(files[i0])       
            for word in ignored_file_with_words:
                if word in files[i0]:
                    try:
                        file_list.remove(files[i0])   
                    except:
                        pass 
    count_file = Counter(file_list).most_common(num)
    repeated_file = []
    for item in count_file:
        if item[1]>1:
            repeated_file.append(item)
    return repeated_file

# 统计各个子文件夹中的文件数量
def count_file_in_sub_directory(directory='./', sort=0, reverse=1, print_show=1, smaller_than_num=None):
    import os
    import numpy as np
    dirs_list = []
    for root, dirs, files in os.walk(directory):
        if dirs != []:
            for i0 in range(len(dirs)):
                dirs_list.append(root+'/'+dirs[i0])
    count_file_array = []
    for sub_dir in dirs_list:
        file_list = []
        for root, dirs, files in os.walk(sub_dir):
            for i0 in range(len(files)):
                file_list.append(files[i0])
        count_file = len(file_list)
        count_file_array.append(count_file)
        if sort == 0:
            if print_show == 1:
                if smaller_than_num == None:
                    print(sub_dir)
                    print(count_file)
                    print()
                else:
                    if count_file<smaller_than_num:
                        print(sub_dir)
                        print(count_file)
                        print()
    if sort == 0:
        sub_directory = dirs_list
        num_in_sub_directory = count_file_array
    if sort == 1:
        sub_directory = []
        num_in_sub_directory = []
        if reverse == 1:
            index_array = np.argsort(count_file_array)[::-1]
        else:
            index_array = np.argsort(count_file_array)
        for i0 in index_array:
            sub_directory.append(dirs_list[i0])
            num_in_sub_directory.append(count_file_array[i0])
            if print_show == 1:
                if smaller_than_num == None:
                    print(dirs_list[i0])
                    print(count_file_array[i0])
                    print()
                else:
                    if count_file_array[i0]<smaller_than_num:
                        print(dirs_list[i0])
                        print(count_file_array[i0])
                        print()
    return sub_directory, num_in_sub_directory

# 在多个子文件夹中产生必要的文件，例如 readme.md
def creat_necessary_file(directory, filename='readme', file_format='.md', content='', overwrite=None, ignored_directory_with_words=[]):
    import os
    directory_with_file = []
    ignored_directory = []
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if root not in directory_with_file:
                directory_with_file.append(root)
            if files[i0] == filename+file_format:
                if root not in ignored_directory:
                    ignored_directory.append(root)
    if overwrite == None:
        for root in ignored_directory:
            directory_with_file.remove(root)
    ignored_directory_more =[]
    for root in directory_with_file: 
        for word in ignored_directory_with_words:
            if word in root:
                if root not in ignored_directory_more:
                    ignored_directory_more.append(root)
    for root in ignored_directory_more:
        directory_with_file.remove(root) 
    for root in directory_with_file:
        os.chdir(root)
        f = open(filename+file_format, 'w', encoding="utf-8")
        f.write(content)
        f.close()

# 删除特定文件名的文件（谨慎使用）
def delete_file_with_specific_name(directory, filename='readme', file_format='.md'):
    import os
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if files[i0] == filename+file_format:
                os.remove(root+'/'+files[i0])

# 将所有文件移到根目录（谨慎使用）
def move_all_files_to_root_directory(directory):
    import os
    import shutil
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            shutil.move(root+'/'+files[i0], directory+'/'+files[i0])
    for i0 in range(100):
        for root, dirs, files in os.walk(directory):
            try:
                os.rmdir(root) 
            except:
                pass