# Module: data_processing

# 模型对话
def chat(prompt='你好', model=1, stream=0, top_p=0.8, temperature=0.85):
    import socket
    import json
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.settimeout(30)
        client_socket.connect(('socket.guanjihuan.com', 12345))
        message = {
            'server': "chat.guanjihuan.com",
            'prompt': prompt,
            'model': model,
            'top_p': top_p,
            'temperature': temperature,
        }
        send_message = json.dumps(message)
        client_socket.send(send_message.encode())
        if stream == 1:
            print('\n--- Begin Stream Message ---\n')
        response = ''
        while True:
            try:
                data = client_socket.recv(1024)
                if data != b'':
                    response_data = data.decode()
                    response_dict = json.loads(response_data)
                    stream_response = response_dict['stream_response']
                    response += stream_response
                    end_message = response_dict['end_message']
                    if end_message == 1:
                        break
                    else:
                        if stream == 1:
                            print(stream_response)
            except:
                break
        client_socket.close()
        if stream == 1:
            print('\n--- End Stream Message ---\n')
    return response

# 并行计算前的预处理，把参数分成多份
def preprocess_for_parallel_calculations(parameter_array_all, task_num=1, task_index=0):
    import numpy as np
    num_all = np.array(parameter_array_all).shape[0]
    if num_all%task_num == 0:
        num_parameter = int(num_all/task_num) 
        parameter_array = parameter_array_all[task_index*num_parameter:(task_index+1)*num_parameter]
    else:
        num_parameter = int(num_all/(task_num-1))
        if task_index != task_num-1:
            parameter_array = parameter_array_all[task_index*num_parameter:(task_index+1)*num_parameter]
        else:
            parameter_array = parameter_array_all[task_index*num_parameter:num_all]
    return parameter_array

# 创建一个sh文件用于提交任务
def make_sh_file(sh_filename='a', command_line='python a.py', cpu_num=1, task_name='task', cd_dir=0):
    sh_content = \
        '#!/bin/sh\n' \
        +'#PBS -N '+task_name+'\n' \
        +'#PBS -l nodes=1:ppn='+str(cpu_num)+'\n'
    if cd_dir==1:
        sh_content += 'cd $PBS_O_WORKDIR\n'
    sh_content += command_line
    with open(sh_filename+'.sh', 'w') as f:
        f.write(sh_content)

# 复制.py和.sh文件，然后提交任务，实现半手动并行
def copy_py_sh_file_and_qsub_task(parameter_array, py_filename='a', old_str_in_py='parameter=0', new_str_in_py='parameter=', sh_filename='a', qsub_task_name='task'):
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
        old_str = qsub_task_name
        new_str = qsub_task_name+'_'+str(index)
        content = content.replace(old_str, new_str)
        with open(sh_filename+'_'+str(index)+'.sh', 'w') as f: 
            f.write(content)
        # qsub task
        os.system('qsub '+new_file)

# 自动先后运行程序
def run_programs_sequentially(program_files=['./a.py', './b.py'], execute='python ', show_time=0):
    import os
    import time
    if show_time == 1:
        start = time.time()
    i0 = 0
    for program_file in program_files:
        i0 += 1
        if show_time == 1:
            start_0 = time.time()
        os.system(execute+program_file)
        if show_time == 1:
            end_0 = time.time()
            print('Running time of program_'+str(i0)+' = '+str((end_0-start_0)/60)+' min')
    if show_time == 1:
        end = time.time()
        print('Total running time = '+str((end-start)/60)+' min')

# 将XYZ数据转成矩阵数据（说明：x_array/y_array的输入和输出不一样。要求z_array数据中y对应的数据为小循环，x对应的数据为大循环）
def convert_xyz_data_into_matrix_data(x_array, y_array, z_array):
    import numpy as np
    x_array_input = np.array(x_array)
    y_array_input = np.array(y_array)
    x_array = np.array(list(set(x_array_input)))
    y_array = np.array(list(set(y_array_input)))
    z_array = np.array(z_array)
    len_x = len(x_array)
    len_y = len(y_array)
    matrix = np.zeros((len_x, len_y))
    for ix in range(len_x):
        for iy in range(len_y):
            matrix[ix, iy] = z_array[ix*len_y+iy]
    return x_array, y_array, matrix

# 将矩阵数据转成XYZ数据（说明：x_array/y_array的输入和输出不一样。生成的z_array数据中y对应的数据为小循环，x对应的数据为大循环）
def convert_matrix_data_into_xyz_data(x_array, y_array, matrix):
    import numpy as np
    x_array_input = np.array(x_array)
    y_array_input = np.array(y_array)
    matrix = np.array(matrix)
    len_x = len(x_array_input)
    len_y = len(y_array_input)
    x_array = np.zeros((len_x*len_y))
    y_array = np.zeros((len_x*len_y))
    z_array = np.zeros((len_x*len_y))
    for ix in range(len_x):
        for iy in range(len_y):
            x_array[ix*len_y+iy] = x_array_input[ix]
            y_array[ix*len_y+iy] = y_array_input[iy]
            z_array[ix*len_y+iy] = matrix[ix, iy]
    return x_array, y_array, z_array

# 通过定义计算R^2（基于实际值和预测值）
def calculate_R2_with_definition(y_true_array, y_pred_array):
    import numpy as np
    y_mean = np.mean(y_true_array)
    SS_tot = np.sum((y_true_array - y_mean) ** 2)
    SS_res = np.sum((y_true_array - y_pred_array) ** 2)
    R2 = 1 - (SS_res / SS_tot)
    return R2

# 通过sklearn计算R^2，和定义的计算结果一致
def calculate_R2_with_sklearn(y_true_array, y_pred_array):
    from sklearn.metrics import r2_score
    R2 = r2_score(y_true_array, y_pred_array)
    return R2

# 通过scipy计算线性回归后的R^2（基于线性回归模型）
def calculate_R2_after_linear_regression_with_scipy(y_true_array, y_pred_array):
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_array, y_pred_array)
    R2 = r_value**2
    return R2

# 获取函数或类的源码（返回字符串）
def get_source(name):
    import inspect
    source = inspect.getsource(name)
    return source

# 判断一个数是否接近于整数
def close_to_integer(value, abs_tol=1e-3):
    import math
    result = math.isclose(value, round(value), abs_tol=abs_tol)
    return result

# 根据子数组的第index个元素对子数组进行排序（index从0开始）
def sort_array_by_index_element(original_array, index):
    sorted_array = sorted(original_array, key=lambda x: x[index])
    return sorted_array

# 随机获得一个整数，左闭右闭
def get_random_number(start=0, end=1):
    import random
    rand_number = random.randint(start, end)   # 左闭右闭 [start, end]
    return rand_number

# 选取一个种子生成固定的随机整数，左闭右开
def generate_random_int_number_for_a_specific_seed(seed=0, x_min=0, x_max=10):
    import numpy as np
    np.random.seed(seed)
    rand_num = np.random.randint(x_min, x_max) # 左闭右开[x_min, x_max)
    return rand_num

# 获取两个模式之间的字符串
def get_string_between_two_patterns(original_string, start, end, include_start_and_end=0):
    import re
    pattern = f'{start}(.*?){end}'
    result = re.search(pattern, original_string)
    if result:
        if include_start_and_end == 0:
            return result.group(1)
        else:
            return start+result.group(1)+end
    else:
        return ''

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

# 使用jieba软件包进行分词
def divide_text_into_words(text):
    import jieba
    words = jieba.lcut(text)
    return words

# 根据一定的字符长度来分割文本
def split_text(text, wrap_width=3000):  
    import textwrap  
    split_text_list = textwrap.wrap(text, wrap_width)
    return split_text_list

# 判断某个字符是中文还是英文或其他
def check_Chinese_or_English(a):  
    if '\u4e00' <= a <= '\u9fff' :  
        word_type = 'Chinese'  
    elif '\x00' <= a <= '\xff':  
        word_type = 'English'
    else:
        word_type = 'Others' 
    return word_type

# 统计中英文文本的字数，默认不包括空格
def count_words(text, include_space=0, show_words=0):
    import jieba
    import guan
    words = jieba.lcut(text)  
    new_words = []
    if include_space == 0:
        for word in words:
            if word != ' ':
                new_words.append(word)
    else:
        new_words = words
    num_words = 0
    new_words_2 = []
    for word in new_words:
        word_type = guan.check_Chinese_or_English(word[0])
        if word_type == 'Chinese':
            num_words += len(word)
            for one_word in word:
                new_words_2.append(one_word)
        elif word_type == 'English' or 'Others':
            num_words += 1
            new_words_2.append(word)
    if show_words == 1:
        print(new_words_2)
    return num_words

# 将RGB转成HEX
def rgb_to_hex(rgb, pound=1):
    if pound==0:
        return '%02x%02x%02x' % rgb
    else:
        return '#%02x%02x%02x' % rgb

# 将HEX转成RGB
def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    length = len(hex)
    return tuple(int(hex[i:i+length//3], 16) for i in range(0, length, length//3))

# 使用MD5进行散列加密
def encryption_MD5(password, salt=''):
    import hashlib
    password = salt+password
    hashed_password = hashlib.md5(password.encode()).hexdigest()
    return hashed_password

# 使用SHA-256进行散列加密
def encryption_SHA_256(password, salt=''):
    import hashlib
    password = salt+password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    return hashed_password