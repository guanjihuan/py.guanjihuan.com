# Module: data_processing

# 获取运行的日期和时间并写入文件
def logging_with_day_and_time(content='', filename='time_logging', file_format='.txt'):
    import datetime
    datetime_today = str(datetime.date.today())
    datetime_time = datetime.datetime.now().strftime('%H:%M:%S')
    with open(filename+file_format, 'a', encoding="utf-8") as f2:
       if content == '':
           f2.write(datetime_today+' '+datetime_time+'\n')
       else:
           f2.write(datetime_today+' '+datetime_time+' '+str(content)+'\n')

# 使用该函数运行某个函数并获取函数计算时间（秒）
def timer(function_name, *args, **kwargs):
    import time
    start = time.time()
    result = function_name(*args, **kwargs)
    end = time.time()
    print(f"Running time of {function_name.__name__}: {end - start} seconds")
    return result

# 使用该函数运行某个函数并实现 try-except-pass 结构
def try_except(function_name, *args, **kwargs):
    try:
        return function_name(*args, **kwargs)
    except:
        pass

# 使用 multiprocessing.Pool 实现自动分配任务并行
def parallel_calculation_with_multiprocessing_Pool(func, args_list=[1, 2, 3], show_time=0):
    import multiprocessing
    import time
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        result_array = pool.map(func, args_list)
    end_time = time.time()
    if show_time:
        print(end_time - start_time)
    return result_array

# 循环一个参数计算某个函数，并返回计算结果的数组
def loop_calculation_with_one_parameter(function_name, parameter_array):
    import numpy as np
    result_array = []
    for parameter in parameter_array:
        result = function_name(parameter)
        result_array.append(result)
    result_array = np.array(result_array)
    return result_array

# 循环两个参数计算某个函数，并返回计算结果的数组
def loop_calculation_with_two_parameters(function_name, parameter_array_1, parameter_array_2):
    import numpy as np
    result_array = np.zeros((len(parameter_array_2), len(parameter_array_1)))
    i1 = 0
    for parameter_1 in parameter_array_1:
        i2 = 0
        for parameter_2 in parameter_array_2:
            result = function_name(parameter_1, parameter_2)
            result_array[i2, i1] = result
            i2 += 1
        i1 += 1
    return result_array

# 循环三个参数计算某个函数，并返回计算结果的数组
def loop_calculation_with_three_parameters(function_name, parameter_array_1, parameter_array_2, parameter_array_3):
    import numpy as np
    result_array = np.zeros((len(parameter_array_3), len(parameter_array_2), len(parameter_array_1)))
    i1 = 0
    for parameter_1 in parameter_array_1:
        i2 = 0
        for parameter_2 in parameter_array_2:
            i3 = 0
            for parameter_3 in parameter_array_3:
                result = function_name(parameter_1, parameter_2, parameter_3)
                result_array[i3, i2, i1] = result
                i3 += 1
            i2 += 1
        i1 += 1
    return result_array

# 文本对比
def word_diff(a, b, print_show=1):
    import difflib
    import jieba
    import logging
    jieba.setLogLevel(logging.ERROR)
    a_words = jieba.lcut(a)
    b_words = jieba.lcut(b)
    sm = difflib.SequenceMatcher(None, a_words, b_words, autojunk=False)
    result = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            result.extend(a_words[i1:i2])
        elif tag == "delete":
            result.append("\033[9;91m" + "".join(a_words[i1:i2]) + "\033[0m")
        elif tag == "insert":
            result.append("\033[92m" + "".join(b_words[j1:j2]) + "\033[0m")
        elif tag == "replace":
            result.append("\033[9;91m" + "".join(a_words[i1:i2]) + "\033[0m")
            result.append(" ")
            result.append("\033[92m" + "".join(b_words[j1:j2]) + "\033[0m")
    diff_result = "".join(result)
    if print_show:
        print(diff_result)
    return diff_result

# 文本对比（写入HTML文件）
def word_diff_to_html(a, b, filename='diff_result', write_file=1):
    import difflib
    from html import escape
    import jieba
    import logging
    jieba.setLogLevel(logging.ERROR)
    a_words = jieba.lcut(a)
    b_words = jieba.lcut(b)
    sm = difflib.SequenceMatcher(None, a_words, b_words, autojunk=False)
    html_parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            html_parts.append("".join(map(escape, a_words[i1:i2])))
        elif tag == "delete":
            html_parts.append(f"<span style='background:#e74c3c;color:white;padding:1px 2px;border-radius:2px;text-decoration:line-through;'>"
                        + "".join(map(escape, a_words[i1:i2]))
                        + "</span>")
        elif tag == "insert":
            html_parts.append(f"<span style='background:#2ecc71;color:white;padding:1px 2px;border-radius:2px;'>"
                        + "".join(map(escape, b_words[j1:j2]))
                        + "</span>")
        elif tag == "replace":
            html_parts.append(f"<span style='background:#e74c3c;color:white;padding:1px 2px;border-radius:2px;text-decoration:line-through;'>"
                        + "".join(map(escape, a_words[i1:i2]))
                        + "</span>")
            html_parts.append(" ")
            html_parts.append(f"<span style='background:#2ecc71;color:white;padding:1px 2px;border-radius:2px;'>"
                        + "".join(map(escape, b_words[j1:j2]))
                        + "</span>")
    diff_result = "".join(html_parts)
    diff_result = diff_result.replace("\n", "<br>")
    if write_file:
        with open(filename+'.html', 'w', encoding='UTF-8') as f:
            f.write(diff_result)
    return diff_result

# 打印数组
def print_array(array, line_break=0):
    if line_break == 0:
        for i0 in array:
            print(i0)
    else:
        for i0 in array:
            print(i0)
            print()

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

# 获取矩阵的维度（考虑单一数值的矩阵维度为1）
def dimension_of_array(array):
    import numpy as np
    array = np.array(array)
    if array.shape==():
        dim = 1
    else:
        dim = array.shape[0]
    return dim

# 检查矩阵是否为厄米矩阵（相对误差为1e-5)
def is_hermitian(matrix):
    import numpy as np
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, np.conj(matrix.T))

# 判断一个数是否接近于整数
def close_to_integer(value, abs_tol=1e-3):
    import math
    result = math.isclose(value, round(value), abs_tol=abs_tol)
    return result

# 从列表中删除某个匹配的元素
def remove_item_in_one_array(array, item):
    new_array = [x for x in array if x != item]
    return new_array 

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

# ​使用Numpy库计算​总体标准差
def standard_deviation(data_array):
    import numpy as np
    std_result = np.std(data_array)
    return std_result

# ​​使用公式计算总体标准差
def standard_deviation_with_formula(data_array):
    import numpy as np
    averaged_data = sum(data_array)/len(data_array)
    averaged_squared_data = sum(np.array(data_array)**2)/len(data_array)
    std_result = np.sqrt(averaged_squared_data-averaged_data**2)
    return std_result

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
    
# 删除某个字符串中两个模式之间的内容，返回新字符串
def remove_substrings(original_string, start, end):
    import re
    escaped_start = re.escape(start)
    escaped_end = re.escape(end)
    pattern = f'{escaped_start}.*?{escaped_end}'
    return re.sub(pattern, '', original_string, flags=re.DOTALL)

# 获取旋转矩阵（输入为角度）
def get_rotation_matrix(angle_deg):
    import numpy as np
    angle_rad = np.radians(angle_deg)
    matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
    return matrix

# 旋转某个点，返回新的点的坐标
def rotate_point(x, y, angle_deg):
    import numpy as np
    rotation_matrix = get_rotation_matrix(angle_deg)
    x, y = np.dot(rotation_matrix, np.array([x, y]))
    return x, y

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

# 根据 “.” 和 “。” 符号进行分句
def split_text_into_sentences(text):
    import re
    pattern = r'(?<=[。])|(?<=\.)(?=\s|$)'
    sentences = re.split(pattern, text)
    sentence_array = [s.strip() for s in sentences if s.strip()]
    return sentence_array

# 根据一定的字符长度来分割文本
def split_text(text, width=100):  
    split_text_list = [text[i:i+width] for i in range(0, len(text), width)]
    return split_text_list

# 使用textwrap根据一定的字符长度来分割文本（会自动微小调节宽度，但存在换行符和空格丢失的问题）
def split_text_with_textwrap(text, width=100):  
    import textwrap
    split_text_list = textwrap.wrap(text, width)
    return split_text_list

# 使用jieba软件包进行分词
def divide_text_into_words(text):
    import jieba
    words = jieba.lcut(text)
    return words

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

# 获取函数或类的源码（返回字符串）
def get_source(name):
    import inspect
    source = inspect.getsource(name)
    return source

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
    hashed_password = hashlib.md5(password.encode('utf-8')).hexdigest()
    return hashed_password

# 使用SHA-256进行散列加密（常用且相对比较安全）
def encryption_SHA_256(password, salt=''):
    import hashlib
    password = salt+password
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return hashed_password

# 使用bcrypt生成盐并加密（常用且更加安全）
def encryption_bcrypt(password):
    import bcrypt
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

# 验证bcrypt加密的密码（这里的hashed_password已经包含了生成时使用的盐，bcrypt.checkpw会自动从hashed_password中提取盐，因此在验证时无需再单独传递盐）
def check_bcrypt_hashed_password(password_input, hashed_password):
    import bcrypt
    return bcrypt.checkpw(password_input.encode('utf-8'), hashed_password)