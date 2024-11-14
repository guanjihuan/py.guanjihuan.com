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
        client_socket.send(send_message.encode('utf-8'))
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

# 创建一个sh文件用于提交任务（LSF）
def make_sh_file_for_bsub(sh_filename='a', command_line='python a.py', cpu_num=1, task_name='task', queue_name='score', cd_dir=0):
    sh_content = \
        '#!/bin/sh\n' \
        +'#BSUB -J '+task_name+'\n' \
        +'#BSUB -q '+queue_name+'\n' \
        +'#BSUB -n '+str(cpu_num)+'\n'
    if cd_dir==1:
        sh_content += 'cd $PBS_O_WORKDIR\n'
    sh_content += command_line
    with open(sh_filename+'.sh', 'w') as f:
        f.write(sh_content)

# 复制.py和.sh文件，然后提交任务，实现半手动并行（PBS）
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

# 复制.py和.sh文件，然后提交任务，实现半手动并行（LSF）
def copy_py_sh_file_and_bsub_task(parameter_array, py_filename='a', old_str_in_py='parameter=0', new_str_in_py='parameter=', sh_filename='a', bsub_task_name='task'):
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
        old_str = bsub_task_name
        new_str = bsub_task_name+'_'+str(index)
        content = content.replace(old_str, new_str)
        with open(sh_filename+'_'+str(index)+'.sh', 'w') as f: 
            f.write(content)
        # bsub task
        os.system('bsub < '+new_file)

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

# 通过定义计算R^2（基于实际值和预测值，数值有可能小于0）
def calculate_R2_with_definition(y_true_array, y_pred_array):
    import numpy as np
    y_mean = np.mean(y_true_array)
    SS_tot = np.sum((y_true_array - y_mean) ** 2)
    SS_res = np.sum((y_true_array - y_pred_array) ** 2)
    R2 = 1 - (SS_res / SS_tot)
    return R2

# 通过sklearn计算R^2，和上面定义的计算结果一致
def calculate_R2_with_sklearn(y_true_array, y_pred_array):
    from sklearn.metrics import r2_score
    R2 = r2_score(y_true_array, y_pred_array)
    return R2

# 通过scipy计算线性回归后的R^2（基于线性回归模型，范围在0和1之间）
def calculate_R2_after_linear_regression_with_scipy(y_true_array, y_pred_array):
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_array, y_pred_array)
    R2 = r_value**2
    return R2

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

# 获取当前日期字符串
def get_date(bar=True):
    import datetime
    datetime_date = str(datetime.date.today())
    if bar==False:
        datetime_date = datetime_date.replace('-', '')
    return datetime_date

# 获取当前时间字符串
def get_time(colon=True):
    import datetime
    datetime_time = datetime.datetime.now().strftime('%H:%M:%S')
    if colon==False:
        datetime_time = datetime_time.replace(':', '')
    return datetime_time

# 获取运行的日期和时间并写入文件
def statistics_with_day_and_time(content='', filename='a', file_format='.txt'):
    import datetime
    datetime_today = str(datetime.date.today())
    datetime_time = datetime.datetime.now().strftime('%H:%M:%S')
    with open(filename+file_format, 'a', encoding="utf-8") as f2:
       if content == '':
           f2.write(datetime_today+' '+datetime_time+'\n')
       else:
           f2.write(datetime_today+' '+datetime_time+' '+content+'\n')

# 获取本月的所有日期
def get_date_array_of_the_current_month(str_or_datetime='str'):
    import datetime
    today = datetime.date.today()
    first_day_of_month = today.replace(day=1)
    if first_day_of_month.month == 12:
        next_month = first_day_of_month.replace(year=first_day_of_month.year + 1, month=1)
    else:
        next_month = first_day_of_month.replace(month=first_day_of_month.month + 1)
    current_date = first_day_of_month
    date_array = []
    while current_date < next_month:
        if str_or_datetime=='str':
            date_array.append(str(current_date))
        elif str_or_datetime=='datetime':
            date_array.append(current_date)
        current_date += datetime.timedelta(days=1)
    return date_array

# 获取上个月份
def get_last_month():
    import datetime
    today = datetime.date.today()
    last_month = today.month - 1
    if last_month == 0:
        last_month = 12
        year_of_last_month = today.year - 1
    else:
        year_of_last_month = today.year
    return year_of_last_month, last_month

# 获取上上个月份
def get_the_month_before_last():
    import datetime
    today = datetime.date.today()
    the_month_before_last = today.month - 2
    if the_month_before_last == 0:
        the_month_before_last = 12 
        year_of_the_month_before_last = today.year - 1
    else:
        year_of_the_month_before_last = today.year
    if the_month_before_last == -1:
        the_month_before_last = 11
        year_of_the_month_before_last = today.year - 1
    else:
        year_of_the_month_before_last = today.year
    return year_of_the_month_before_last, the_month_before_last

# 获取上个月的所有日期
def get_date_array_of_the_last_month(str_or_datetime='str'):
    import datetime
    import guan
    today = datetime.date.today()
    year_of_last_month, last_month = guan.get_last_month()
    first_day_of_month = today.replace(year=year_of_last_month, month=last_month, day=1)
    if first_day_of_month.month == 12:
        next_month = first_day_of_month.replace(year=first_day_of_month.year + 1, month=1)
    else:
        next_month = first_day_of_month.replace(month=first_day_of_month.month + 1)
    current_date = first_day_of_month
    date_array = []
    while current_date < next_month:
        if str_or_datetime=='str':
            date_array.append(str(current_date))
        elif str_or_datetime=='datetime':
            date_array.append(current_date)
        current_date += datetime.timedelta(days=1)
    return date_array

# 获取上上个月的所有日期
def get_date_array_of_the_month_before_last(str_or_datetime='str'):
    import datetime
    import guan
    today = datetime.date.today()
    year_of_last_last_month, last_last_month = guan.get_the_month_before_last()
    first_day_of_month = today.replace(year=year_of_last_last_month, month=last_last_month, day=1)
    if first_day_of_month.month == 12:
        next_month = first_day_of_month.replace(year=first_day_of_month.year + 1, month=1)
    else:
        next_month = first_day_of_month.replace(month=first_day_of_month.month + 1)
    current_date = first_day_of_month
    date_array = []
    while current_date < next_month:
        if str_or_datetime=='str':
            date_array.append(str(current_date))
        elif str_or_datetime=='datetime':
            date_array.append(current_date)
        current_date += datetime.timedelta(days=1)
    return date_array

# 根据新的日期，填充数组中缺少的数据为零
def fill_zero_data_for_new_dates(old_dates, new_dates, old_data_array):
    new_data_array = []
    for date in new_dates:
        if str(date) not in old_dates:
            new_data_array.append(0)
        else:
            index = old_dates.index(date)
            new_data_array.append(old_data_array[index])
    return new_data_array

# 获取CPU使用率
def get_cpu_usage(interval=1):
    import psutil
    cpu_usage = psutil.cpu_percent(interval=interval)
    return cpu_usage

# 获取内存信息
def get_memory_info():
    import psutil
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total/(1024**2)
    used_memory = memory_info.used/(1024**2)
    available_memory = memory_info.available/(1024**2)
    used_memory_percent = memory_info.percent
    return total_memory, used_memory, available_memory, used_memory_percent

# 获取MAC地址
def get_mac_address():
    import uuid
    mac_address = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
    mac_address = '-'.join([mac_address[i:i+2] for i in range(0, 11, 2)])
    return mac_address

# 获取软件包中的所有模块名
def get_all_modules_in_one_package(package_name='guan'):
    import pkgutil
    package = __import__(package_name)
    module_names = [name for _, name, _ in pkgutil.iter_modules(package.__path__)]
    return module_names

# 获取软件包中一个模块的所有函数名
def get_all_functions_in_one_module(module_name, package_name='guan'):
    import inspect
    function_names = []
    module = __import__(f"{package_name}.{module_name}", fromlist=[""])
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            function_names.append(name)
    return function_names

# 获取软件包中的所有函数名
def get_all_functions_in_one_package(package_name='guan', print_show=1):
    import guan
    module_names = guan.get_all_modules_in_one_package(package_name=package_name)
    all_function_names = []
    for module_name in module_names:
        function_names = guan.get_all_functions_in_one_module(module_name, package_name='guan')
        if print_show == 1:
            print('Module:', module_name)
        for name in function_names:
            all_function_names.append(name)
            if print_show == 1:
                print('function:', name)
        if print_show == 1:
            print()
    return all_function_names

# 获取调用本函数的函数名
def get_calling_function_name(layer=1):
    import inspect
    caller = inspect.stack()[layer]
    calling_function_name = caller.function
    return calling_function_name

# 统计Python文件中import的数量并排序
def count_number_of_import_statements(filename, file_format='.py', num=1000):
    with open(filename+file_format, 'r') as file:
        lines = file.readlines()
    import_array = []
    for line in lines:
        if 'import ' in line:
            line = line.strip()
            import_array.append(line)
    from collections import Counter
    import_statement_counter = Counter(import_array).most_common(num)
    return import_statement_counter

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