# Module: data_processing

# 并行计算前的预处理，把参数分成多份
def preprocess_for_parallel_calculations(parameter_array_all, cpus=1, task_index=0):
    import numpy as np
    num_all = np.array(parameter_array_all).shape[0]
    if num_all%cpus == 0:
        num_parameter = int(num_all/cpus) 
        parameter_array = parameter_array_all[task_index*num_parameter:(task_index+1)*num_parameter]
    else:
        num_parameter = int(num_all/(cpus-1))
        if task_index != cpus-1:
            parameter_array = parameter_array_all[task_index*num_parameter:(task_index+1)*num_parameter]
        else:
            parameter_array = parameter_array_all[task_index*num_parameter:num_all]
    import guan
    guan.statistics_of_guan_package()
    return parameter_array

# 在一组数据中找到数值相近的数
def find_close_values_in_one_array(array, precision=1e-2):
    new_array = []
    i0 = 0
    for a1 in array:
        j0 = 0
        for a2 in array:
            if j0>i0 and abs(a1-a2)<precision: 
                new_array.append([a1, a2])
            j0 +=1
        i0 += 1
    import guan
    guan.statistics_of_guan_package()
    return new_array

# 寻找能带的简并点
def find_degenerate_points(k_array, eigenvalue_array, precision=1e-2):
    import guan
    degenerate_k_array = []
    degenerate_eigenvalue_array = []
    i0 = 0
    for k in k_array:
        degenerate_points = guan.find_close_values_in_one_array(eigenvalue_array[i0], precision=precision)
        if len(degenerate_points) != 0:
            degenerate_k_array.append(k)
            degenerate_eigenvalue_array.append(degenerate_points)
        i0 += 1
    import guan
    guan.statistics_of_guan_package()
    return degenerate_k_array, degenerate_eigenvalue_array

# 随机获得一个整数，左闭右闭
def get_random_number(start=0, end=1):
    import random
    rand_number = random.randint(start, end)   # 左闭右闭 [start, end]
    import guan
    guan.statistics_of_guan_package()
    return rand_number

# 选取一个种子生成固定的随机整数
def generate_random_int_number_for_a_specific_seed(seed=0, x_min=0, x_max=10):
    import numpy as np
    np.random.seed(seed)
    rand_num = np.random.randint(x_min, x_max) # 左闭右开[x_min, x_max)
    import guan
    guan.statistics_of_guan_package()
    return rand_num

# 使用jieba分词
def divide_text_into_words(text):
    import jieba
    words = jieba.lcut(text)
    import guan
    guan.statistics_of_guan_package()
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
    import guan
    guan.statistics_of_guan_package()
    return num_words

# 统计运行的日期和时间，写进文件
def statistics_with_day_and_time(content='', filename='a', file_format='.txt'):
    import datetime
    datetime_today = str(datetime.date.today())
    datetime_time = datetime.datetime.now().strftime('%H:%M:%S')
    with open(filename+file_format, 'a', encoding="utf-8") as f2:
       if content == '':
           f2.write(datetime_today+' '+datetime_time+'\n')
       else:
           f2.write(datetime_today+' '+datetime_time+' '+content+'\n')
    import guan
    guan.statistics_of_guan_package()

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
    import guan
    guan.statistics_of_guan_package()
    return import_statement_counter

# 根据一定的字符长度来分割文本
def split_text(text, wrap_width=3000):  
    import textwrap  
    split_text_list = textwrap.wrap(text, wrap_width)
    import guan
    guan.statistics_of_guan_package()
    return split_text_list

# 将RGB转成HEX
def rgb_to_hex(rgb, pound=1):
    import guan
    guan.statistics_of_guan_package()
    if pound==0:
        return '%02x%02x%02x' % rgb
    else:
        return '#%02x%02x%02x' % rgb

# 将HEX转成RGB
def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    length = len(hex)
    import guan
    guan.statistics_of_guan_package()
    return tuple(int(hex[i:i+length//3], 16) for i in range(0, length, length//3))

# 使用MD5进行散列加密
def encryption_MD5(password, salt=''):
    import hashlib
    password = salt+password
    hashed_password = hashlib.md5(password.encode()).hexdigest()
    import guan
    guan.statistics_of_guan_package()
    return hashed_password

# 使用SHA-256进行散列加密
def encryption_SHA_256(password, salt=''):
    import hashlib
    password = salt+password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    import guan
    guan.statistics_of_guan_package()
    return hashed_password

# 获取CPU使用率
def get_cpu_usage(interval=1):
    import psutil
    cpu_usage = psutil.cpu_percent(interval=interval)
    import guan
    guan.statistics_of_guan_package()
    return cpu_usage

# 获取本月的所有日期
def get_days_of_the_current_month(str_or_datetime='str'):
    import datetime
    today = datetime.date.today()
    first_day_of_month = today.replace(day=1)
    if first_day_of_month.month == 12:
        next_month = first_day_of_month.replace(year=first_day_of_month.year + 1, month=1)
    else:
        next_month = first_day_of_month.replace(month=first_day_of_month.month + 1)
    current_date = first_day_of_month
    day_array = []
    while current_date < next_month:
        if str_or_datetime=='str':
            day_array.append(str(current_date))
        elif str_or_datetime=='datetime':
            day_array.append(current_date)
        current_date += datetime.timedelta(days=1)
    import guan
    guan.statistics_of_guan_package()
    return day_array

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
    import guan
    guan.statistics_of_guan_package()
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
    import guan
    guan.statistics_of_guan_package()
    return year_of_the_month_before_last, the_month_before_last

# 获取上个月的所有日期
def get_days_of_the_last_month(str_or_datetime='str'):
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
    day_array = []
    while current_date < next_month:
        if str_or_datetime=='str':
            day_array.append(str(current_date))
        elif str_or_datetime=='datetime':
            day_array.append(current_date)
        current_date += datetime.timedelta(days=1)
    guan.statistics_of_guan_package()
    return day_array

# 获取上上个月的所有日期
def get_days_of_the_month_before_last(str_or_datetime='str'):
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
    day_array = []
    while current_date < next_month:
        if str_or_datetime=='str':
            day_array.append(str(current_date))
        elif str_or_datetime=='datetime':
            day_array.append(current_date)
        current_date += datetime.timedelta(days=1)
    guan.statistics_of_guan_package()
    return day_array

# 获取所有股票
def all_stocks():
    import numpy as np
    import akshare as ak
    stocks = ak.stock_zh_a_spot_em()
    title = np.array(stocks.columns)
    stock_data = stocks.values
    import guan
    guan.statistics_of_guan_package()
    return title, stock_data

# 获取所有股票的代码
def all_stock_symbols():
    import guan
    title, stock_data = guan.all_stocks()
    stock_symbols = stock_data[:, 1]
    guan.statistics_of_guan_package()
    return stock_symbols

# 从股票代码获取股票名称
def find_stock_name_from_symbol(symbol='000002'):
    import guan
    title, stock_data = guan.all_stocks()
    for stock in stock_data:
        if symbol in stock:
           stock_name = stock[2]
    guan.statistics_of_guan_package()
    return stock_name

# 获取单个股票的历史数据
def history_data_of_one_stock(symbol='000002', period='daily', start_date="19000101", end_date='21000101'):
    # period = 'daily'
    # period = 'weekly'
    # period = 'monthly'
    import numpy as np
    import akshare as ak
    stock = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date)
    title = np.array(stock.columns)
    stock_data = stock.values[::-1]
    import guan
    guan.statistics_of_guan_package()
    return title, stock_data

# 获取软件包中的所有模块名
def get_all_modules_in_one_package(package_name='guan'):
    import pkgutil
    package = __import__(package_name)
    module_names = [name for _, name, _ in pkgutil.iter_modules(package.__path__)]
    import guan
    guan.statistics_of_guan_package()
    return module_names

# 获取软件包中一个模块的所有函数名
def get_all_functions_in_one_module(module_name, package_name='guan'):
    import inspect
    function_names = []
    module = __import__(f"{package_name}.{module_name}", fromlist=[""])
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            function_names.append(name)
    import guan
    guan.statistics_of_guan_package()
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
    guan.statistics_of_guan_package()
    return all_function_names

def get_PID(name):
    import subprocess
    command = "ps -ef | grep "+name
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        ps_ef = result.stdout
    import re
    ps_ef = re.split(r'\s+', ps_ef)
    id_running = ps_ef[1]
    import guan
    guan.statistics_of_guan_package()
    return id_running

# 在服务器上运行大语言模型，通过Python函数调用
def chat(prompt='你好', stream_show=1, top_p=0.8, temperature=0.8):
    import socket
    import json
    response = ''
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.settimeout(10)
            client_socket.connect(('socket.guanjihuan.com', 12345))
            message = {
                'server': "chat.guanjihuan.com",
                'prompt': prompt,
                'top_p': top_p,
                'temperature': temperature,
            }
            send_message = json.dumps(message)
            client_socket.send(send_message.encode())
            try:
                while True:
                    try:
                        data = client_socket.recv(1024)
                    except:
                        break
                    stream_response = data.decode()
                    if '连接失败！请过段时间再试或者联系管理员。' in stream_response:
                        print('连接失败！请过段时间再试或者联系管理员。')
                        break
                    elif 'End_response_from_chat.guanjihuan.com.' in stream_response:
                        break
                    elif stream_response == '':
                        break
                    else:
                        if stream_show == 1:
                            print(stream_response)
                            print('\n---\n')
                        response = stream_response
            except:
                pass
            client_socket.close()
    except:
        print('连接失败！请过段时间再试或者联系管理员。')
    import guan
    guan.statistics_of_guan_package()
    return response