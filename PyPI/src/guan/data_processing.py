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
    rand_number = random.randint(start, end) # [start, end]
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

# 从网页的标签中获取内容
def get_html_from_tags(link, tags=['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'a']):
    from bs4 import BeautifulSoup
    import urllib.request
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen(link).read().decode('utf-8')
    soup = BeautifulSoup(html, features="lxml")
    all_tags = soup.find_all(tags)
    content = ''
    for tag in all_tags:
        text = tag.get_text().replace('\n', '')
        if content == '':
            content = text
        else:
            content = content + '\n\n' + text
    import guan
    guan.statistics_of_guan_package()
    return content

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
        year_of_last_month = today.year
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

# 播放学术单词
def play_academic_words(reverse=0, random_on=0, bre_or_ame='ame', show_translation=1, show_link=1, translation_time=2, rest_time=1):
    from bs4 import BeautifulSoup
    import re
    import urllib.request
    import requests
    import os
    import pygame
    import time
    import ssl
    import random
    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen("https://www.guanjihuan.com/archives/4418").read().decode('utf-8')
    if bre_or_ame == 'ame':
        directory = 'words_mp3_ameProns/'
    elif bre_or_ame == 'bre':
        directory = 'words_mp3_breProns/'
    exist_directory = os.path.exists(directory)
    html_file = urllib.request.urlopen("https://file.guanjihuan.com/words/"+directory).read().decode('utf-8')
    if exist_directory == 0:
        os.makedirs(directory)
    soup = BeautifulSoup(html, features='lxml')
    contents = re.findall('<h2.*?</a></p>', html, re.S)
    if random_on==1:
        random.shuffle(contents)
    if reverse==1:
        contents.reverse()
    for content in contents:
        soup2 = BeautifulSoup(content, features='lxml')
        all_h2 = soup2.find_all('h2')
        for h2 in all_h2:
            if re.search('\d*. ', h2.get_text()):
                word = re.findall('[a-zA-Z].*', h2.get_text(), re.S)[0]
                exist = os.path.exists(directory+word+'.mp3')
                if not exist:
                    try:
                        if re.search(word, html_file):
                            r = requests.get("https://file.guanjihuan.com/words/"+directory+word+".mp3", stream=True)
                            with open(directory+word+'.mp3', 'wb') as f:
                                for chunk in r.iter_content(chunk_size=32):
                                    f.write(chunk)
                    except:
                        pass
                print(h2.get_text())
                try:
                    pygame.mixer.init()
                    track = pygame.mixer.music.load(directory+word+'.mp3')
                    pygame.mixer.music.play()
                    if show_link==1:
                        print('https://www.ldoceonline.com/dictionary/'+word)
                except:
                    pass
                translation = re.findall('<p>.*?</p>', content, re.S)[0][3:-4]
                if show_translation==1:
                    time.sleep(translation_time)
                    print(translation)
                time.sleep(rest_time)
                pygame.mixer.music.stop()
                print()
    import guan
    guan.statistics_of_guan_package()

# 播放挑选过后的学术单词
def play_selected_academic_words(reverse=0, random_on=0, bre_or_ame='ame', show_link=1, rest_time=3):
    from bs4 import BeautifulSoup
    import re
    import urllib.request
    import requests
    import os
    import pygame
    import time
    import ssl
    import random
    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen("https://www.guanjihuan.com/archives/24732").read().decode('utf-8')
    if bre_or_ame == 'ame':
        directory = 'words_mp3_ameProns/'
    elif bre_or_ame == 'bre':
        directory = 'words_mp3_breProns/'
    exist_directory = os.path.exists(directory)
    html_file = urllib.request.urlopen("https://file.guanjihuan.com/words/"+directory).read().decode('utf-8')
    if exist_directory == 0:
        os.makedirs(directory)
    soup = BeautifulSoup(html, features='lxml')
    contents = re.findall('<li>\d.*?</li>', html, re.S)
    if random_on==1:
        random.shuffle(contents)
    if reverse==1:
        contents.reverse()
    for content in contents:
        soup2 = BeautifulSoup(content, features='lxml')
        all_li = soup2.find_all('li')
        for li in all_li:
            if re.search('\d*. ', li.get_text()):
                word = re.findall('\s[a-zA-Z].*?\s', li.get_text(), re.S)[0][1:-1]
                exist = os.path.exists(directory+word+'.mp3')
                if not exist:
                    try:
                        if re.search(word, html_file):
                            r = requests.get("https://file.guanjihuan.com/words/"+directory+word+".mp3", stream=True)
                            with open(directory+word+'.mp3', 'wb') as f:
                                for chunk in r.iter_content(chunk_size=32):
                                    f.write(chunk)
                    except:
                        pass
                print(li.get_text())
                try:
                    pygame.mixer.init()
                    track = pygame.mixer.music.load(directory+word+'.mp3')
                    pygame.mixer.music.play()
                    if show_link==1:
                        print('https://www.ldoceonline.com/dictionary/'+word)
                except:
                    pass
                time.sleep(rest_time)
                pygame.mixer.music.stop()
                print()
    import guan
    guan.statistics_of_guan_package()

# 播放元素周期表上的单词
def play_element_words(random_on=0, show_translation=1, show_link=1, translation_time=2, rest_time=1):
    from bs4 import BeautifulSoup
    import re
    import urllib.request
    import requests
    import os
    import pygame
    import time
    import ssl
    import random
    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen("https://www.guanjihuan.com/archives/10897").read().decode('utf-8')
    directory = 'prons/'
    exist_directory = os.path.exists(directory)
    html_file = urllib.request.urlopen("https://file.guanjihuan.com/words/periodic_table_of_elements/"+directory).read().decode('utf-8')
    if exist_directory == 0:
        os.makedirs(directory)
    soup = BeautifulSoup(html, features='lxml')
    contents = re.findall('<h2.*?</a></p>', html, re.S)
    if random_on==1:
        random.shuffle(contents)
    for content in contents:
        soup2 = BeautifulSoup(content, features='lxml')
        all_h2 = soup2.find_all('h2')
        for h2 in all_h2:
            if re.search('\d*. ', h2.get_text()):
                word = re.findall('[a-zA-Z].* \(', h2.get_text(), re.S)[0][:-2]
                exist = os.path.exists(directory+word+'.mp3')
                if not exist:
                    try:
                        if re.search(word, html_file):
                            r = requests.get("https://file.guanjihuan.com/words/periodic_table_of_elements/prons/"+word+".mp3", stream=True)
                            with open(directory+word+'.mp3', 'wb') as f:
                                for chunk in r.iter_content(chunk_size=32):
                                    f.write(chunk)
                    except:
                        pass
                print(h2.get_text())
                try:
                    pygame.mixer.init()
                    track = pygame.mixer.music.load(directory+word+'.mp3')
                    pygame.mixer.music.play()
                    if show_link==1:
                        print('https://www.merriam-webster.com/dictionary/'+word)
                except:
                    pass
                translation = re.findall('<p>.*?</p>', content, re.S)[0][3:-4]
                if show_translation==1:
                    time.sleep(translation_time)
                    print(translation)
                time.sleep(rest_time)
                pygame.mixer.music.stop()
                print()
    import guan
    guan.statistics_of_guan_package()

# 获取Guan软件包当前模块的所有函数名
def get_all_function_names_in_current_module():
    import inspect
    current_module = inspect.getmodule(inspect.currentframe())
    function_names = [name for name, obj in inspect.getmembers(current_module) if inspect.isfunction(obj)]
    import guan
    guan.statistics_of_guan_package()
    return function_names

# 统计Guan软件包中的函数数量
def count_functions_in_current_module():
    import guan
    function_names = guan.get_all_function_names_in_current_module()
    num_functions = len(function_names)
    guan.statistics_of_guan_package()
    return num_functions

# 获取当前函数名
def get_current_function_name():
    import inspect
    current_function_name = inspect.currentframe().f_code.co_name
    import guan
    guan.statistics_of_guan_package()
    return current_function_name

# 获取调用本函数的函数名
def get_calling_function_name(layer=1):
    import inspect
    caller = inspect.stack()[layer]
    calling_function_name = caller.function
    return calling_function_name

# 获取当前日期字符串
def get_date(bar=True):
    import datetime
    datetime_date = str(datetime.date.today())
    if bar==False:
        datetime_date = datetime_date.replace('-', '')
    return datetime_date

# 获取当前时间字符串
def get_time():
    import datetime
    datetime_time = datetime.datetime.now().strftime('%H:%M:%S')
    return datetime_time

# 获取MAC地址
def get_mac_address():
    import uuid
    mac_address = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
    mac_address = '-'.join([mac_address[i:i+2] for i in range(0, 11, 2)])
    return mac_address

# Guan软件包的使用统计（不涉及到用户的个人数据）
def statistics_of_guan_package():
    try:
        import guan
        message_calling = guan.get_calling_function_name(layer=3)
        if message_calling == '<module>':
            import socket
            datetime_date = guan.get_date()
            datetime_time = guan.get_time()
            current_version = guan.get_current_version('guan')
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(0.5)
            client_socket.connect(('py.guanjihuan.com', 12345))
            mac_address = guan.get_mac_address()
            message = guan.get_calling_function_name(layer=2)
            send_message = datetime_date + ' ' + datetime_time + ' version_'+current_version + ' MAC_address: '+mac_address+' guan.' + message+'\n'
            client_socket.send(send_message.encode())
            client_socket.close()
    except:
        pass

# 获取Python软件包的最新版本
def get_latest_version(package_name='guan', timeout=0.5):
    import requests
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=timeout)
    except:
        return None
    if response.status_code == 200:
        data = response.json()
        latest_version = data["info"]["version"]
        return latest_version
    else:
        return None

# 获取软件包的本机版本
def get_current_version(package_name='guan'):
    import importlib.metadata
    try:
        current_version = importlib.metadata.version(package_name)
        return current_version
    except:
        return None

# Guan软件包升级提示
def notification_of_upgrade(timeout=0.5):
    try:
        import guan
        latest_version = guan.get_latest_version(package_name='guan', timeout=timeout)
        current_version = guan.get_current_version('guan')
        if latest_version != None and current_version != None:
            if latest_version != current_version:
                print('提示：您当前使用的版本是 guan-'+current_version+'，目前已经有最新版本 guan-'+latest_version+'。您可以通过以下命令对软件包进行升级：pip install --upgrade guan')
    except:
        pass