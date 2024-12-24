# Module: data_processing

# AI模型对话
def chat(prompt='你好', stream=1, model=1, top_p=0.8, temperature=0.85):
    import socket
    import json
    import time
    import guan
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.settimeout(30)
        client_socket.connect(('socket.guanjihuan.com', 12345))
        split_text_list = guan.split_text(prompt, width=100)
        message_times = len(split_text_list)
        if message_times == 1 or message_times == 0:
            message = {
                'server': "chat.guanjihuan.com",
                'prompt': prompt,
                'model': model,
                'top_p': top_p,
                'temperature': temperature,
            }
            send_message = json.dumps(message)
            client_socket.send(send_message.encode('utf-8'))
        else:
            end_message = 0
            for i0 in range(message_times):
                if i0 == message_times-1:
                    end_message = 1
                prompt_0 = split_text_list[i0]
                message = {
                    'server': "chat.guanjihuan.com",
                    'prompt': prompt_0,
                    'model': model,
                    'top_p': top_p,
                    'temperature': temperature,
                    'end_message': end_message,
                }
                send_message = json.dumps(message)
                client_socket.send(send_message.encode('utf-8'))
                time.sleep(0.15)
        if stream == 1:
            print('\n--- Begin Stream Message ---\n')
        response = ''
        while True:
            if prompt == '':
                break
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
                            print(stream_response, end='', flush=True)
            except:
                break
        client_socket.close()
        if stream == 1:
            print('\n\n--- End Stream Message ---\n')
    return response

# 在云端服务器上运行函数（需要函数是独立可运行的代码）
def run(function_name, *args, **kwargs):
    import socket
    import json
    import pickle
    import base64
    import time
    import guan
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(('run.guanjihuan.com', 12345))
        function_source = guan.get_source(function_name)
        split_text_list = guan.split_text(function_source, width=100)
        message_times = len(split_text_list)
        if message_times == 1 or message_times == 0:
            message = {
                'server': "run.guanjihuan.com",
                'function_name': function_name.__name__,
                'function_source': function_source,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            send_message = json.dumps(message)
            client_socket.send(send_message.encode())
        else:
            end_message = 0
            for i0 in range(message_times):
                if i0 == message_times-1:
                    end_message = 1
                source_0 = split_text_list[i0]
                message = {
                    'server': "run",
                    'function_name': function_name.__name__,
                    'function_source': source_0,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'end_message': end_message,
                }
                send_message = json.dumps(message)
                client_socket.send(send_message.encode())
                time.sleep(0.15)
        print('\nguan.run: 云端服务器正在计算，请等待返回结果。\n')
        return_data = ''
        print_data = ''
        while True:
            try:
                data = client_socket.recv(1024)
                return_text = data.decode()
                return_dict = json.loads(return_text)
                return_data += return_dict['return_data']
                print_data += return_dict['print_data']
                end_message = return_dict['end_message']
                if end_message == 1:
                    break
            except:
                break
        if print_data != '':
            print('--- Start Print ---\n')
            print(print_data)
            print('--- End Print ---\n')
            print('guan.run: 云端服务器计算结束，以上是打印结果。\n')
        else:
            print('guan.run: 云端服务器计算结束。\n')
        return_data = pickle.loads(base64.b64decode(return_data))
        client_socket.close()
    return return_data

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

# 从列表中删除某个匹配的元素
def remove_item_in_one_array(array, item):
    new_array = [x for x in array if x != item]
    return new_array 

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

# 将PDF文件转成文本
def pdf_to_text(pdf_path):
    from pdfminer.pdfparser import PDFParser, PDFDocument
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox
    from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
    import logging 
    logging.Logger.propagate = False 
    logging.getLogger().setLevel(logging.ERROR) 
    praser = PDFParser(open(pdf_path, 'rb'))
    doc = PDFDocument()
    praser.set_document(doc)
    doc.set_parser(praser)
    doc.initialize()
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        content = ''
        for page in doc.get_pages():
            interpreter.process_page(page)                        
            layout = device.get_result()                     
            for x in layout:
                if isinstance(x, LTTextBox):
                    content  = content + x.get_text().strip()
    return content

# 获取PDF文件页数
def get_pdf_page_number(pdf_path):
    import PyPDF2
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    return num_pages

# 获取PDF文件指定页面的内容
def pdf_to_txt_for_a_specific_page(pdf_path, page_num=1):
    import PyPDF2
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    for page_num0 in range(num_pages):
        if page_num0 == page_num-1:
            page = pdf_reader.pages[page_num0]
            page_text = page.extract_text()
    pdf_file.close()
    return page_text

# 获取PDF文献中的链接。例如: link_starting_form='https://doi.org'
def get_links_from_pdf(pdf_path, link_starting_form=''):
    import PyPDF2
    import re
    reader = PyPDF2.PdfReader(pdf_path)
    pages = len(reader.pages)
    i0 = 0
    links = []
    for page in range(pages):
        pageSliced = reader.pages[page]
        pageObject = pageSliced.get_object() 
        if '/Annots' in pageObject.keys():
            ann = pageObject['/Annots']
            old = ''
            for a in ann:
                u = a.get_object() 
                if '/A' in u.keys():
                    if '/URI' in u['/A']: 
                        if re.search(re.compile('^'+link_starting_form), u['/A']['/URI']):
                            if u['/A']['/URI'] != old:
                                links.append(u['/A']['/URI']) 
                                i0 += 1
                                old = u['/A']['/URI']
    return links

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

# 获取内存信息
def get_memory_info():
    import psutil
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total/(1024**2)
    used_memory = memory_info.used/(1024**2)
    available_memory = memory_info.available/(1024**2)
    used_memory_percent = memory_info.percent
    return total_memory, used_memory, available_memory, used_memory_percent

# 获取CPU的平均使用率
def get_cpu_usage(interval=1):
    import psutil
    cpu_usage = psutil.cpu_percent(interval=interval)
    return cpu_usage

# 获取每个CPU核心的使用率，返回列表
def get_cpu_usage_array_per_core(interval=1):
    import psutil
    cpu_usage_array_per_core = psutil.cpu_percent(interval=interval, percpu=True)
    return cpu_usage_array_per_core

# 获取使用率最高的CPU核心的使用率
def get_cpu_max_usage_for_all_cores(interval=1):
    import guan
    cpu_usage_array_per_core = guan.get_cpu_usage_array_per_core(interval=interval)
    max_cpu_usage = max(cpu_usage_array_per_core)
    return max_cpu_usage

# 获取非零使用率的CPU核心的平均使用率
def get_cpu_averaged_usage_for_non_zero_cores(interval=1):
    import guan
    cpu_usage_array_per_core = guan.get_cpu_usage_array_per_core(interval=interval)
    cpu_usage_array_per_core_new = guan.remove_item_in_one_array(cpu_usage_array_per_core, 0.0)
    averaged_cpu_usage = sum(cpu_usage_array_per_core_new)/len(cpu_usage_array_per_core_new)
    return averaged_cpu_usage

# 在一定数量周期内得到CPU的使用率信息。默认为10秒钟收集一次，(interval+sleep_interval)*times 为收集的时间范围，范围默认为60秒，即1分钟后返回列表，总共得到6组数据。其中，数字第一列和第二列分别是平均值和最大值。
def get_cpu_information_for_times(interval=1, sleep_interval=9, times=6):
    import guan
    import time
    cpu_information_array = []
    for _ in range(times):
        cpu_information = []
        datetime_date = guan.get_date()
        datetime_time = guan.get_time()
        cpu_information.append(datetime_date)
        cpu_information.append(datetime_time)
        cpu_usage_array_per_core = guan.get_cpu_usage_array_per_core(interval=interval)
        cpu_information.append(sum(cpu_usage_array_per_core)/len(cpu_usage_array_per_core))
        cpu_information.append(max(cpu_usage_array_per_core))
        for cpu_usage in cpu_usage_array_per_core:
            cpu_information.append(cpu_usage)
        cpu_information_array.append(cpu_information)
        time.sleep(sleep_interval)
    return cpu_information_array

# 将得到的CPU的使用率信息写入文件。默认为半分钟收集一次，(interval+sleep_interval)*times 为收集的时间范围，范围默认为60分钟，即1小时写入文件一次，总共得到120组数据。其中，数字第一列和第二列分别是平均值和最大值。
def write_cpu_information_to_file(filename='./cpu_usage', interval=1, sleep_interval=29, times=120):
    import guan
    guan.make_file(filename+'.txt')
    while True:
        f = guan.open_file(filename)
        cpu_information_array = guan.get_cpu_information_for_times(interval=interval, sleep_interval=sleep_interval, times=times)
        for cpu_information in cpu_information_array:
            i0 = 0
            for information in cpu_information: 
                if i0 < 2:
                    f.write(str(information)+' ')
                else:
                    f.write(f'{information:.1f} ')
                i0 += 1
            f.write('\n')
        f.close()

# 画CPU的使用率图。默认为画最近的120个数据，以及不画CPU核心的最大使用率。
def plot_cpu_information(filename='./cpu_usage', recent_num=120, max_cpu=0):
    import guan
    from datetime import datetime
    with open(filename+".txt", "r") as file:
        lines = file.readlines()
        lines = lines[::-1]
        timestamps_array = []
        averaged_cpu_usage_array = []
        max_cpu_usage_array = []
        i0 = 0
        for line in lines:
            i0 += 1
            if i0 >= recent_num:
                break
            cpu_information = line.strip()
            information = cpu_information.split()
            time_str = information[0]+' '+information[1]
            time_format = "%Y-%m-%d %H:%M:%S"
            timestamps_array.append(datetime.strptime(time_str, time_format))
            averaged_cpu_usage_array.append(float(information[2]))
            max_cpu_usage_array.append(float(information[3]))
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=0.3, adjust_left=0.15, labelsize=16, fontfamily='Times New Roman')
    plt.xticks(rotation=90) 
    guan.plot_without_starting_fig_ax(plt, fig, ax, timestamps_array, averaged_cpu_usage_array, style='o-')
    legend_array = ['Averaged']
    if max_cpu == 1:
        guan.plot_without_starting_fig_ax(plt, fig, ax, timestamps_array, max_cpu_usage_array, style='o-')
        legend_array.append('Max')
    guan.plot_without_starting_fig_ax(plt, fig, ax, [], [], xlabel='Time', ylabel='CPU usage', fontsize=20)
    plt.legend(legend_array)
    plt.show()

# 画详细的CPU的使用率图，分CPU核心画图。
def plot_detailed_cpu_information(filename='./cpu_usage', recent_num=120):
    import guan
    from datetime import datetime
    with open(filename+".txt", "r") as file:
        lines = file.readlines()
        lines = lines[::-1]
        timestamps_array = []
        i0 = 0
        core_num = len(lines[0].strip().split())-4
        detailed_cpu_usage_array = []
        for line in lines:
            i0 += 1
            if i0 > recent_num:
                break
            cpu_information = line.strip()
            information = cpu_information.split()
            time_str = information[0]+' '+information[1]
            time_format = "%Y-%m-%d %H:%M:%S"
            timestamps_array.append(datetime.strptime(time_str, time_format))
            detailed_cpu_usage = []
            for core in range(core_num):
                detailed_cpu_usage.append(float(information[4+core]))
            detailed_cpu_usage_array.append(detailed_cpu_usage)
    for core in range(core_num):
        plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=0.3, adjust_left=0.15, labelsize=16, fontfamily='Times New Roman')
        plt.xticks(rotation=90) 
        guan.plot_without_starting_fig_ax(plt, fig, ax, timestamps_array, [row[core] for row in detailed_cpu_usage_array], style='o-')
        legend_array = []
        legend_array.append(f'CPU {core+1}')
        guan.plot_without_starting_fig_ax(plt, fig, ax, [], [], xlabel='Time', ylabel='CPU usage', fontsize=20)
        plt.legend(legend_array)
        plt.show()

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

# 获取软件包的本机版本
def get_current_version(package_name='guan'):
    import importlib.metadata
    try:
        current_version = importlib.metadata.version(package_name)
        return current_version
    except:
        return None

# 获取Python软件包的最新版本
def get_latest_version(package_name='guan', timeout=5):
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

# 获取包含某个字符的进程PID值
def get_PID_array(name):
    import subprocess
    command = "ps -ef | grep "+name
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        ps_ef = result.stdout
    import re
    ps_ef_1 = re.split(r'\n', ps_ef)
    id_running_array = []
    for ps_ef_item in ps_ef_1:
        if ps_ef_item != '':
            ps_ef_2 = re.split(r'\s+', ps_ef_item)
            id_running_array.append(ps_ef_2[1])
    return id_running_array

# 每日git commit次数的统计
def statistics_of_git_commits(print_show=0, str_or_datetime='str'):
    import subprocess
    import collections
    since_date = '100 year ago'
    result = subprocess.run(
        ['git', 'log', f'--since={since_date}', '--pretty=format:%ad', '--date=short'],
        stdout=subprocess.PIPE,
        text=True)
    commits = result.stdout.strip().split('\n')
    counter = collections.Counter(commits)
    daily_commit_counts = dict(sorted(counter.items()))
    date_array = []
    commit_count_array = []
    for date, count in daily_commit_counts.items():
        if print_show == 1:
            print(f"{date}: {count} commits")
        if str_or_datetime=='datetime':
            import datetime
            date_array.append(datetime.datetime.strptime(date, "%Y-%m-%d"))
        elif str_or_datetime=='str':
            date_array.append(date)
        commit_count_array.append(count)
    return date_array, commit_count_array

# 将文件目录结构写入Markdown文件
def write_file_list_in_markdown(directory='./', filename='a', reverse_positive_or_negative=1, starting_from_h1=None, banned_file_format=[], hide_file_format=None, divided_line=None, show_second_number=None, show_third_number=None): 
    import os
    f = open(filename+'.md', 'w', encoding="utf-8")
    filenames1 = os.listdir(directory)
    u0 = 0
    for filename1 in filenames1[::reverse_positive_or_negative]:
        filename1_with_path = os.path.join(directory,filename1) 
        if os.path.isfile(filename1_with_path):
            if os.path.splitext(filename1)[1] not in banned_file_format:
                if hide_file_format == None:
                    f.write('+ '+str(filename1)+'\n\n')
                else:
                    f.write('+ '+str(os.path.splitext(filename1)[0])+'\n\n')
        else:
            u0 += 1
            if divided_line != None and u0 != 1:
                f.write('--------\n\n')
            if starting_from_h1 == None:
                f.write('#')
            f.write('# '+str(filename1)+'\n\n')

            filenames2 = os.listdir(filename1_with_path) 
            i0 = 0     
            for filename2 in filenames2[::reverse_positive_or_negative]:
                filename2_with_path = os.path.join(directory, filename1, filename2) 
                if os.path.isfile(filename2_with_path):
                    if os.path.splitext(filename2)[1] not in banned_file_format:
                        if hide_file_format == None:
                            f.write('+ '+str(filename2)+'\n\n')
                        else:
                            f.write('+ '+str(os.path.splitext(filename2)[0])+'\n\n')
                else: 
                    i0 += 1
                    if starting_from_h1 == None:
                        f.write('#')
                    if show_second_number != None:
                        f.write('## '+str(i0)+'. '+str(filename2)+'\n\n')
                    else:
                        f.write('## '+str(filename2)+'\n\n')
                    
                    j0 = 0
                    filenames3 = os.listdir(filename2_with_path)
                    for filename3 in filenames3[::reverse_positive_or_negative]:
                        filename3_with_path = os.path.join(directory, filename1, filename2, filename3) 
                        if os.path.isfile(filename3_with_path): 
                            if os.path.splitext(filename3)[1] not in banned_file_format:
                                if hide_file_format == None:
                                    f.write('+ '+str(filename3)+'\n\n')
                                else:
                                    f.write('+ '+str(os.path.splitext(filename3)[0])+'\n\n')
                        else:
                            j0 += 1
                            if starting_from_h1 == None:
                                f.write('#')
                            if show_third_number != None:
                                f.write('### ('+str(j0)+') '+str(filename3)+'\n\n')
                            else:
                                f.write('### '+str(filename3)+'\n\n')

                            filenames4 = os.listdir(filename3_with_path)
                            for filename4 in filenames4[::reverse_positive_or_negative]:
                                filename4_with_path = os.path.join(directory, filename1, filename2, filename3, filename4) 
                                if os.path.isfile(filename4_with_path):
                                    if os.path.splitext(filename4)[1] not in banned_file_format:
                                        if hide_file_format == None:
                                            f.write('+ '+str(filename4)+'\n\n')
                                        else:
                                            f.write('+ '+str(os.path.splitext(filename4)[0])+'\n\n')
                                else: 
                                    if starting_from_h1 == None:
                                        f.write('#')
                                    f.write('#### '+str(filename4)+'\n\n')

                                    filenames5 = os.listdir(filename4_with_path)
                                    for filename5 in filenames5[::reverse_positive_or_negative]:
                                        filename5_with_path = os.path.join(directory, filename1, filename2, filename3, filename4, filename5) 
                                        if os.path.isfile(filename5_with_path): 
                                            if os.path.splitext(filename5)[1] not in banned_file_format:
                                                if hide_file_format == None:
                                                    f.write('+ '+str(filename5)+'\n\n')
                                                else:
                                                    f.write('+ '+str(os.path.splitext(filename5)[0])+'\n\n')
                                        else:
                                            if starting_from_h1 == None:
                                                f.write('#')
                                            f.write('##### '+str(filename5)+'\n\n')

                                            filenames6 = os.listdir(filename5_with_path)
                                            for filename6 in filenames6[::reverse_positive_or_negative]:
                                                filename6_with_path = os.path.join(directory, filename1, filename2, filename3, filename4, filename5, filename6) 
                                                if os.path.isfile(filename6_with_path): 
                                                    if os.path.splitext(filename6)[1] not in banned_file_format:
                                                        if hide_file_format == None:
                                                            f.write('+ '+str(filename6)+'\n\n')
                                                        else:
                                                            f.write('+ '+str(os.path.splitext(filename6)[0])+'\n\n')
                                                else:
                                                    if starting_from_h1 == None:
                                                        f.write('#')
                                                    f.write('###### '+str(filename6)+'\n\n')
    f.close()

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
    return content

# 从HTML中获取所有的链接
def get_links_from_html(html_link, links_with_text=0):
    from bs4 import BeautifulSoup
    import urllib.request
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen(html_link).read().decode('utf-8')
    soup = BeautifulSoup(html, features="lxml")
    a_tags = soup.find_all('a')
    if links_with_text == 0:
        link_array = [tag.get('href') for tag in a_tags if tag.get('href')]
        return link_array
    else:
        link_array_with_text = [(tag.get('href'), tag.text) for tag in a_tags if tag.get('href')]
        return link_array_with_text

# 检查链接的有效性
def check_link(url, timeout=3, allow_redirects=True):
    import requests
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=allow_redirects)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

# 检查链接数组中链接的有效性
def check_link_array(link_array, timeout=3, allow_redirects=True, try_again=0, print_show=1):
    import guan
    failed_link_array0 = []
    for link in link_array:
        if link=='#' or guan.check_link(link, timeout=timeout, allow_redirects=allow_redirects):
            pass
        else:
            failed_link_array0.append(link)
            if print_show:
                print(link)
    failed_link_array = []
    if try_again:
        if print_show:
            print('\nTry again:\n')
        for link in failed_link_array0:
            if link=='#' or guan.check_link(link, timeout=timeout, allow_redirects=allow_redirects):
                pass
            else:
                failed_link_array.append(link)
                if print_show:
                    print(link)
    else:
        failed_link_array = failed_link_array0
    return failed_link_array

# 生成二维码
def creat_qrcode(data="https://www.guanjihuan.com", filename='a', file_format='.png'):
    import qrcode
    img = qrcode.make(data)
    img.save(filename+file_format)

# 通过Sci-Hub网站下载文献
def download_with_scihub(address=None, num=1):
    from bs4 import BeautifulSoup
    import re
    import requests
    import os
    if num==1 and address!=None:
        address_array = [address]
    else:
        address_array = []
        for i in range(num):
            address = input('\nInput：')
            address_array.append(address)
    for address in address_array:
        r = requests.post('https://sci-hub.st/', data={'request': address})
        print('\nResponse：', r)
        print('Address：', r.url)
        soup = BeautifulSoup(r.text, features='lxml')
        pdf_URL = soup.embed['src']
        # pdf_URL = soup.iframe['src'] # This is a code line of history version which fails to get pdf URL.
        if re.search(re.compile('^https:'), pdf_URL):
            pass
        else:
            pdf_URL = 'https:'+pdf_URL
        print('PDF address：', pdf_URL)
        name = re.search(re.compile('fdp.*?/'),pdf_URL[::-1]).group()[::-1][1::]
        print('PDF name：', name)
        print('Directory：', os.getcwd())
        print('\nDownloading...')
        r = requests.get(pdf_URL, stream=True)
        with open(name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=32):
                f.write(chunk)
        print('Completed!\n')
    if num != 1:
        print('All completed!\n')

# 将字符串转成音频
def str_to_audio(str='hello world', filename='str', rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0):
    import pyttsx3
    import guan
    if print_text==1:
        print(str)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  
    engine.setProperty('voice', voices[voice].id)
    engine.setProperty("rate", rate)
    if save==1:
        engine.save_to_file(str, filename+'.wav')
        engine.runAndWait()
        print('Wav file saved!')
        if compress==1:
            import os
            os.rename(filename+'.wav', 'temp.wav')
            guan.compress_wav_to_mp3('temp.wav', output_filename=filename+'.mp3', bitrate=bitrate)
            os.remove('temp.wav')
    if read==1:
        engine.say(str)
        engine.runAndWait()

# 将txt文件转成音频
def txt_to_audio(txt_path, rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0):
    import pyttsx3
    import guan
    f = open(txt_path, 'r', encoding ='utf-8')
    text = f.read()
    if print_text==1:
        print(text)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  
    engine.setProperty('voice', voices[voice].id)
    engine.setProperty("rate", rate)
    if save==1:
        import re
        filename = re.split('[/,\\\]', txt_path)[-1][:-4]
        engine.save_to_file(text, filename+'.wav')
        engine.runAndWait()
        print('Wav file saved!')
        if compress==1:
            import os
            os.rename(filename+'.wav', 'temp.wav')
            guan.compress_wav_to_mp3('temp.wav', output_filename=filename+'.mp3', bitrate=bitrate)
            os.remove('temp.wav')
    if read==1:
        engine.say(text)
        engine.runAndWait()

# 将PDF文件转成音频
def pdf_to_audio(pdf_path, rate=125, voice=1, read=1, save=0, compress=0, bitrate='16k', print_text=0):
    import pyttsx3
    import guan
    text = guan.pdf_to_text(pdf_path)
    text = text.replace('\n', ' ')
    if print_text==1:
        print(text)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')  
    engine.setProperty('voice', voices[voice].id)
    engine.setProperty("rate", rate)
    if save==1:
        import re
        filename = re.split('[/,\\\]', pdf_path)[-1][:-4]
        engine.save_to_file(text, filename+'.wav')
        engine.runAndWait()
        print('Wav file saved!')
        if compress==1:
            import os
            os.rename(filename+'.wav', 'temp.wav')
            guan.compress_wav_to_mp3('temp.wav', output_filename=filename+'.mp3', bitrate=bitrate)
            os.remove('temp.wav')
    if read==1:
        engine.say(text)
        engine.runAndWait()

# 将wav音频文件压缩成MP3音频文件
def compress_wav_to_mp3(wav_path, output_filename='a.mp3', bitrate='16k'):
    # Note: Beside the installation of pydub, you may also need download FFmpeg on http://www.ffmpeg.org/download.html and add the bin path to the environment variable.
    from pydub import AudioSegment
    sound = AudioSegment.from_mp3(wav_path)
    sound.export(output_filename,format="mp3",bitrate=bitrate)

# 将WordPress导出的XML格式文件转换成多个MarkDown格式的文件
def convert_wordpress_xml_to_markdown(xml_file='./a.xml', convert_content=1, replace_more=[]):
    import xml.etree.ElementTree as ET
    import re
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for item in root.findall('.//item'):
        title = item.find('title').text
        content = item.find('.//content:encoded', namespaces={'content': 'http://purl.org/rss/1.0/modules/content/'}).text
        if convert_content == 1:
            try:
                content = re.sub(r'<!--.*?-->', '', content)
                content = content.replace('<p>', '')
                content = content.replace('</p>', '')
                content = content.replace('<ol>', '')
                content = content.replace('</ol>', '')
                content = content.replace('<ul>', '')
                content = content.replace('</ul>', '')
                content = content.replace('<strong>', '')
                content = content.replace('</strong>', '')
                content = content.replace('</li>', '')
                content = content.replace('<li>', '+ ')
                content = content.replace('</h3>', '')
                content = re.sub(r'<h2.*?>', '## ', content)
                content = re.sub(r'<h3.*?>', '### ', content)
                content = re.sub(r'<h4.*?>', '#### ', content)
                for replace_item in replace_more:
                    content = content.replace(replace_item, '')
                for _ in range(100):
                    content = content.replace('\n\n\n', '\n\n')
            except:
                print(f'提示：字符串替换出现问题！出现问题的内容为：{content}')
        else:
            pass
        markdown_content = f"# {title}\n{content}"
        markdown_file_path = f"{title}.md"
        cleaned_filename = re.sub(r'[/:*?"<>|\'\\]', ' ', markdown_file_path)
        with open(cleaned_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)

# 凯利公式
def kelly_formula(p, b, a=1):
    f=(p/a)-((1-p)/b)
    return f

# 获取所有股票
def all_stocks():
    import numpy as np
    import akshare as ak
    stocks = ak.stock_zh_a_spot_em()
    title = np.array(stocks.columns)
    stock_data = stocks.values
    return title, stock_data

# 获取所有股票的代码
def all_stock_symbols():
    import guan
    title, stock_data = guan.all_stocks()
    stock_symbols = stock_data[:, 1]
    return stock_symbols

# 股票代码的分类
def stock_symbols_classification():
    import guan
    import re
    stock_symbols = guan.all_stock_symbols()
    # 上交所主板
    stock_symbols_60 = []
    for stock_symbol in stock_symbols:
        find_600 = re.findall(r'^600', stock_symbol)
        find_601 = re.findall(r'^601', stock_symbol)
        find_603 = re.findall(r'^603', stock_symbol)
        find_605 = re.findall(r'^605', stock_symbol)
        if find_600 != [] or find_601 != [] or find_603 != [] or find_605 != []:
            stock_symbols_60.append(stock_symbol)
    # 深交所主板
    stock_symbols_00 = []
    for stock_symbol in stock_symbols:
        find_000 = re.findall(r'^000', stock_symbol)
        find_001 = re.findall(r'^001', stock_symbol)
        find_002 = re.findall(r'^002', stock_symbol)
        find_003 = re.findall(r'^003', stock_symbol)
        if find_000 != [] or find_001 != [] or find_002 != [] or find_003 != []:
            stock_symbols_00.append(stock_symbol)
    # 创业板
    stock_symbols_30 = []
    for stock_symbol in stock_symbols:
        find_300 = re.findall(r'^300', stock_symbol)
        find_301 = re.findall(r'^301', stock_symbol)
        if find_300 != [] or find_301 != []:
            stock_symbols_30.append(stock_symbol)
    # 科创板
    stock_symbols_68 = []
    for stock_symbol in stock_symbols:
        find_688 = re.findall(r'^688', stock_symbol)
        find_689 = re.findall(r'^689', stock_symbol)
        if find_688 != [] or find_689 != []:
            stock_symbols_68.append(stock_symbol)
    # 新三板
    stock_symbols_8_4 = []
    for stock_symbol in stock_symbols:
        find_82 = re.findall(r'^82', stock_symbol)
        find_83 = re.findall(r'^83', stock_symbol)
        find_87 = re.findall(r'^87', stock_symbol)
        find_88 = re.findall(r'^88', stock_symbol)
        find_430 = re.findall(r'^430', stock_symbol)
        find_420 = re.findall(r'^420', stock_symbol)
        find_400 = re.findall(r'^400', stock_symbol)
        if find_82 != [] or find_83 != [] or find_87 != [] or find_88 != [] or find_430 != [] or find_420 != [] or find_400 != []:
            stock_symbols_8_4.append(stock_symbol)
    # 检查遗漏的股票代码
    stock_symbols_others = []
    for stock_symbol in stock_symbols:
        if stock_symbol not in stock_symbols_60 and stock_symbol not in stock_symbols_00 and stock_symbol not in stock_symbols_30 and stock_symbol not in stock_symbols_68 and stock_symbol not in stock_symbols_8_4:
            stock_symbols_others.others.append(stock_symbol) 
    return stock_symbols_60, stock_symbols_00, stock_symbols_30, stock_symbols_68, stock_symbols_8_4, stock_symbols_others

# 股票代码各个分类的数量
def statistics_of_stock_symbols_classification():
    import guan
    stock_symbols_60, stock_symbols_00, stock_symbols_30, stock_symbols_68, stock_symbols_8_4, stock_symbols_others = guan.stock_symbols_classification()
    num_stocks_60 = len(stock_symbols_60)
    num_stocks_00 = len(stock_symbols_00)
    num_stocks_30 = len(stock_symbols_30)
    num_stocks_68 = len(stock_symbols_68)
    num_stocks_8_4 = len(stock_symbols_8_4)
    num_stocks_others= len(stock_symbols_others)
    return num_stocks_60, num_stocks_00, num_stocks_30, num_stocks_68, num_stocks_8_4, num_stocks_others

# 从股票代码获取股票名称
def find_stock_name_from_symbol(symbol='000002'):
    import guan
    title, stock_data = guan.all_stocks()
    for stock in stock_data:
        if symbol in stock:
           stock_name = stock[2]
    return stock_name

# 市值排序
def sorted_market_capitalization(num=10):
    import numpy as np
    import guan
    title, stock_data = guan.all_stocks()
    new_stock_data = []
    for stock in stock_data:
        if np.isnan(float(stock[9])):
            continue
        else:
            new_stock_data.append(stock)
    new_stock_data = np.array(new_stock_data)
    list_index = np.argsort(new_stock_data[:, 17])
    list_index = list_index[::-1]
    if num == None:
        num = len(list_index)
    sorted_array = []
    for i0 in range(num):
        stock_symbol = new_stock_data[list_index[i0], 1]
        stock_name = new_stock_data[list_index[i0], 2]
        market_capitalization = new_stock_data[list_index[i0], 17]/1e8
        sorted_array.append([i0+1, stock_symbol, stock_name, market_capitalization])
    return sorted_array

# 美股市值排序
def sorted_market_capitalization_us(num=10):
    import akshare as ak
    import numpy as np
    stocks = ak.stock_us_spot_em()
    stock_data = stocks.values
    new_stock_data = []
    for stock in stock_data:
        if np.isnan(float(stock[9])):
            continue
        else:
            new_stock_data.append(stock)
    new_stock_data = np.array(new_stock_data)
    list_index = np.argsort(new_stock_data[:, 9])
    list_index = list_index[::-1]
    if num == None:
        num = len(list_index)
    sorted_array = []
    for i0 in range(num):
        stock_symbol = new_stock_data[list_index[i0], 15]
        stock_name = new_stock_data[list_index[i0], 1]
        market_capitalization = new_stock_data[list_index[i0], 9]/1e8
        sorted_array.append([i0+1, stock_symbol, stock_name, market_capitalization])
    return sorted_array

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
    return title, stock_data

# 绘制股票图
def plot_stock_line(date_array, opening_array, closing_array, high_array, low_array, lw_open_close=6, lw_high_low=2, xlabel='date', ylabel='price', title='', fontsize=20, labelsize=20, adjust_bottom=0.2, adjust_left=0.2, fontfamily='Times New Roman'):
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize, fontfamily=fontfamily)
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
    for i0 in range(len(date_array)):
        if opening_array[i0] <= closing_array[i0]:
            ax.vlines(date_array[i0], opening_array[i0], closing_array[i0], linestyle='-', color='red', lw=lw_open_close)
            ax.vlines(date_array[i0], low_array[i0], high_array[i0], color='red', linestyle='-', lw=lw_high_low)
        else:
            ax.vlines(date_array[i0], opening_array[i0], closing_array[i0], linestyle='-', color='green', lw=lw_open_close)
            ax.vlines(date_array[i0], low_array[i0], high_array[i0], color='green', linestyle='-', lw=lw_high_low)
    plt.show()
    plt.close('all')

# Guan软件包的使用统计（仅仅统计装机数和import次数）
def statistics_of_guan_package(function_name=None):
    import guan
    try:
        import socket
        datetime_date = guan.get_date()
        datetime_time = guan.get_time()
        current_version = guan.get_current_version('guan')
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(0.5)
        client_socket.connect(('socket.guanjihuan.com', 12345))
        mac_address = guan.get_mac_address()
        if function_name == None:
            message = {
                'server': 'py.guanjihuan.com',
                'date': datetime_date,
                'time': datetime_time,
                'version': current_version,
                'MAC_address': mac_address,
            }
        else:
            message = {
                'server': 'py.guanjihuan.com',
                'date': datetime_date,
                'time': datetime_time,
                'version': current_version,
                'MAC_address': mac_address,
                'function_name': function_name
            }
        import json
        send_message = json.dumps(message)
        client_socket.send(send_message.encode())
        client_socket.close()
    except:
        pass

# Guan软件包升级检查和提示（如果无法连接或者版本为最新，那么均没有提示）
def notification_of_upgrade(timeout=5):
    try:
        import guan
        latest_version = guan.get_latest_version(package_name='guan', timeout=timeout)
        current_version = guan.get_current_version('guan')
        if latest_version != None and current_version != None:
            if latest_version != current_version:
                print('升级提示：您当前使用的版本是 guan-'+current_version+'，目前已经有最新版本 guan-'+latest_version+'。您可以通过以下命令对软件包进行升级：pip install --upgrade guan -i https://pypi.python.org/simple 或 pip install --upgrade guan')
    except:
        pass