# Module: others

# CPU性能测试（十亿次循环的浮点加法运算的时间，约30秒左右）
def cpu_test_with_addition(print_show=1):
    import time
    result = 0.0
    start_time = time.time()
    for _ in range(int(1e9)):
        result += 1e-9
    end_time = time.time()
    run_time = end_time - start_time
    if print_show:
        print(run_time)
    return run_time

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

# 使用pdfplumber将PDF文件转成文本
def pdf_to_text_with_pdfplumber(pdf_path):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        all_text = []
        for page in pdf.pages:
            text = page.extract_text()
            all_text.append(text)
        content = "\n\n".join(all_text)
        return content

# 使用pdfminer3k将PDF文件转成文本（仅仅支持旧版本的 pdfminer3k）
def pdf_to_text_with_pdfminer3k(pdf_path):
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

# 使用PyPDF2将PDF文件转成文本
def pdf_to_text_with_PyPDF2_for_all_pages(pdf_path):
    import guan
    num_pages = guan.get_pdf_page_number(pdf_path)
    content = ''
    for i0 in range(num_pages):
        page_text = guan.pdf_to_txt_for_a_specific_page(pdf_path, page_num=i0+1)
        content += page_text + '\n\n'
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

# 获取CPU使用率（基于性能计数器，适用于Windows系统）
def get_cpu_usage_for_windows(interval=1.0):
    import time
    import ctypes
    from ctypes import wintypes
    class FILETIME(ctypes.Structure):
        _fields_ = [
            ('dwLowDateTime', wintypes.DWORD),
            ('dwHighDateTime', wintypes.DWORD)
        ]
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    # 第一次采样
    idle1 = FILETIME()
    kernel1 = FILETIME()
    user1 = FILETIME()
    kernel32.GetSystemTimes(ctypes.byref(idle1), ctypes.byref(kernel1), ctypes.byref(user1))
    time.sleep(interval)
    # 第二次采样
    idle2 = FILETIME()
    kernel2 = FILETIME()
    user2 = FILETIME()
    kernel32.GetSystemTimes(ctypes.byref(idle2), ctypes.byref(kernel2), ctypes.byref(user2))
    # 计算时间差
    def filetime_to_int(ft):
        return (ft.dwHighDateTime << 32) + ft.dwLowDateTime
    idle = filetime_to_int(idle2) - filetime_to_int(idle1)
    kernel = filetime_to_int(kernel2) - filetime_to_int(kernel1)
    user = filetime_to_int(user2) - filetime_to_int(user1)
    total = kernel + user
    if total == 0:
        return 0.0
    return 100.0 * (total - idle) / total

# 获取CPU使用率（基于/proc/stat，适用于Linux系统）
def get_cpu_usage_for_linux(interval=1.0):
    import time
    def read_cpu_stats():
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('cpu '):
                    parts = line.split()
                    return list(map(int, parts[1:]))
        return None
    stats1 = read_cpu_stats()
    if not stats1:
        return 0.0
    time.sleep(interval)
    stats2 = read_cpu_stats()
    if not stats2:
        return 0.0
    idle1 = stats1[3] + stats1[4]
    total1 = sum(stats1)
    idle2 = stats2[3] + stats2[4]
    total2 = sum(stats2)
    total_delta = total2 - total1
    idle_delta = idle2 - idle1
    if total_delta == 0:
        return 0.0
    return 100.0 * (total_delta - idle_delta) / total_delta

# 使用psutil获取CPU的平均使用率
def get_cpu_usage(interval=1):
    import psutil
    cpu_usage = psutil.cpu_percent(interval=interval)
    return cpu_usage

# 使用psutil获取每个CPU核心的使用率，返回列表
def get_cpu_usage_array_per_core(interval=1):
    import psutil
    cpu_usage_array_per_core = psutil.cpu_percent(interval=interval, percpu=True)
    return cpu_usage_array_per_core

# 使用psutil获取使用率最高的CPU核心的使用率
def get_cpu_max_usage_for_all_cores(interval=1):
    import guan
    cpu_usage_array_per_core = guan.get_cpu_usage_array_per_core(interval=interval)
    max_cpu_usage = max(cpu_usage_array_per_core)
    return max_cpu_usage

# 使用psutil获取非零使用率的CPU核心的平均使用率
def get_cpu_averaged_usage_for_non_zero_cores(interval=1):
    import guan
    cpu_usage_array_per_core = guan.get_cpu_usage_array_per_core(interval=interval)
    cpu_usage_array_per_core_new = guan.remove_item_in_one_array(cpu_usage_array_per_core, 0.0)
    averaged_cpu_usage = sum(cpu_usage_array_per_core_new)/len(cpu_usage_array_per_core_new)
    return averaged_cpu_usage

# 使用psutil在一定数量周期内得到CPU的使用率信息。默认为1秒钟收集一次，(interval+sleep_interval)*times 为收集的时间范围，范围默认为60秒，即1分钟后返回列表，总共得到60组数据。其中，数字第一列和第二列分别是平均值和最大值。
def get_cpu_information_for_times(interval=1, sleep_interval=0, times=60):
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

# 使用psutil获取CPU的使用率，将得到的CPU的使用率信息写入文件。默认为1分钟收集一次，(interval+sleep_interval)*times 为收集的时间范围，范围默认为60分钟，即1小时写入文件一次，总共得到60组数据。其中，数字第一列和第二列分别是平均值和最大值。
def write_cpu_information_to_file(filename='./cpu_usage', interval=1, sleep_interval=59, times=60):
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

# 使用psutil获取CPU的使用率，画CPU的使用率图。默认为画最近的60个数据，以及不画CPU核心的最大使用率。
def plot_cpu_information(filename='./cpu_usage', recent_num=60, max_cpu=0):
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

# 使用psutil获取CPU的使用率，画详细的CPU的使用率图，分CPU核心画图。
def plot_detailed_cpu_information(filename='./cpu_usage', recent_num=60):
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

# 寻找所有的git仓库 
def find_git_repositories(base_path='./', ignored_directory_with_words=[]):
    import os
    git_repository_array = []
    for root, dirs, files in os.walk(base_path):
        if '.git' in dirs:
            ignore_signal = 0
            for word in  ignored_directory_with_words:
                if word in root:
                    ignore_signal = 1
                    break
            if ignore_signal == 0:
                git_repository_array.append(root)
    return git_repository_array

# 在git仓库列表中找到有修改待commit的
def get_git_repositories_to_commit(git_repository_array):
    import os
    import subprocess
    git_repository_array_to_commit = []
    for repository in git_repository_array:
        os.chdir(repository)
        status = subprocess.check_output(['git', 'status']).decode('utf-8') 
        if 'nothing to commit, working tree clean' in status:
            pass
        else:
            git_repository_array_to_commit.append(repository)
    return git_repository_array_to_commit

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

# 通过Sci-Hub网站下载文献（该方法可能失效）
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
        find_302 = re.findall(r'^302', stock_symbol)
        if find_300 != [] or find_301 != [] or find_302 != []:
            stock_symbols_30.append(stock_symbol)
    # 科创板
    stock_symbols_68 = []
    for stock_symbol in stock_symbols:
        find_688 = re.findall(r'^688', stock_symbol)
        find_689 = re.findall(r'^689', stock_symbol)
        if find_688 != [] or find_689 != []:
            stock_symbols_68.append(stock_symbol)
    # 北交所和新三板
    stock_symbols_8_4_9 = []
    for stock_symbol in stock_symbols:
        find_83 = re.findall(r'^83', stock_symbol)
        find_87 = re.findall(r'^87', stock_symbol)
        find_430 = re.findall(r'^430', stock_symbol)
        find_420 = re.findall(r'^420', stock_symbol)
        find_400 = re.findall(r'^400', stock_symbol)
        find_920 = re.findall(r'^920', stock_symbol)
        if find_83 != [] or find_87 != [] or find_430 != [] or find_420 != [] or find_400 != [] or find_920 != []:
            stock_symbols_8_4_9.append(stock_symbol)
    # 检查遗漏的股票代码
    stock_symbols_others = []
    for stock_symbol in stock_symbols:
        if stock_symbol not in stock_symbols_60 and stock_symbol not in stock_symbols_00 and stock_symbol not in stock_symbols_30 and stock_symbol not in stock_symbols_68 and stock_symbol not in stock_symbols_8_4_9:
            stock_symbols_others.append(stock_symbol) 
    return stock_symbols_60, stock_symbols_00, stock_symbols_30, stock_symbols_68, stock_symbols_8_4_9, stock_symbols_others

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

# Guan软件包升级检查和提示（对于无法连接或者版本为最新的情况，检查结果都没有提示）
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

# --- 自定义类和使用自定义类的函数 custom classes and functions using objects of custom classes ---

# 原子类
class Atom:
    def __init__(self, name='atom', index=0, x=0, y=0, z=0, energy=0):
        self.name = name
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.energy = energy

# 将原子对象列表转成多个独立列表
def convert_atom_object_list_to_multiple_lists(atom_object_list):
    name_list = []
    index_list = []
    x_list = []
    y_list = []
    z_list = []
    energy_list = []
    for atom_object in atom_object_list:
        name_list.append(atom_object.name)
        index_list.append(atom_object.index)
        x_list.append(atom_object.x)
        y_list.append(atom_object.y)
        z_list.append(atom_object.z)
        energy_list.append(atom_object.energy)
    return name_list, index_list, x_list, y_list, z_list, energy_list

# 将原子对象列表转成原子字典列表
def convert_atom_object_list_to_atom_dict_list(atom_object_list):
    atom_dict_list = []
    for atom_object in atom_object_list:
        atom_dict = {
            'name': atom_object.name,
            'index': atom_object.index, 
            'x': atom_object.x,
            'y': atom_object.y,
            'z': atom_object.z,
            'energy': atom_object.energy,
        }
        atom_dict_list.append(atom_dict)
    return atom_dict_list

# 从原子对象列表中获取 (x, y) 坐标数组
def get_coordinate_array_from_atom_object_list(atom_object_list):
    coordinate_array = []
    for atom in atom_object_list:
        x = atom.x
        y = atom.y
        coordinate_array.append([x, y])
    return coordinate_array

# 从原子对象列表中获取 x 和 y 的最大值和最小值
def get_max_min_x_y_from_atom_object_list(atom_object_list):
    import guan
    coordinate_array = guan.get_coordinate_array_from_atom_object_list(atom_object_list)
    x_array = []
    for coordinate in coordinate_array:
        x_array.append(coordinate[0])
    y_array = []
    for coordinate in coordinate_array:
        y_array.append(coordinate[1])
    max_x = max(x_array)
    min_x = min(x_array)
    max_y = max(y_array)
    min_y = min(y_array)
    return max_x, min_x, max_y, min_y

# 从原子对象列表中获取满足坐标条件的索引
def get_index_via_coordinate_from_atom_object_list(atom_object_list, x=0, y=0, z=0, eta=1e-3):
    for atom in atom_object_list:
        x_i = atom.x
        y_i = atom.y
        z_i = atom.z
        index = atom.index
        if abs(x-x_i)<eta and abs(y-y_i)<eta and abs(z-z_i)<eta:
            return index

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

# --- 废弃函数/版本兼容（不推荐使用，并可能在未来的版本中被移除）deprecated ---

def make_sh_file(sh_filename='a', command_line='python a.py', cpu_num=1, task_name='task', cd_dir=0):
    import guan
    print('Warning: The current function name has been deprecated, which will be deleted in the future version. Please change it into guan.make_sh_file_for_qsub().')
    guan.make_sh_file_for_qsub(sh_filename=sh_filename, command_line=command_line, cpu_num=cpu_num, task_name=task_name, cd_dir=cd_dir)

def plot_without_starting_fig(plt, fig, ax, x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, style='', y_min=None, y_max=None, linewidth=None, markersize=None, color=None, fontfamily='Times New Roman'):
    import guan
    print('Warning: The current function name has been deprecated, which will be deleted in the future version. Please change it into guan.plot_without_starting_fig_ax().')
    guan.plot_without_starting_fig_ax(plt, fig, ax, x_array, y_array, xlabel=xlabel, ylabel=ylabel, title=title, fontsize=fontsize, style=style, y_min=y_min, y_max=y_max, linewidth=linewidth, markersize=markersize, color=color, fontfamily=fontfamily)

def draw_dots_and_lines_without_starting_fig(plt, fig, ax, coordinate_array, draw_dots=1, draw_lines=1, max_distance=1, line_style='-k', linewidth=1, dot_style='ro', markersize=3):
    import guan
    print('Warning: The current function name has been deprecated, which will be deleted in the future version. Please change it into guan.draw_dots_and_lines_without_starting_fig_ax().')
    guan.draw_dots_and_lines_without_starting_fig_ax(plt, fig, ax, coordinate_array, draw_dots=draw_dots, draw_lines=draw_lines, max_distance=max_distance, line_style=line_style, linewidth=linewidth, dot_style=dot_style, markersize=markersize)

def get_days_of_the_current_month(str_or_datetime='str'):
    import guan
    print('Warning: The current function name has been deprecated, which will be deleted in the future version. Please change it into guan.get_date_array_of_the_current_month().')
    date_array = guan.get_date_array_of_the_current_month(str_or_datetime=str_or_datetime)
    return date_array

def get_days_of_the_last_month(str_or_datetime='str'):
    import guan
    print('Warning: The current function name has been deprecated, which will be deleted in the future version. Please change it into guan.get_date_array_of_the_last_month().')
    date_array = guan.get_date_array_of_the_last_month(str_or_datetime=str_or_datetime)
    return date_array

def get_days_of_the_month_before_last(str_or_datetime='str'):
    import guan
    print('Warning: The current function name has been deprecated, which will be deleted in the future version. Please change it into guan.get_date_array_of_the_month_before_last().')
    date_array = guan.get_date_array_of_the_month_before_last(str_or_datetime=str_or_datetime)
    return date_array

def pdf_to_text(pdf_path):
    import guan
    print('Warning: The current function name has been deprecated, which will be deleted in the future version. Please change it into guan.pdf_to_text_with_pdfminer3k().')
    content = guan.pdf_to_text_with_pdfminer3k(pdf_path)
    return content

def statistics_with_day_and_time(content='', filename='time_logging', file_format='.txt'):
    import guan
    print('Warning: The current function name has been deprecated, which will be deleted in the future version. Please change it into guan.logging_with_day_and_time().')
    guan.logging_with_day_and_time(content=content, filename=filename, file_format=file_format)