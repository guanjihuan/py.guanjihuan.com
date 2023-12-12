# Module: others
import guan

# 获取CPU使用率
@guan.statistics_decorator
def get_cpu_usage(interval=1):
    import psutil
    cpu_usage = psutil.cpu_percent(interval=interval)
    return cpu_usage

# 获取内存信息
@guan.statistics_decorator
def get_memory_info():
    import psutil
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total/(1024**2)
    used_memory = memory_info.used/(1024**2)
    available_memory = memory_info.available/(1024**2)
    used_memory_percent = memory_info.percent
    return total_memory, used_memory, available_memory, used_memory_percent

# 将WordPress导出的XML格式文件转换成多个MarkDown格式的文件
@guan.statistics_decorator
def convert_wordpress_xml_to_markdown(xml_file='./a.xml', convert_content=1, replace_more=[]):
    import xml.etree.ElementTree as ET
    import re
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for item in root.findall('.//item'):
        title = item.find('title').text
        content = item.find('.//content:encoded', namespaces={'content': 'http://purl.org/rss/1.0/modules/content/'}).text
        if convert_content == 1:
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
        else:
            pass
        markdown_content = f"# {title}\n{content}"
        markdown_file_path = f"{title}.md"
        cleaned_filename = re.sub(r'[/:*?"<>|\'\\]', ' ', markdown_file_path)
        with open(cleaned_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)

# 获取运行的日期和时间并写入文件
@guan.statistics_decorator
def statistics_with_day_and_time(content='', filename='a', file_format='.txt'):
    import datetime
    datetime_today = str(datetime.date.today())
    datetime_time = datetime.datetime.now().strftime('%H:%M:%S')
    with open(filename+file_format, 'a', encoding="utf-8") as f2:
       if content == '':
           f2.write(datetime_today+' '+datetime_time+'\n')
       else:
           f2.write(datetime_today+' '+datetime_time+' '+content+'\n')

# 统计Python文件中import的数量并排序
@guan.statistics_decorator
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

# 根据一定的字符长度来分割文本
@guan.statistics_decorator
def split_text(text, wrap_width=3000):  
    import textwrap  
    split_text_list = textwrap.wrap(text, wrap_width)
    return split_text_list

# 获取本月的所有日期
@guan.statistics_decorator
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
    return day_array

# 获取上个月份
@guan.statistics_decorator
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
@guan.statistics_decorator
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
@guan.statistics_decorator
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
    return day_array

# 获取上上个月的所有日期
@guan.statistics_decorator
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
    return day_array

# 获取所有股票
@guan.statistics_decorator
def all_stocks():
    import numpy as np
    import akshare as ak
    stocks = ak.stock_zh_a_spot_em()
    title = np.array(stocks.columns)
    stock_data = stocks.values
    return title, stock_data

# 获取所有股票的代码
@guan.statistics_decorator
def all_stock_symbols():
    import guan
    title, stock_data = guan.all_stocks()
    stock_symbols = stock_data[:, 1]
    return stock_symbols

# 从股票代码获取股票名称
@guan.statistics_decorator
def find_stock_name_from_symbol(symbol='000002'):
    import guan
    title, stock_data = guan.all_stocks()
    for stock in stock_data:
        if symbol in stock:
           stock_name = stock[2]
    return stock_name

# 获取单个股票的历史数据
@guan.statistics_decorator
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

# 获取软件包中的所有模块名
@guan.statistics_decorator
def get_all_modules_in_one_package(package_name='guan'):
    import pkgutil
    package = __import__(package_name)
    module_names = [name for _, name, _ in pkgutil.iter_modules(package.__path__)]
    return module_names

# 获取软件包中一个模块的所有函数名
@guan.statistics_decorator
def get_all_functions_in_one_module(module_name, package_name='guan'):
    import inspect
    function_names = []
    module = __import__(f"{package_name}.{module_name}", fromlist=[""])
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            function_names.append(name)
    return function_names

# 获取软件包中的所有函数名
@guan.statistics_decorator
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

# 获取包含某个字符的进程PID值
@guan.statistics_decorator
def get_PID(name):
    import subprocess
    command = "ps -ef | grep "+name
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        ps_ef = result.stdout
    import re
    ps_ef = re.split(r'\s+', ps_ef)
    id_running = ps_ef[1]
    return id_running

# 获取函数的源码
@guan.statistics_decorator
def get_function_source(function_name):
    import inspect
    function_source = inspect.getsource(function_name)
    return function_source

# 查找文件名相同的文件
@guan.statistics_decorator
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
@guan.statistics_decorator
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

# 改变当前的目录位置
@guan.statistics_decorator
def change_directory_by_replacement(current_key_word='code', new_key_word='data'):
    import os
    code_path = os.getcwd()
    data_path = code_path.replace('\\', '/') 
    data_path = data_path.replace(current_key_word, new_key_word) 
    if os.path.exists(data_path) == False:
        os.makedirs(data_path)
    os.chdir(data_path)

# 在多个子文件夹中产生必要的文件，例如 readme.md
@guan.statistics_decorator
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

# 删除特定文件名的文件（慎用）
@guan.statistics_decorator
def delete_file_with_specific_name(directory, filename='readme', file_format='.md'):
    import os
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if files[i0] == filename+file_format:
                os.remove(root+'/'+files[i0])

# 将所有文件移到根目录（慎用）
@guan.statistics_decorator
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

# 将文件目录结构写入Markdown文件
@guan.statistics_decorator
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
@guan.statistics_decorator
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

# 生成二维码
@guan.statistics_decorator
def creat_qrcode(data="https://www.guanjihuan.com", filename='a', file_format='.png'):
    import qrcode
    img = qrcode.make(data)
    img.save(filename+file_format)

# 将PDF文件转成文本
@guan.statistics_decorator
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
@guan.statistics_decorator
def get_pdf_page_number(pdf_path):
    import PyPDF2
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    return num_pages

# 获取PDF文件指定页面的内容
@guan.statistics_decorator
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
@guan.statistics_decorator
def get_links_from_pdf(pdf_path, link_starting_form=''):
    import PyPDF2
    import re
    pdfReader = PyPDF2.PdfFileReader(pdf_path)
    pages = pdfReader.getNumPages()
    i0 = 0
    links = []
    for page in range(pages):
        pageSliced = pdfReader.getPage(page)
        pageObject = pageSliced.getObject()
        if '/Annots' in pageObject.keys():
            ann = pageObject['/Annots']
            old = ''
            for a in ann:
                u = a.getObject()
                if '/A' in u.keys():
                    if re.search(re.compile('^'+link_starting_form), u['/A']['/URI']):
                        if u['/A']['/URI'] != old:
                            links.append(u['/A']['/URI']) 
                            i0 += 1
                            old = u['/A']['/URI']
    return links

# 通过Sci-Hub网站下载文献
@guan.statistics_decorator
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
@guan.statistics_decorator
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
@guan.statistics_decorator
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
@guan.statistics_decorator
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
@guan.statistics_decorator
def compress_wav_to_mp3(wav_path, output_filename='a.mp3', bitrate='16k'):
    # Note: Beside the installation of pydub, you may also need download FFmpeg on http://www.ffmpeg.org/download.html and add the bin path to the environment variable.
    from pydub import AudioSegment
    sound = AudioSegment.from_mp3(wav_path)
    sound.export(output_filename,format="mp3",bitrate=bitrate)

# Guan软件包的使用统计（不涉及到用户的个人数据）
global_variable_of_first_guan_package_calling = []
def statistics_of_guan_package(function_name=None):
    import guan
    if function_name == None:
        function_name = guan.get_calling_function_name(layer=2)
    else:
        pass
    global global_variable_of_first_guan_package_calling
    if function_name not in global_variable_of_first_guan_package_calling:
        global_variable_of_first_guan_package_calling.append(function_name)
        try:
            import socket
            datetime_date = guan.get_date()
            datetime_time = guan.get_time()
            current_version = guan.get_current_version('guan')
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(0.5)
            client_socket.connect(('socket.guanjihuan.com', 12345))
            mac_address = guan.get_mac_address()
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

# 获取MAC地址
def get_mac_address():
    import uuid
    mac_address = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
    mac_address = '-'.join([mac_address[i:i+2] for i in range(0, 11, 2)])
    return mac_address

# 获取调用本函数的函数名
def get_calling_function_name(layer=1):
    import inspect
    caller = inspect.stack()[layer]
    calling_function_name = caller.function
    return calling_function_name

# 获取Python软件包的最新版本
def get_latest_version(package_name='guan', timeout=2):
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
def notification_of_upgrade(timeout=2):
    try:
        latest_version = get_latest_version(package_name='guan', timeout=timeout)
        current_version = get_current_version('guan')
        if latest_version != None and current_version != None:
            if latest_version != current_version:
                print('升级提示：您当前使用的版本是 guan-'+current_version+'，目前已经有最新版本 guan-'+latest_version+'。您可以通过以下命令对软件包进行升级：pip install --upgrade guan -i https://pypi.python.org/simple 或 pip install --upgrade guan')
    except:
        pass
import random
rand_number = random.randint(1, 10)
if rand_number == 5:
    notification_of_upgrade(timeout=2)