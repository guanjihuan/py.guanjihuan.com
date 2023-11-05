# Module: file_processing

# 自动先后运行程序（串行）
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
    import guan
    guan.statistics_of_guan_package()

# 如果不存在文件夹，则新建文件夹
def make_directory(directory='./test'):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    import guan
    guan.statistics_of_guan_package()

# 复制一份文件
def copy_file(file1='./a.txt', file2='./b.txt'):
    import shutil
    shutil.copy(file1, file2)
    import guan
    guan.statistics_of_guan_package()

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
    import guan
    guan.statistics_of_guan_package()

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
    import guan
    guan.statistics_of_guan_package()
    return content

# 获取PDF文件页数
def get_pdf_page_number(pdf_path):
    import PyPDF2
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    import guan
    guan.statistics_of_guan_package()
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
    import guan
    guan.statistics_of_guan_package()
    return page_text

# 获取PDF文献中的链接。例如: link_starting_form='https://doi.org'
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
    import guan
    guan.statistics_of_guan_package()
    return links

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
    import guan
    guan.statistics_of_guan_package()

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
    import guan
    guan.statistics_of_guan_package()

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
    import guan
    guan.statistics_of_guan_package()
    return repeated_file

# 统计各个子文件夹中的文件数量
def count_file_in_sub_directory(directory='./', smaller_than_num=None, sort=0):
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
            if smaller_than_num == None:
                print(sub_dir)
                print(count_file)
                print()
            else:
                if count_file<smaller_than_num:
                    print(sub_dir)
                    print(count_file)
                    print()
    if sort == 1:
        index_array = np.argsort(count_file_array)
        if smaller_than_num == None:
            for i0 in index_array:
                print(dirs_list[i0])
                print(count_file_array[i0])
                print()
        else:
            for i0 in index_array:
                if count_file_array[i0]<smaller_than_num:
                    print(dirs_list[i0])
                    print(count_file_array[i0])
                    print()
    import guan
    guan.statistics_of_guan_package()
    return dirs_list, count_file_array

# 产生必要的文件，例如readme.md
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
    import guan
    guan.statistics_of_guan_package()

# 删除特定文件名的文件
def delete_file_with_specific_name(directory, filename='readme', file_format='.md'):
    import os
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if files[i0] == filename+file_format:
                os.remove(root+'/'+files[i0])
    import guan
    guan.statistics_of_guan_package()

# 所有文件移到根目录（慎用）
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
    import guan
    guan.statistics_of_guan_package()

# 改变当前的目录位置
def change_directory_by_replacement(current_key_word='code', new_key_word='data'):
    import os
    code_path = os.getcwd()
    data_path = code_path.replace('\\', '/') 
    data_path = data_path.replace(current_key_word, new_key_word) 
    if os.path.exists(data_path) == False:
        os.makedirs(data_path)
    os.chdir(data_path)
    import guan
    guan.statistics_of_guan_package()

# 将文本转成音频
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
    guan.statistics_of_guan_package()

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
    guan.statistics_of_guan_package()

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
    guan.statistics_of_guan_package()

# 将wav音频文件压缩成MP3音频文件
def compress_wav_to_mp3(wav_path, output_filename='a.mp3', bitrate='16k'):
    # Note: Beside the installation of pydub, you may also need download FFmpeg on http://www.ffmpeg.org/download.html and add the bin path to the environment variable.
    from pydub import AudioSegment
    sound = AudioSegment.from_mp3(wav_path)
    sound.export(output_filename,format="mp3",bitrate=bitrate)
    import guan
    guan.statistics_of_guan_package()
