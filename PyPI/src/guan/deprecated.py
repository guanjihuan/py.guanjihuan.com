# Module: deprecated

# --- 废弃函数/版本兼容（不推荐使用，并可能在未来的版本中被移除） ---

# AI 对话（当前默认模型为 'hunyuan-lite'，无记忆）
def chat(prompt='你好', model=1, stream=1, stream_label=0):
    import requests
    url = "http://api.guanjihuan.com/chat"
    data = {
        "prompt": prompt, 
        "model": model,
    }
    if stream == 1:
        if stream_label == 1:
            print('\n--- Start Chat Stream Message ---\n')
    requests_response = requests.post(url, json=data, stream=True)
    response = ''
    if requests_response.status_code == 200:
        for line in requests_response.iter_lines():
            if line:
                if stream == 1:
                    print(line.decode('utf-8'), end='', flush=True)
                response += line.decode('utf-8')
        print()
    else:
        pass
    if stream == 1:
        if stream_label == 1:
            print('\n--- End Chat Stream Message ---\n')
    return response

# 加上函数代码的 AI 对话（当前默认模型为 'hunyuan-lite'，无记忆）
def chat_with_function_code(function_name, prompt='', model=1, stream=1):
    import guan
    function_source = guan.get_source(function_name)
    if prompt == '':
        response = guan.chat(prompt=function_source, model=model, stream=stream)
    else:
        response = guan.chat(prompt=function_source+'\n\n'+prompt, model=model, stream=stream)
    return response

# 机器人自动对话（当前默认模型为 'hunyuan-lite'，无记忆）
def auto_chat(prompt='你好', round=2, model=1, stream=1):
    import guan
    response0 = prompt
    for i0 in range(round):
        print(f'\n【对话第 {i0+1} 轮】\n')
        print('机器人 1: ')
        response1 = guan.chat(prompt=response0, model=model, stream=stream)
        print('机器人 2: ')
        response0 = guan.chat(prompt=response1, model=model, stream=stream)

# 机器人自动对话（引导对话）（当前默认模型为 'hunyuan-lite'，无记忆）
def auto_chat_with_guide(prompt='你好', guide_message='（回答字数少于30个字，最后反问我一个问题）', round=5, model=1, stream=1):
    import guan
    response0 = prompt
    for i0 in range(round):
        print(f'\n【对话第 {i0+1} 轮】\n')
        print('机器人 1: ')
        response1 = guan.chat(prompt=response0+guide_message, model=model, stream=stream)
        print('机器人 2: ')
        response0 = guan.chat(prompt=response1+guide_message, model=model, stream=stream)

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