# Module: deprecated

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