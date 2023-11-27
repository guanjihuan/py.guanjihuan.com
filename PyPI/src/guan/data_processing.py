# Module: data_processing (including figure-plotting and file-reading/writing)

# 导入plt, fig, ax
def import_plt_and_start_fig_ax(adjust_bottom=0.2, adjust_left=0.2, labelsize=20):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=adjust_bottom, left=adjust_left)
    ax.grid()
    ax.tick_params(labelsize=labelsize) 
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    import guan
    guan.statistics_of_guan_package()
    return plt, fig, ax

# 基于plt, fig, ax画图
def plot_without_starting_fig(plt, fig, ax, x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, style='', y_min=None, y_max=None, linewidth=None, markersize=None, color=None): 
    if color==None:
        ax.plot(x_array, y_array, style, linewidth=linewidth, markersize=markersize)
    else:
        ax.plot(x_array, y_array, style, linewidth=linewidth, markersize=markersize, color=color)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y_min=min(y_array)
        if y_max==None:
            y_max=max(y_array)
        ax.set_ylim(y_min, y_max)
    import guan
    guan.statistics_of_guan_package()

# 画图
def plot(x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style='', y_min=None, y_max=None, linewidth=None, markersize=None, adjust_bottom=0.2, adjust_left=0.2): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize)
    ax.plot(x_array, y_array, style, linewidth=linewidth, markersize=markersize)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y_min=min(y_array)
        if y_max==None:
            y_max=max(y_array)
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')
    guan.statistics_of_guan_package()

# 一组横坐标数据，两组纵坐标数据画图
def plot_two_array(x_array, y1_array, y2_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, markersize_1=None, markersize_2=None, adjust_bottom=0.2, adjust_left=0.2): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize) 
    ax.plot(x_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y1_min=min(y1_array)
            y2_min=min(y2_array)
            y_min=min([y1_min, y2_min])
        if y_max==None:
            y1_max=max(y1_array)
            y2_max=max(y2_array)
            y_max=max([y1_max, y2_max])
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')
    guan.statistics_of_guan_package()

# 两组横坐标数据，两组纵坐标数据画图
def plot_two_array_with_two_horizontal_array(x1_array, x2_array, y1_array, y2_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, markersize_1=None, markersize_2=None, adjust_bottom=0.2, adjust_left=0.2): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize) 
    ax.plot(x1_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x2_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y1_min=min(y1_array)
            y2_min=min(y2_array)
            y_min=min([y1_min, y2_min])
        if y_max==None:
            y1_max=max(y1_array)
            y2_max=max(y2_array)
            y_max=max([y1_max, y2_max])
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')
    guan.statistics_of_guan_package()

# 一组横坐标数据，三组纵坐标数据画图
def plot_three_array(x_array, y1_array, y2_array, y3_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', style_3='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, linewidth_3=None,markersize_1=None, markersize_2=None, markersize_3=None, adjust_bottom=0.2, adjust_left=0.2): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize) 
    ax.plot(x_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.plot(x_array, y3_array, style_3, linewidth=linewidth_3, markersize=markersize_3)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y1_min=min(y1_array)
            y2_min=min(y2_array)
            y3_min=min(y3_array)
            y_min=min([y1_min, y2_min, y3_min])
        if y_max==None:
            y1_max=max(y1_array)
            y2_max=max(y2_array)
            y3_max=max(y3_array)
            y_max=max([y1_max, y2_max, y3_max])
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')
    guan.statistics_of_guan_package()

# 三组横坐标数据，三组纵坐标数据画图
def plot_three_array_with_three_horizontal_array(x1_array, x2_array, x3_array, y1_array, y2_array, y3_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', style_3='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, linewidth_3=None,markersize_1=None, markersize_2=None, markersize_3=None, adjust_bottom=0.2, adjust_left=0.2): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize) 
    ax.plot(x1_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x2_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.plot(x3_array, y3_array, style_3, linewidth=linewidth_3, markersize=markersize_3)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    if y_min!=None or y_max!=None:
        if y_min==None:
            y1_min=min(y1_array)
            y2_min=min(y2_array)
            y3_min=min(y3_array)
            y_min=min([y1_min, y2_min, y3_min])
        if y_max==None:
            y1_max=max(y1_array)
            y2_max=max(y2_array)
            y3_max=max(y3_array)
            y_max=max([y1_max, y2_max, y3_max])
        ax.set_ylim(y_min, y_max)
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')
    guan.statistics_of_guan_package()

# 画三维图
def plot_3d_surface(x_array, y_array, matrix, xlabel='x', ylabel='y', zlabel='z', title='', fontsize=20, labelsize=15, show=1, save=0, filename='a', file_format='.jpg', dpi=300, z_min=None, z_max=None, rcount=100, ccount=100): 
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    matrix = np.array(matrix)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.subplots_adjust(bottom=0.1, right=0.65) 
    x_array, y_array = np.meshgrid(x_array, y_array)
    if len(matrix.shape) == 2:
        surf = ax.plot_surface(x_array, y_array, matrix, rcount=rcount, ccount=ccount, cmap=cm.coolwarm, linewidth=0, antialiased=False) 
    elif len(matrix.shape) == 3:
        for i0 in range(matrix.shape[2]):
            surf = ax.plot_surface(x_array, y_array, matrix[:,:,i0], rcount=rcount, ccount=ccount, cmap=cm.coolwarm, linewidth=0, antialiased=False) 
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_zlabel(zlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.zaxis.set_major_locator(LinearLocator(5)) 
    ax.zaxis.set_major_formatter('{x:.2f}')  
    if z_min!=None or z_max!=None:
        if z_min==None:
            z_min=matrix.min()
        if z_max==None:
            z_max=matrix.max()
        ax.set_zlim(z_min, z_max)
    ax.tick_params(labelsize=labelsize) 
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 
    cax = plt.axes([0.8, 0.1, 0.05, 0.8]) 
    cbar = fig.colorbar(surf, cax=cax)  
    cbar.ax.tick_params(labelsize=labelsize)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')
    import guan
    guan.statistics_of_guan_package()

# 画Contour图
def plot_contour(x_array, y_array, matrix, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=15, cmap='jet', levels=None, show=1, save=0, filename='a', file_format='.jpg', dpi=300):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2, right=0.75, left=0.2) 
    x_array, y_array = np.meshgrid(x_array, y_array)
    contour = ax.contourf(x_array,y_array,matrix,cmap=cmap, levels=levels) 
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.tick_params(labelsize=labelsize) 
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    cax = plt.axes([0.8, 0.2, 0.05, 0.68])
    cbar = fig.colorbar(contour, cax=cax) 
    cbar.ax.tick_params(labelsize=labelsize) 
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')
    import guan
    guan.statistics_of_guan_package()

# 画棋盘图/伪彩色图
def plot_pcolor(x_array, y_array, matrix, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=15, cmap='jet', levels=None, show=1, save=0, filename='a', file_format='.jpg', dpi=300):  
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2, right=0.75, left=0.2) 
    x_array, y_array = np.meshgrid(x_array, y_array)
    contour = ax.pcolor(x_array,y_array,matrix, cmap=cmap)
    ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman') 
    ax.tick_params(labelsize=labelsize) 
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    cax = plt.axes([0.8, 0.2, 0.05, 0.68])
    cbar = fig.colorbar(contour, cax=cax) 
    cbar.ax.tick_params(labelsize=labelsize) 
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')
    import guan
    guan.statistics_of_guan_package()

# 通过坐标画点和线
def draw_dots_and_lines(coordinate_array, draw_dots=1, draw_lines=1, max_distance=1.1, line_style='-k', linewidth=1, dot_style='ro', markersize=3, show=1, save=0, filename='a', file_format='.eps', dpi=300):
    import numpy as np
    import matplotlib.pyplot as plt
    coordinate_array = np.array(coordinate_array)
    print(coordinate_array.shape)
    x_range = max(coordinate_array[:, 0])-min(coordinate_array[:, 0])
    y_range = max(coordinate_array[:, 1])-min(coordinate_array[:, 1])
    fig, ax = plt.subplots(figsize=(6*x_range/y_range,6))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.axis('off')
    if draw_lines==1:
        for i1 in range(coordinate_array.shape[0]):
            for i2 in range(coordinate_array.shape[0]):
                if np.sqrt((coordinate_array[i1, 0] - coordinate_array[i2, 0])**2+(coordinate_array[i1, 1] - coordinate_array[i2, 1])**2) < max_distance:
                    ax.plot([coordinate_array[i1, 0], coordinate_array[i2, 0]], [coordinate_array[i1, 1], coordinate_array[i2, 1]], line_style, linewidth=linewidth)
    if draw_dots==1:
        for i in range(coordinate_array.shape[0]):
            ax.plot(coordinate_array[i, 0], coordinate_array[i, 1], dot_style, markersize=markersize)
    if show==1:
        plt.show()
    if save==1:
        if file_format=='.eps':
            plt.savefig(filename+file_format)
        else:
            plt.savefig(filename+file_format, dpi=dpi)
    import guan
    guan.statistics_of_guan_package()

# 合并两个图片
def combine_two_images(image_path_array, figsize=(16,8), show=0, save=1, filename='a', file_format='.jpg', dpi=300):
    import numpy as np
    num = np.array(image_path_array).shape[0]
    if num != 2:
        print('Error: The number of images should be two!')
    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0) 
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        image_1 = mpimg.imread(image_path_array[0])
        image_2 = mpimg.imread(image_path_array[1])
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        ax1.axis('off')
        ax2.axis('off')
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename+file_format, dpi=dpi)
        plt.close('all')
    import guan
    guan.statistics_of_guan_package()

# 合并三个图片
def combine_three_images(image_path_array, figsize=(16,5), show=0, save=1, filename='a', file_format='.jpg', dpi=300):
    import numpy as np
    num = np.array(image_path_array).shape[0]
    if num != 3:
        print('Error: The number of images should be three!')
    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0) 
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        image_1 = mpimg.imread(image_path_array[0])
        image_2 = mpimg.imread(image_path_array[1])
        image_3 = mpimg.imread(image_path_array[2])
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        ax3.imshow(image_3)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename+file_format, dpi=dpi)
        plt.close('all')
    import guan
    guan.statistics_of_guan_package()

# 合并四个图片
def combine_four_images(image_path_array, figsize=(16,16), show=0, save=1, filename='a', file_format='.jpg', dpi=300):
    import numpy as np
    num = np.array(image_path_array).shape[0]
    if num != 4:
        print('Error: The number of images should be four!')
    else:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0) 
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        image_1 = mpimg.imread(image_path_array[0])
        image_2 = mpimg.imread(image_path_array[1])
        image_3 = mpimg.imread(image_path_array[2])
        image_4 = mpimg.imread(image_path_array[3])
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        ax3.imshow(image_3)
        ax4.imshow(image_4)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        if show == 1:
            plt.show()
        if save == 1:
            plt.savefig(filename+file_format, dpi=dpi)
        plt.close('all')
    import guan
    guan.statistics_of_guan_package()

# 对某个目录中的txt文件批量读取和画图
def batch_reading_and_plotting(directory, xlabel='x', ylabel='y'):
    import re
    import os
    import guan
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.search('^txt.', file[::-1]):
                filename = file[:-4]
                x_array, y_array = guan.read_one_dimensional_data(filename=filename)
                guan.plot(x_array, y_array, xlabel=xlabel, ylabel=ylabel, title=filename, show=0, save=1, filename=filename)
    guan.statistics_of_guan_package()

# 将图片制作GIF动画
def make_gif(image_path_array, filename='a', duration=0.1):
    import imageio
    images = []
    for image_path in image_path_array:
        im = imageio.imread(image_path)
        images.append(im)
    imageio.mimsave(filename+'.gif', images, 'GIF', duration=duration)
    import guan
    guan.statistics_of_guan_package()

# 选取Matplotlib颜色
def color_matplotlib():
    color_array = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    import guan
    guan.statistics_of_guan_package()
    return color_array

# 将变量存到文件
def dump_data(data, filename, file_format='.txt'):
    import pickle
    with open(filename+file_format, 'wb') as f:
        pickle.dump(data, f)
    import guan
    guan.statistics_of_guan_package()

# 从文件中恢复数据到变量
def load_data(filename, file_format='.txt'):
    import pickle
    with open(filename+file_format, 'rb') as f:
        data = pickle.load(f)
    import guan
    guan.statistics_of_guan_package()
    return data

# 读取文件中的一维数据（一行一组x和y）
def read_one_dimensional_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    for row in row_list:
        column = np.array(row.split()) 
        if column.shape[0] != 0:  
            x_array = np.append(x_array, [float(column[0])], axis=0)  
            y_row = np.zeros(dim_column-1)
            for dim0 in range(dim_column-1):
                y_row[dim0] = float(column[dim0+1])
            if np.array(y_array).shape[0] == 0:
                y_array = [y_row]
            else:
                y_array = np.append(y_array, [y_row], axis=0)
    import guan
    guan.statistics_of_guan_package()
    return x_array, y_array

# 读取文件中的一维数据（一行一组x和y）（支持复数形式）
def read_one_dimensional_complex_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    for row in row_list:
        column = np.array(row.split()) 
        if column.shape[0] != 0:  
            x_array = np.append(x_array, [complex(column[0])], axis=0)  
            y_row = np.zeros(dim_column-1, dtype=complex)
            for dim0 in range(dim_column-1):
                y_row[dim0] = complex(column[dim0+1])
            if np.array(y_array).shape[0] == 0:
                y_array = [y_row]
            else:
                y_array = np.append(y_array, [y_row], axis=0)
    import guan
    guan.statistics_of_guan_package()
    return x_array, y_array

# 读取文件中的二维数据（第一行和第一列分别为横纵坐标）
def read_two_dimensional_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    matrix = np.array([])
    for i0 in range(row_list.shape[0]):
        column = np.array(row_list[i0].split()) 
        if i0 == 0:
            x_str = column[1::] 
            x_array = np.zeros(x_str.shape[0])
            for i00 in range(x_str.shape[0]):
                x_array[i00] = float(x_str[i00]) 
        elif column.shape[0] != 0: 
            y_array = np.append(y_array, [float(column[0])], axis=0)  
            matrix_row = np.zeros(dim_column-1)
            for dim0 in range(dim_column-1):
                matrix_row[dim0] = float(column[dim0+1])
            if np.array(matrix).shape[0] == 0:
                matrix = [matrix_row]
            else:
                matrix = np.append(matrix, [matrix_row], axis=0)
    import guan
    guan.statistics_of_guan_package()
    return x_array, y_array, matrix

# 读取文件中的二维数据（第一行和第一列分别为横纵坐标）（支持复数形式）
def read_two_dimensional_complex_data(filename='a', file_format='.txt'): 
    import numpy as np
    f = open(filename+file_format, 'r')
    text = f.read()
    f.close()
    row_list = np.array(text.split('\n')) 
    dim_column = np.array(row_list[0].split()).shape[0] 
    x_array = np.array([])
    y_array = np.array([])
    matrix = np.array([])
    for i0 in range(row_list.shape[0]):
        column = np.array(row_list[i0].split()) 
        if i0 == 0:
            x_str = column[1::] 
            x_array = np.zeros(x_str.shape[0], dtype=complex)
            for i00 in range(x_str.shape[0]):
                x_array[i00] = complex(x_str[i00]) 
        elif column.shape[0] != 0: 
            y_array = np.append(y_array, [complex(column[0])], axis=0)  
            matrix_row = np.zeros(dim_column-1, dtype=complex)
            for dim0 in range(dim_column-1):
                matrix_row[dim0] = complex(column[dim0+1])
            if np.array(matrix).shape[0] == 0:
                matrix = [matrix_row]
            else:
                matrix = np.append(matrix, [matrix_row], axis=0)
    import guan
    guan.statistics_of_guan_package()
    return x_array, y_array, matrix

# 读取文件中的二维数据（不包括x和y）
def read_two_dimensional_data_without_xy_array(filename='a', file_format='.txt'):
    import numpy as np
    matrix = np.loadtxt(filename+file_format)
    import guan
    guan.statistics_of_guan_package()
    return matrix

# 打开文件用于新增内容
def open_file(filename='a', file_format='.txt'):
    f = open(filename+file_format, 'a', encoding='UTF-8')
    import guan
    guan.statistics_of_guan_package()
    return f

# 在文件中写入一维数据（一行一组x和y）
def write_one_dimensional_data(x_array, y_array, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_one_dimensional_data_without_opening_file(x_array, y_array, f)
    import guan
    guan.statistics_of_guan_package()

# 在文件中写入一维数据（一行一组x和y）（需要输入已打开的文件）
def write_one_dimensional_data_without_opening_file(x_array, y_array, f):
    import numpy as np
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    i0 = 0
    for x0 in x_array:
        f.write(str(x0)+'   ')
        if len(y_array.shape) == 1:
            f.write(str(y_array[i0])+'\n')
        elif len(y_array.shape) == 2:
            for j0 in range(y_array.shape[1]):
                f.write(str(y_array[i0, j0])+'   ')
            f.write('\n')
        i0 += 1
    import guan
    guan.statistics_of_guan_package()

# 在文件中写入二维数据（第一行和第一列分别为横纵坐标）
def write_two_dimensional_data(x_array, y_array, matrix, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_two_dimensional_data_without_opening_file(x_array, y_array, matrix, f)
    guan.statistics_of_guan_package()

# 在文件中写入二维数据（第一行和第一列分别为横纵坐标）（需要输入已打开的文件）
def write_two_dimensional_data_without_opening_file(x_array, y_array, matrix, f):
    import numpy as np
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    matrix = np.array(matrix)
    f.write('0   ')
    for x0 in x_array:
        f.write(str(x0)+'   ')
    f.write('\n')
    i0 = 0
    for y0 in y_array:
        f.write(str(y0))
        j0 = 0
        for x0 in x_array:
            f.write('   '+str(matrix[i0, j0])+'   ')
            j0 += 1
        f.write('\n')
        i0 += 1
    import guan
    guan.statistics_of_guan_package()

# 在文件中写入二维数据（不包括x和y）
def write_two_dimensional_data_without_xy_array(matrix, filename='a', file_format='.txt'):
    import guan
    with open(filename+file_format, 'w', encoding='UTF-8') as f:
        guan.write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f)
    guan.statistics_of_guan_package()

# 在文件中写入二维数据（不包括x和y）（需要输入已打开的文件）
def write_two_dimensional_data_without_xy_array_and_without_opening_file(matrix, f):
    for row in matrix:
        for element in row:
            f.write(str(element)+'   ')
        f.write('\n')
    import guan
    guan.statistics_of_guan_package()

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
    import guan
    guan.statistics_of_guan_package()

# 获取目录中的所有文件名
def get_all_filenames_in_directory(directory='./', file_format=None):
    import os
    file_list = []
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if file_format == None:
                file_list.append(files[i0])
            else:
                if file_format in files[i0]:
                    file_list.append(files[i0])
    return file_list

# 读取文件夹中某种文本文件类型的文件路径和内容
def read_text_files_in_directory(directory='./', file_format='.md'):
    import os
    file_list = []
    for root, dirs, files in os.walk(directory):
        for i0 in range(len(files)):
            if file_format in files[i0]:
                file_list.append(root+'/'+files[i0])
    content_array = []
    for file in file_list:
        with open(file, 'r') as f:
            content_array.append(f.read())
    import guan
    guan.statistics_of_guan_package()
    return file_list, content_array

# 在多个文本文件中查找关键词
def find_words_in_multiple_files(words, directory='./', file_format='.md'):
    import guan
    file_list, content_array = guan.read_text_files_in_directory(directory=directory, file_format=file_format)
    num_files = len(file_list)
    file_list_with_words = []
    for i0 in range(num_files):
        if words in content_array[i0]:
            file_list_with_words.append(file_list[i0])
    guan.statistics_of_guan_package()
    return file_list_with_words

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

# 使用jieba软件包进行分词
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