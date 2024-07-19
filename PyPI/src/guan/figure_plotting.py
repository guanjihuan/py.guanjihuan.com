# Module: figure_plotting

# 导入plt, fig, ax
def import_plt_and_start_fig_ax(adjust_bottom=0.2, adjust_left=0.2, labelsize=20, fontfamily='Times New Roman'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=adjust_bottom, left=adjust_left)
    ax.grid()
    ax.tick_params(labelsize=labelsize) 
    if fontfamily=='Times New Roman':
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
    return plt, fig, ax

# 基于plt, fig, ax画图
def plot_without_starting_fig_ax(plt, fig, ax, x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, style='', y_min=None, y_max=None, linewidth=None, markersize=None, color=None, fontfamily='Times New Roman'): 
    if color==None:
        ax.plot(x_array, y_array, style, linewidth=linewidth, markersize=markersize)
    else:
        ax.plot(x_array, y_array, style, linewidth=linewidth, markersize=markersize, color=color)
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if y_min!=None or y_max!=None:
        if y_min==None:
            y_min=min(y_array)
        if y_max==None:
            y_max=max(y_array)
        ax.set_ylim(y_min, y_max)

# 画图
def plot(x_array, y_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style='', y_min=None, y_max=None, linewidth=None, markersize=None, adjust_bottom=0.2, adjust_left=0.2, fontfamily='Times New Roman'): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize, fontfamily=fontfamily)
    ax.plot(x_array, y_array, style, linewidth=linewidth, markersize=markersize)
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
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

# 一组横坐标数据，两组纵坐标数据画图
def plot_two_array(x_array, y1_array, y2_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, markersize_1=None, markersize_2=None, adjust_bottom=0.2, adjust_left=0.2, fontfamily='Times New Roman'): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize, fontfamily=fontfamily) 
    ax.plot(x_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
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

# 两组横坐标数据，两组纵坐标数据画图
def plot_two_array_with_two_horizontal_array(x1_array, x2_array, y1_array, y2_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, markersize_1=None, markersize_2=None, adjust_bottom=0.2, adjust_left=0.2, fontfamily='Times New Roman'): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize, fontfamily=fontfamily) 
    ax.plot(x1_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x2_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
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

# 一组横坐标数据，三组纵坐标数据画图
def plot_three_array(x_array, y1_array, y2_array, y3_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', style_3='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, linewidth_3=None,markersize_1=None, markersize_2=None, markersize_3=None, adjust_bottom=0.2, adjust_left=0.2, fontfamily='Times New Roman'): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize, fontfamily=fontfamily) 
    ax.plot(x_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.plot(x_array, y3_array, style_3, linewidth=linewidth_3, markersize=markersize_3)
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
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

# 三组横坐标数据，三组纵坐标数据画图
def plot_three_array_with_three_horizontal_array(x1_array, x2_array, x3_array, y1_array, y2_array, y3_array, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=20, show=1, save=0, filename='a', file_format='.jpg', dpi=300, style_1='', style_2='', style_3='', y_min=None, y_max=None, linewidth_1=None, linewidth_2=None, linewidth_3=None,markersize_1=None, markersize_2=None, markersize_3=None, adjust_bottom=0.2, adjust_left=0.2, fontfamily='Times New Roman'): 
    import guan
    plt, fig, ax = guan.import_plt_and_start_fig_ax(adjust_bottom=adjust_bottom, adjust_left=adjust_left, labelsize=labelsize, fontfamily=fontfamily) 
    ax.plot(x1_array, y1_array, style_1, linewidth=linewidth_1, markersize=markersize_1)
    ax.plot(x2_array, y2_array, style_2, linewidth=linewidth_2, markersize=markersize_2)
    ax.plot(x3_array, y3_array, style_3, linewidth=linewidth_3, markersize=markersize_3)
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
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

# 画三维图
def plot_3d_surface(x_array, y_array, matrix, xlabel='x', ylabel='y', zlabel='z', title='', fontsize=20, labelsize=15, show=1, save=0, filename='a', file_format='.jpg', dpi=300, z_min=None, z_max=None, rcount=100, ccount=100, fontfamily='Times New Roman'): 
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
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_zlabel(zlabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_zlabel(zlabel, fontsize=fontsize)
    ax.zaxis.set_major_locator(LinearLocator(5)) 
    ax.zaxis.set_major_formatter('{x:.2f}')  
    if z_min!=None or z_max!=None:
        if z_min==None:
            z_min=matrix.min()
        if z_max==None:
            z_max=matrix.max()
        ax.set_zlim(z_min, z_max)
    ax.tick_params(labelsize=labelsize) 
    if fontfamily=='Times New Roman':
        labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
        [label.set_fontname('Times New Roman') for label in labels] 
    cax = plt.axes([0.8, 0.1, 0.05, 0.8]) 
    cbar = fig.colorbar(surf, cax=cax)  
    cbar.ax.tick_params(labelsize=labelsize)
    if fontfamily=='Times New Roman':
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

# 画Contour图
def plot_contour(x_array, y_array, matrix, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=15, cmap='jet', levels=None, show=1, save=0, filename='a', file_format='.jpg', dpi=300, fontfamily='Times New Roman'):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2, right=0.75, left=0.2) 
    x_array, y_array = np.meshgrid(x_array, y_array)
    contour = ax.contourf(x_array,y_array,matrix,cmap=cmap, levels=levels) 
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize) 
        ax.set_ylabel(ylabel, fontsize=fontsize) 
    ax.tick_params(labelsize=labelsize)
    if fontfamily=='Times New Roman':
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
    cax = plt.axes([0.8, 0.2, 0.05, 0.68])
    cbar = fig.colorbar(contour, cax=cax) 
    cbar.ax.tick_params(labelsize=labelsize)
    if fontfamily=='Times New Roman':
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

# 画棋盘图/伪彩色图
def plot_pcolor(x_array, y_array, matrix, xlabel='x', ylabel='y', title='', fontsize=20, labelsize=15, cmap='jet', levels=None, show=1, save=0, filename='a', file_format='.jpg', dpi=300, fontfamily='Times New Roman'):  
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2, right=0.75, left=0.2) 
    x_array, y_array = np.meshgrid(x_array, y_array)
    contour = ax.pcolor(x_array,y_array,matrix, cmap=cmap)
    if fontfamily=='Times New Roman':
        ax.set_title(title, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily='Times New Roman')
    else:
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=labelsize)
    if fontfamily=='Times New Roman':
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
    cax = plt.axes([0.8, 0.2, 0.05, 0.68])
    cbar = fig.colorbar(contour, cax=cax) 
    cbar.ax.tick_params(labelsize=labelsize)
    if fontfamily=='Times New Roman':
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_family('Times New Roman')
    if save == 1:
        plt.savefig(filename+file_format, dpi=dpi) 
    if show == 1:
        plt.show()
    plt.close('all')

# 基于plt, fig, ax，通过坐标画点和线
def draw_dots_and_lines_without_starting_fig_ax(plt, fig, ax, coordinate_array, draw_dots=1, draw_lines=1, max_distance=1, line_style='-k', linewidth=1, dot_style='ro', markersize=3):
    import numpy as np
    coordinate_array = np.array(coordinate_array)
    if draw_lines==1:
        for i1 in range(coordinate_array.shape[0]):
            for i2 in range(coordinate_array.shape[0]):
                if np.sqrt((coordinate_array[i1, 0] - coordinate_array[i2, 0])**2+(coordinate_array[i1, 1] - coordinate_array[i2, 1])**2) <= max_distance:
                    ax.plot([coordinate_array[i1, 0], coordinate_array[i2, 0]], [coordinate_array[i1, 1], coordinate_array[i2, 1]], line_style, linewidth=linewidth)
    if draw_dots==1:
        for i in range(coordinate_array.shape[0]):
            ax.plot(coordinate_array[i, 0], coordinate_array[i, 1], dot_style, markersize=markersize)


# 通过坐标画点和线
def draw_dots_and_lines(coordinate_array, draw_dots=1, draw_lines=1, max_distance=1, line_style='-k', linewidth=1, dot_style='ro', markersize=3, show=1, save=0, filename='a', file_format='.eps', dpi=300):
    import numpy as np
    import matplotlib.pyplot as plt
    coordinate_array = np.array(coordinate_array)
    x_range = max(coordinate_array[:, 0])-min(coordinate_array[:, 0])
    y_range = max(coordinate_array[:, 1])-min(coordinate_array[:, 1])
    fig, ax = plt.subplots(figsize=(6*x_range/y_range,6))
    ax.set_aspect('equal') # important code ensuring that the x and y axes have the same scale.
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.axis('off')
    if draw_lines==1:
        for i1 in range(coordinate_array.shape[0]):
            for i2 in range(coordinate_array.shape[0]):
                if np.sqrt((coordinate_array[i1, 0] - coordinate_array[i2, 0])**2+(coordinate_array[i1, 1] - coordinate_array[i2, 1])**2) <= max_distance:
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

# 将图片制作GIF动画
def make_gif(image_path_array, filename='a', duration=0.1):
    import imageio
    images = []
    for image_path in image_path_array:
        im = imageio.imread(image_path)
        images.append(im)
    imageio.mimsave(filename+'.gif', images, 'GIF', duration=duration)

# 选取Matplotlib颜色
def color_matplotlib():
    color_array = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    return color_array