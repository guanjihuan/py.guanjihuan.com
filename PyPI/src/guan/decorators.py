# Module: decorators

# 函数的装饰器，用于获取计算时间（秒）
def timer_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Running time of {func.__name__}: {end - start} seconds")
        return result
    return wrapper

# 函数的装饰器，用于获取计算时间（分）
def timer_decorator_minutes(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Running time of {func.__name__}: {(end - start)/60} minutes")
        return result
    return wrapper

# 函数的装饰器，用于获取计算时间（时）
def timer_decorator_hours(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Running time of {func.__name__}: {(end - start)/3600} hours")
        return result
    return wrapper

# 函数的装饰器，用于获取计算时间（秒，分，时），可将运行的时间写入文件
def timer_decorator_with_parameters(unit='second', print_show=1, write_file=0, filename='timer'):
    def timer_decorator(func):
        import time
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if unit == 'second':
                timer_text = f"Running time of {func.__name__}: {end - start} seconds"
            elif unit == 'minute':
                timer_text = f"Running time of {func.__name__}: {(end - start)/60} minutes"
            elif unit == 'hour':
                timer_text = f"Running time of {func.__name__}: {(end - start)/3600} hours"
            else:
                timer_text = f"Running time of {func.__name__}: {end - start} seconds"
            if print_show == 1:
                print(timer_text)
            if write_file == 1:
                with open(filename+'.txt', 'a') as f:
                    f.write(timer_text+'\n')
            return result
        return wrapper
    return timer_decorator

# 函数的装饰器，用于实现 try except 结构
def try_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            pass
    return wrapper

# # 函数的装饰器，用于GUAN软件包函数的使用统计
# def statistics_decorator(func):
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         import guan
#         guan.statistics_of_guan_package(func.__name__)
#         return result
#     return wrapper