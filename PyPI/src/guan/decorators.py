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

# 函数的装饰器，用于GUAN软件包函数的使用统计
def statistics_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        import guan
        guan.statistics_of_guan_package(func.__name__)
        return result
    return wrapper