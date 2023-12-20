# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about (Ji-Huan Guan, 关济寰). The primary location of this package is on website https://py.guanjihuan.com. The GitHub location of this package is on https://github.com/guanjihuan/py.guanjihuan.com.

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

# 函数的装饰器，用于GUAN软件包的统计
def statistics_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        import guan
        guan.statistics_of_guan_package(func.__name__)
        return result
    return wrapper

from .basic_functions import *
from .Fourier_transform import *
from .Hamiltonian_of_examples import *
from .band_structures_and_wave_functions import *
from .Green_functions import *
from .density_of_states import *
from .quantum_transport import *
from .topological_invariant import *
from .machine_learning import *
from .data_processing import *
from .others import *