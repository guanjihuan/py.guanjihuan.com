# Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about (Ji-Huan Guan, 关济寰). The primary location of this package is on website https://py.guanjihuan.com. The GitHub location of this package is on https://github.com/guanjihuan/py.guanjihuan.com.

# 函数的装饰器，用于软件包的统计
def function_decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        import guan
        guan.statistics_of_guan_package(func.__name__)
    return wrapper

from .basic_functions import *
from .Fourier_transform import *
from .Hamiltonian_of_examples import *
from .band_structures_and_wave_functions import *
from .Green_functions import *
from .density_of_states import *
from .quantum_transport import *
from .topological_invariant import *
from .data_processing import *
from .others import *