# Module: basic_functions
import guan

# 测试
@guan.function_decorator
def test():
    import guan
    current_version = guan.get_current_version('guan')
    print(f'Success in the installation of Guan package! The installed version is guan-{current_version}.')
    guan.notification_of_upgrade(timeout=5)

# 泡利矩阵
@guan.function_decorator
def sigma_0():
    import numpy as np
    return np.eye(2)

@guan.function_decorator
def sigma_x():
    import numpy as np
    return np.array([[0, 1],[1, 0]])

@guan.function_decorator
def sigma_y():
    import numpy as np
    return np.array([[0, -1j],[1j, 0]])

@guan.function_decorator
def sigma_z():
    import numpy as np
    return np.array([[1, 0],[0, -1]])

# 泡利矩阵的张量积
@guan.function_decorator
def sigma_00():
    import numpy as np
    import guan
    return np.kron(guan.sigma_0(), guan.sigma_0())

@guan.function_decorator
def sigma_0x():
    import numpy as np
    import guan
    return np.kron(guan.sigma_0(), guan.sigma_x())

@guan.function_decorator
def sigma_0y():
    import numpy as np
    import guan
    return np.kron(guan.sigma_0(), guan.sigma_y())

@guan.function_decorator
def sigma_0z():
    import numpy as np
    import guan
    return np.kron(guan.sigma_0(), guan.sigma_z())

@guan.function_decorator
def sigma_x0():
    import numpy as np
    import guan
    return np.kron(guan.sigma_x(), guan.sigma_0())

@guan.function_decorator
def sigma_xx():
    import numpy as np
    import guan
    return np.kron(guan.sigma_x(), guan.sigma_x())

@guan.function_decorator
def sigma_xy():
    import numpy as np
    import guan
    return np.kron(guan.sigma_x(), guan.sigma_y())

@guan.function_decorator
def sigma_xz():
    import numpy as np
    import guan
    return np.kron(guan.sigma_x(), guan.sigma_z())

@guan.function_decorator
def sigma_y0():
    import numpy as np
    import guan
    return np.kron(guan.sigma_y(), guan.sigma_0())

@guan.function_decorator
def sigma_yx():
    import numpy as np
    import guan
    return np.kron(guan.sigma_y(), guan.sigma_x())

@guan.function_decorator
def sigma_yy():
    import numpy as np
    import guan
    return np.kron(guan.sigma_y(), guan.sigma_y())

@guan.function_decorator
def sigma_yz():
    import numpy as np
    import guan
    return np.kron(guan.sigma_y(), guan.sigma_z())

@guan.function_decorator
def sigma_z0():
    import numpy as np
    import guan
    return np.kron(guan.sigma_z(), guan.sigma_0())

@guan.function_decorator
def sigma_zx():
    import numpy as np
    import guan
    return np.kron(guan.sigma_z(), guan.sigma_x())

@guan.function_decorator
def sigma_zy():
    import numpy as np
    import guan
    return np.kron(guan.sigma_z(), guan.sigma_y())

@guan.function_decorator
def sigma_zz():
    import numpy as np
    import guan
    return np.kron(guan.sigma_z(), guan.sigma_z())