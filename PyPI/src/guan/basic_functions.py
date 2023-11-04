# Module: basic_functions

# 测试
def test():
    print('\nSuccess in the installation of Guan package!\n')
    import guan
    guan.statistics_of_guan_package()

# 泡利矩阵
def sigma_0():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.eye(2)

def sigma_x():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.array([[0, 1],[1, 0]])

def sigma_y():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.array([[0, -1j],[1j, 0]])

def sigma_z():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.array([[1, 0],[0, -1]])

# 泡利矩阵的张量积
def sigma_00():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_0(), guan.sigma_0())

def sigma_0x():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_0(), guan.sigma_x())

def sigma_0y():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_0(), guan.sigma_y())

def sigma_0z():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_0(), guan.sigma_z())

def sigma_x0():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_x(), guan.sigma_0())

def sigma_xx():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_x(), guan.sigma_x())

def sigma_xy():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_x(), guan.sigma_y())

def sigma_xz():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_x(), guan.sigma_z())

def sigma_y0():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_y(), guan.sigma_0())

def sigma_yx():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_y(), guan.sigma_x())

def sigma_yy():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_y(), guan.sigma_y())

def sigma_yz():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_y(), guan.sigma_z())

def sigma_z0():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_z(), guan.sigma_0())

def sigma_zx():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_z(), guan.sigma_x())

def sigma_zy():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_z(), guan.sigma_y())

def sigma_zz():
    import numpy as np
    import guan
    guan.statistics_of_guan_package()
    return np.kron(guan.sigma_z(), guan.sigma_z())