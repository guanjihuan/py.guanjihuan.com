# Module: basic_functions

# 测试
def test():
    import guan
    current_version = guan.get_current_version('guan')
    print(f'Congratulations on successfully installing Guan package! The installed version is guan-{current_version}.')

# 基本常数
def fundamental_constants():
    constants = {}
    constants['c'] = 2.99792458e8             # 真空光速，单位：m/s（米每秒）
    constants['h'] = 6.62607015e-34           # 普朗克常数，单位：J·s（焦耳·秒）
    constants['hbar'] = 1.054571817e-34       # 约化普朗克常数 ħ = h / (2π)，单位：J·s（焦耳·秒）
    constants['q'] = 1.602176634e-19          # 基本电荷（元电荷），单位：C（库仑）
    constants['eV'] = 1.602176634e-19         # 1 电子伏特（eV）对应的焦耳值，即 1 eV = q J
    constants['k'] = 1.380649e-23             # 玻尔兹曼常数，单位：J/K（焦耳每开尔文）
    constants['NA'] = 6.02214076e23           # 阿伏伽德罗常数，单位：mol⁻¹（每摩尔的粒子数）
    constants['G'] = 6.67430e-11              # 万有引力常数，单位：m³/(kg·s²)
    constants['mu0'] = 1.25663706212e-6       # 真空磁导率 μ₀ ≈ 4π × 10⁻⁷，单位：H/m（亨利每米）
    constants['eps0'] = 8.8541878128e-12      # 真空介电常数 ε₀ = 1 / (μ₀ c²)，单位：F/m（法拉每米）
    constants['me'] = 9.1093837015e-31        # 电子静止质量，单位：kg（千克）
    constants['mp'] = 1.67262192595e-27       # 质子静止质量，单位：kg（千克）
    constants['mn'] = 1.67492749804e-27       # 中子静止质量，单位：kg（千克）
    constants['alpha'] = 7.2973525693e-3      # 精细结构常数 α = e² / (4π ε₀ ħ c) ≈ 1/137，无量纲
    constants['a0'] = 5.29177210903e-11       # 玻尔半径，单位：m
    constants['T0'] = 273.15                  # 标准温度参考点（0°C 对应的开尔文温度偏移），单位：K（开尔文）
    return constants

# 泡利矩阵
def sigma_0():
    import numpy as np
    return np.eye(2)

def sigma_x():
    import numpy as np
    return np.array([[0, 1],[1, 0]])

def sigma_y():
    import numpy as np
    return np.array([[0, -1j],[1j, 0]])

def sigma_z():
    import numpy as np
    return np.array([[1, 0],[0, -1]])

# 泡利矩阵的张量积
def sigma_00():
    import numpy as np
    import guan
    return np.kron(guan.sigma_0(), guan.sigma_0())

def sigma_0x():
    import numpy as np
    import guan
    return np.kron(guan.sigma_0(), guan.sigma_x())

def sigma_0y():
    import numpy as np
    import guan
    return np.kron(guan.sigma_0(), guan.sigma_y())

def sigma_0z():
    import numpy as np
    import guan
    return np.kron(guan.sigma_0(), guan.sigma_z())

def sigma_x0():
    import numpy as np
    import guan
    return np.kron(guan.sigma_x(), guan.sigma_0())

def sigma_xx():
    import numpy as np
    import guan
    return np.kron(guan.sigma_x(), guan.sigma_x())

def sigma_xy():
    import numpy as np
    import guan
    return np.kron(guan.sigma_x(), guan.sigma_y())

def sigma_xz():
    import numpy as np
    import guan
    return np.kron(guan.sigma_x(), guan.sigma_z())

def sigma_y0():
    import numpy as np
    import guan
    return np.kron(guan.sigma_y(), guan.sigma_0())

def sigma_yx():
    import numpy as np
    import guan
    return np.kron(guan.sigma_y(), guan.sigma_x())

def sigma_yy():
    import numpy as np
    import guan
    return np.kron(guan.sigma_y(), guan.sigma_y())

def sigma_yz():
    import numpy as np
    import guan
    return np.kron(guan.sigma_y(), guan.sigma_z())

def sigma_z0():
    import numpy as np
    import guan
    return np.kron(guan.sigma_z(), guan.sigma_0())

def sigma_zx():
    import numpy as np
    import guan
    return np.kron(guan.sigma_z(), guan.sigma_x())

def sigma_zy():
    import numpy as np
    import guan
    return np.kron(guan.sigma_z(), guan.sigma_y())

def sigma_zz():
    import numpy as np
    import guan
    return np.kron(guan.sigma_z(), guan.sigma_z())