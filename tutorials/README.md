# Tutorials

Learning examples for the [Guan package](https://py.guanjihuan.com), extracted from the blog post
[Guan软件包学习示例](https://www.guanjihuan.com/archives/38367).

One example per file. Each file is a self-contained script that you can run directly, or copy the
functions you need into your own project.

## Installation

```bash
pip install --upgrade guan
```

```python
import guan
```

## Run the examples

```bash
python tutorials/daily_scenarios/01_ai_chat.py
python tutorials/physics_scenarios/03_band_structure.py
```

## Directory

```
tutorials/
├── 00_test.py                                  # package self test
├── daily_scenarios/                            # 一、日常场景
│   ├── 01_ai_chat.py
│   ├── 02_timing_log.py
│   ├── 03_function_timer.py
│   ├── 04_try_except.py
│   ├── 05_dump_and_load_data.py
│   ├── 06_print_array.py
│   ├── 07_matrix_in_file.py
│   ├── 08_parallel_calculation.py
│   ├── 09_loop_calculation.py
│   ├── 10_is_hermitian.py
│   └── 11_split_parameters.py
└── physics_scenarios/                          # 二、物理场景
    ├── 01_pauli_matrices.py
    ├── 02_real_space_hamiltonian.py
    ├── 03_band_structure.py
    ├── 04_chern_number_and_wilson_loop.py
    ├── 05_density_of_states_with_green_function.py
    ├── 06_conductance_and_scattering_matrix.py
    └── 07_wavefunction_gauge.py
```

## 一、日常场景 / Daily scenarios

| File | 说明 | Requirements |
| --- | --- | --- |
| `00_test.py` | 软件包自检 `guan.test()` | - |
| `01_ai_chat.py` | AI 对话：内置模型、OpenAI 兼容接口、LangChain、Ollama、llama.cpp、ModelScope | OpenAI 接口需要 API Key；Ollama / llama.cpp / ModelScope 需要本地模型 |
| `02_timing_log.py` | 带日期和时间的运行日志 `guan.logging_with_day_and_time` | - |
| `03_function_timer.py` | 函数计时器：`@guan.timer_decorator` 与 `guan.timer` | - |
| `04_try_except.py` | try except 简写：`@guan.try_decorator` 与 `guan.try_except` | - |
| `05_dump_and_load_data.py` | 变量写入文件并读回 | numpy |
| `06_print_array.py` | 打印数组 | numpy |
| `07_matrix_in_file.py` | 矩阵写入 Markdown / LaTeX 格式文件后查看 | numpy |
| `08_parallel_calculation.py` | 多进程并行计算 | - |
| `09_loop_calculation.py` | 循环参数计算（一、二、三个参数）与作图 | numpy, matplotlib |
| `10_is_hermitian.py` | 检查是否为厄米矩阵（相对误差 1e-5） | numpy |
| `11_split_parameters.py` | 数组分割，为并行计算准备参数 | numpy |

## 二、物理场景 / Physics scenarios

| File | 说明 | Requirements |
| --- | --- | --- |
| `01_pauli_matrices.py` | 泡利矩阵 | numpy |
| `02_real_space_hamiltonian.py` | 一维 / 二维方格 / 三维立方格的实空间哈密顿量 | numpy |
| `03_band_structure.py` | 能带图计算：一维链、方格带、石墨烯锯齿形带 | numpy, matplotlib |
| `04_chern_number_and_wilson_loop.py` | 陈数与 Wilson loop 计算 | numpy |
| `05_density_of_states_with_green_function.py` | 用格林函数计算态密度（方格与立方格，局域态密度） | numpy, matplotlib |
| `06_conductance_and_scattering_matrix.py` | 电导与散射矩阵的计算 | numpy, matplotlib |
| `07_wavefunction_gauge.py` | 波函数规范（fixed gauge）的选取 | numpy |

## 说明 / Notes

+ Some examples need an API Key, a local Ollama/llama.cpp/ModelScope model, or a specific model path.
  Edit the paths in the scripts before running them.
+ `04_try_except.py` contains a deliberate `bug_code` call to demonstrate the exception handling.
+ Figures are saved to the current working directory when the plotting functions are used.

## License / 许可

The source post is published under [CC BY](https://www.guanjihuan.com), by
[Ji-Huan Guan](https://www.guanjihuan.com/about). Please cite the source when quoting:
https://www.guanjihuan.com/archives/38367
