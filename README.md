## Guan package

Guan is an open-source python package developed and maintained by https://www.guanjihuan.com/about (Ji-Huan Guan, 关济寰). With this package, you can calculate band structures, density of states, quantum transport and topological invariant of tight-binding models. Other frequently used functions are also integrated in this package, such as file reading/writing, figure plotting, data processing, etc.

The primary location of this package is on website https://py.guanjihuan.com. 

## Installation

pip install --upgrade guan

## Usage

import guan

## Summary of API Reference

+ basic functions
+ Fourier transform
+ Hamiltonian of finite size systems
+ Hamiltonian of models in the reciprocal space
+ band structures and wave functions
+ Green functions
+ density of states
+ quantum transport
+ topological invariant
+ read and write
+ plot figures
+ data processing
+ others


## About this package

+ The original motivation of this project is for self use. Based on this project, the frequent functions can be imported by “import guan” instead of repeated copies and pastes. You can also install and use this open-source package if some functions are helpful for you. If one function is not good enough, you can copy the source code and modify it. You can also feed back to guanjihuan@163.com. The modifications and supplements will be in the following updated version.
+ All realizations of this package are based on functions without any class, which are concise and convenient. The boring document is omitted and you have to read the Source Code for details if it is necessary. Nevertheless, you don’t have to be worried about the difficulty, because all functions are simple enough without too many prejudgments and the variable names are written commonly as far as possible for the easy reading.
+ Before the beginning of function call in your project, you are recommended to briefly read the Source Code to know the specific formats of input and output about the function. Applying functions mechanically may cause errors. Notice that as the package is developed, the function names may be changed in the future version. Therefore, the latest API Reference is important and helpful.