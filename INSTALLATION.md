## Installation

Press the shortcut key "Win+R", type words "cmd", press "Enter", then open a terminal in Windows.

Execute the following command in a terminal to install (upgrade) Guan package to the latest version (https://pypi.org/project/guan/):

```bash
pip install --upgrade guan -i https://pypi.python.org/simple
```

or

```bash
pip install --upgrade guan
```

See the installed version of Guan package on your computer:

```bash
pip show guan
```

If you do not use this package anymore, you can uninstall it:

```bash
pip uninstall guan
```

Usage:

```python
import guan
```

Prerequisites:

```text
Python3

[Packages partially used, which have been included in Anaconda]
numpy
math
cmath
os
copy
matplotlib
re
datetime
bs4
requests
functools
sympy
collections
pickle
time
urllib
hashlib
shutil
logging
ssl
random
json
...

[Packages partially used, which need to be additionally installed when they are needed]
jieba (pip install jieba)
imageio (pip install imageio)
pdfplumber (pip install pdfplumber)
PyPDF2 (pip install PyPDF2)
pdfminer3k (pip install pdfminer3k)
pygame (pip install pygame)
akshare (pip install akshare)
pyttsx3 (pip install pyttsx3)
pydub (pip install pydub)
qrcode (pip install qrcode)
FFmpeg (http://www.ffmpeg.org/download.html)
...
```
