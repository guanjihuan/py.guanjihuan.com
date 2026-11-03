"""Python 源码着色。

网站是静态生成的，所以高亮放在构建时做完：

  * 只用标准库，不引入 Pygments 之类的第三方依赖
  * 不依赖外部 JS，读者打开页面不用下载任何脚本
  * 输出的 HTML 只用 tok-* 类标记，配色在 static/style.css 里，
    深色 / 浅色主题各一套

高亮只影响外观：任何时候高亮失败都退回纯文本，代码内容不会变。
"""

from __future__ import annotations

import builtins
import html
import keyword
import re

# Python 关键字（含软关键字）与内置名
KEYWORDS = frozenset(keyword.kwlist) | frozenset(keyword.softkwlist)
BUILTINS = frozenset(dir(builtins)) | {"self", "cls"}

# 字符串前缀：r / b / u / f 及其组合
_STRING_RE = re.compile(r"(?P<prefix>[rRbBuUfF]{0,3})(?P<quote>'''|\"\"\"|'|\")")
# 数字：十六进制 / 二进制 / 八进制 / 小数 / 科学计数
_NUMBER_RE = re.compile(
    r"0[xX][0-9a-fA-F_]+"
    r"|0[bB][01_]+"
    r"|0[oO][0-7_]+"
    r"|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?[jJ]?"
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

# Markdown 围栏代码块里表示 Python 的语言标记
PYTHON_LANGUAGES = {"python", "py", "python2", "python3", "py3", "ipython"}


def _span(css_class: str, text: str) -> str:
    """把一段文本包进带类名的 span。"""
    return f'<span class="{css_class}">{html.escape(text)}</span>'


def _is_ident_start(char: str) -> bool:
    """是否是标识符的起始字符。

    只认 ASCII：Python 标识符可以含中文，但那些名字不需要着色，
    当成普通字符更稳妥。
    """
    return char.isascii() and (char.isalpha() or char == "_")


def _scan_string(code: str, start: int, quote: str) -> int:
    """从引号处往后找字符串结尾，返回结束位置（不含）。"""
    index = start + len(quote)
    size = len(code)
    while index < size:
        char = code[index]
        if char == "\\":
            index += 2
            continue
        if code.startswith(quote, index):
            return index + len(quote)
        index += 1
    return size


def highlight_python(code: str) -> str:
    """把 Python 源码转成带颜色标记的 HTML（内容已转义）。"""
    out: list[str] = []
    index = 0
    size = len(code)
    # 上一个关键字是 def 还是 class：接下来的名字是定义名
    expect_name = ""

    while index < size:
        char = code[index]

        # 注释
        if char == "#":
            end = code.find("\n", index)
            if end == -1:
                end = size
            out.append(_span("tok-com", code[index:end]))
            index = end
            continue

        # 字符串（可能带前缀）
        match = _STRING_RE.match(code, index)
        if match:
            quote = match.group("quote")
            end = _scan_string(code, match.start("quote"), quote)
            out.append(_span("tok-str", code[index:end]))
            index = end
            continue

        # 装饰器
        if char == "@" and index + 1 < size and _is_ident_start(code[index + 1]):
            match = _IDENT_RE.match(code, index + 1)
            end = match.end()
            out.append(_span("tok-dec", code[index:end]))
            index = end
            continue

        # 数字
        if (char.isascii() and char.isdigit()) or (
            char == "." and index + 1 < size and code[index + 1].isascii() and code[index + 1].isdigit()
        ):
            match = _NUMBER_RE.match(code, index)
            if match:
                out.append(_span("tok-num", match.group(0)))
                index = match.end()
                continue

        # 标识符：关键字、内置名、函数名
        if _is_ident_start(char):
            match = _IDENT_RE.match(code, index)
            word = match.group(0)
            end = match.end()
            if word in KEYWORDS:
                css_class = "tok-kw"
                expect_name = "def" if word == "def" else ("class" if word == "class" else "")
            elif expect_name:
                css_class = "tok-def"
                expect_name = ""
            elif word in BUILTINS:
                css_class = "tok-bi"
            else:
                # 后面紧跟左括号的是函数调用
                probe = end
                while probe < size and code[probe] in " \t":
                    probe += 1
                css_class = "tok-fn" if probe < size and code[probe] == "(" else ""
            out.append(_span(css_class, word) if css_class else html.escape(word))
            index = end
            continue

        # 其余字符（标点、运算符、空白）保持原样
        out.append(html.escape(char))
        index += 1

    return "".join(out)


def highlight(code: str, language: str = "") -> str:
    """按语言给代码着色；不认识的语言按纯文本处理。"""
    name = (language or "").strip().lower()
    if name in PYTHON_LANGUAGES:
        return highlight_python(code)
    return html.escape(code)


def code_block(code: str, language: str = "", css_class: str = "code") -> str:
    """生成一个代码块（与页面里其它代码块保持同样的结构）。"""
    name = (language or "").strip()
    lang_attr = f' class="language-{html.escape(name, quote=True)}"' if name else ""
    return f'<pre class="{css_class}"><code{lang_attr}>{highlight(code, name)}</code></pre>'
