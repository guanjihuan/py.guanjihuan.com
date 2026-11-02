"""网站构建脚本的自检：用人造包检验通用性。

人造包刻意与本仓库的 guan 包结构不同，覆盖各种「源码大改」后可能出现的形态：

包一（mypkg）
  * 包名不同（mypkg）、目录布局不同（<root>/src/mypkg）
  * 元数据写在 pyproject.toml 的 [project] 表里（不是 setup.cfg）
  * 函数说明用 docstring，而不是定义上方的注释
  * 带类型注解、async 函数、@deprecated 装饰器
  * 有嵌套子包（mypkg.sub.extra）
  * 有私有函数（下划线开头）与显式 __all__
  * 有一个语法错误的文件（构建不能中断）

包二（edge）—— 各种边角情形
  * 空文件、带 BOM 的文件、CRLF 换行的文件、GBK 编码的文件
  * 文件名不是合法标识符（my-module.py）
  * 函数内部再定义函数（不应出现在 API 顶层）
  * 类里的 property 与 async 方法
  * `from . import helper` 形式的相对导入
  * README 里带本地图片、表格、任务列表
  * README 与源码里出现 {{version}} 这类模板占位符（不能被替换掉）

包三（empty）—— 什么都没有
  * 没有任何函数时，「API 参考」页应当自动消失

用法：
    python website/selftest.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_site  # noqa: E402
import extract  # noqa: E402
import i18n  # noqa: E402
import md_render  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


# --------------------------------------------------------------------------
# 包一：结构与人造包完全不同
# --------------------------------------------------------------------------

PYPROJECT = """\
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "mypkg"
version = "9.9.9"
description = "A synthetic package for testing"
requires-python = ">=3.10"
authors = [{name = "Tester", email = "tester@example.com"}]

[project.urls]
Homepage = "https://example.com/mypkg"
Repository = "https://github.com/example/mypkg"

[tool.setuptools]
packages = ["mypkg", "mypkg.sub"]
"""

INIT = '''\
"""A synthetic package."""

from .core import *
from .sub import *
from .extra_only import special

__all__ = ["greet", "VERSION", "special"]

VERSION = "9.9.9"
'''

CORE = '''\
"""Core helpers of the synthetic package."""


def greet(name: str = "world") -> str:
    """Return a greeting for *name*.

    This description comes from a docstring, not from a comment above.
    """
    return f"hello {name}"


def _hidden_helper(x: int) -> int:
    """Not exported because of the leading underscore."""
    return x


async def fetch(url: str) -> str:
    """An async function."""
    return url


@deprecated
def old_api():
    """Deprecated on purpose."""
    return 1


def table():
    """Return a literal table."""
    return {"alpha": 1, "beta": 2}


def use_greet():
    """Call another function through the package namespace."""
    import mypkg

    return mypkg.greet("there")
'''

EXTRA_ONLY = '''\
def special():
    """Only exported through an explicit import."""
    return 42
'''

SUB_INIT = '''\
"""A subpackage."""

from .extra import *
'''

SUB_EXTRA = '''\
def sub_function(x):
    """A function living in a nested subpackage."""
    return x


SUB_CONSTANTS = {
    "one": 1,
    "two": 2,
}
'''

BROKEN = '''\
def broken(
'''


def make_package(root: Path) -> Path:
    """在 root 下造出人造包，返回包源码目录。"""
    package = root / "src" / "mypkg"
    (package / "sub").mkdir(parents=True)
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (package / "__init__.py").write_text(INIT, encoding="utf-8")
    (package / "core.py").write_text(CORE, encoding="utf-8")
    (package / "extra_only.py").write_text(EXTRA_ONLY, encoding="utf-8")
    (package / "broken.py").write_text(BROKEN, encoding="utf-8")
    (package / "sub" / "__init__.py").write_text(SUB_INIT, encoding="utf-8")
    (package / "sub" / "extra.py").write_text(SUB_EXTRA, encoding="utf-8")
    (root / "README.md").write_text(
        "# mypkg\n\nA synthetic package.\n\n+ one\n+ two\n", encoding="utf-8"
    )
    (root / "INSTALLATION.md").write_text(
        "## Installation\n\n```bash\npip install --upgrade mypkg\n```\n",
        encoding="utf-8",
    )
    return package


# --------------------------------------------------------------------------
# 包二：边角情形
# --------------------------------------------------------------------------

EDGE_PYPROJECT = """\
[project]
name = "edge"
version = "0.1"
description = "Edge cases"
"""

EDGE_INIT = '''\
"""Edge case package."""

import os
import json as js

from . import helper
from .plain import plain_function
'''

EDGE_PLAIN = '''\
"""Plain module."""


def plain_function():
    """A plain function."""
    return 1


def outer():
    """A function with a nested function inside."""

    def inner():
        return 2

    return inner()


def tricky():
    """模板占位符 {{version}} 不能被替换；<script>alert(1)</script> 必须被转义。"""
    return 1


def plain():
    """Module-level function named plain."""
    return 0


def use_widget():
    """Call a class of the same package, to exercise cross references."""
    from .cls import Widget

    return Widget()


class Container:
    """A class whose method has the same name as a module-level function."""

    def plain(self):
        """Method named plain."""
        return 0
'''

EDGE_CLS = '''\
"""A class with fancy members."""


class Widget:
    """A widget."""

    kind = "widget"

    @property
    def size(self) -> int:
        """The size."""
        return 1

    async def refresh(self) -> None:
        """Refresh the widget."""

    def plain(self):
        """A plain method."""
        return 3
'''


def make_edge_package(root: Path) -> Path:
    """造出各种边角情形的包。"""
    package = root / "src" / "edge"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(EDGE_PYPROJECT, encoding="utf-8")

    (package / "__init__.py").write_text(EDGE_INIT, encoding="utf-8")
    (package / "plain.py").write_text(EDGE_PLAIN, encoding="utf-8")
    (package / "cls.py").write_text(EDGE_CLS, encoding="utf-8")
    (package / "helper.py").write_text("", encoding="utf-8")  # 空文件

    # 带 BOM 的文件
    (package / "bom.py").write_text(
        '"""With BOM."""\n\n\ndef bom_func():\n    """BOM function."""\n    return 1\n',
        encoding="utf-8-sig",
    )

    # CRLF 换行
    (package / "crlf.py").write_bytes(
        b'"""CRLF module."""\r\n\r\n\r\ndef crlf_func():\r\n    """CRLF function."""\r\n    return 1\r\n'
    )

    # GBK 编码
    (package / "gbk.py").write_bytes(
        '"""GBK 模块。"""\n\n\ndef gbk_func():\n    """中文说明。"""\n    return 1\n'.encode(
            "gbk"
        )
    )

    # 文件名不是合法标识符
    (package / "my-module.py").write_text(
        '"""Odd name."""\n\n\ndef odd_func():\n    """Odd function."""\n    return 1\n',
        encoding="utf-8",
    )

    # README：本地图片 + 表格 + 任务列表 + 模板占位符
    (root / "docs").mkdir()
    (root / "docs" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (root / "README.md").write_text(
        "# Edge\n\n"
        "![logo](docs/logo.png)\n\n"
        "版本号是 {{version}}，不要被替换掉。\n\n"
        "| 列 | 值 |\n|---|---|\n| a | 1 |\n| b |\n\n"
        "- [ ] 待办\n- [x] 完成\n",
        encoding="utf-8",
    )
    return package


# --------------------------------------------------------------------------
# 包四：元数据的各种退化情形
# --------------------------------------------------------------------------

META_INIT = '''\
"""Meta package."""

__version__ = "7.7.7"


def only():
    """Only one function."""
    return 1
'''


def make_meta_package(root: Path) -> Path:
    """没有可读元数据时，应当退回到目录名与 __init__.py 里的 __version__。"""
    package = root / "src" / "meta"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(META_INIT, encoding="utf-8")
    # 故意写坏的 pyproject.toml，不应让构建崩溃
    (root / "pyproject.toml").write_text(
        "[project\nname = broken\n", encoding="utf-8"
    )
    return package


# --------------------------------------------------------------------------
# 包五：setup.cfg 里用 attr: 写版本号
# --------------------------------------------------------------------------

ATTR_INIT = '''\
"""Attr package."""

__version__ = "5.6.7"


def only():
    """Only one function."""
    return 1
'''


def make_attr_package(root: Path) -> Path:
    package = root / "src" / "attrpkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(ATTR_INIT, encoding="utf-8")
    (root / "setup.cfg").write_text(
        "[metadata]\n"
        "name = attrpkg\n"
        "version = attr: attrpkg.__version__\n"
        "description = Version read from the package itself\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# attrpkg\n", encoding="utf-8")
    return package


# --------------------------------------------------------------------------
# 包六：只有类，没有模块级函数
# --------------------------------------------------------------------------

CLASS_ONLY_INIT = '''\
"""Only classes."""

from .thing import Thing
'''


def make_class_only_package(root: Path) -> Path:
    package = root / "src" / "classonly"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(CLASS_ONLY_INIT, encoding="utf-8")
    (package / "thing.py").write_text(
        '"""A thing."""\n'
        "\n"
        "\n"
        "class Thing:\n"
        '    """A thing."""\n'
        "\n"
        "    def go(self):\n"
        '        """Go."""\n'
        "        return 1\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# classonly\n", encoding="utf-8")
    return package


# --------------------------------------------------------------------------
# 包三：空包
# --------------------------------------------------------------------------


def make_empty_package(root: Path) -> Path:
    package = root / "src" / "empty"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "empty"\nversion = "0"\ndescription = "Nothing here"\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text('"""Nothing here."""\n', encoding="utf-8")
    (root / "README.md").write_text("# Empty\n", encoding="utf-8")
    return package


def check_meta(data: dict, output: Path) -> None:
    api = json.loads((output / "api.json").read_text(encoding="utf-8"))
    print("\n元数据退化")
    check(api["package"]["name"] == "meta", "元数据坏掉时退回目录名")
    check(api["package"]["version"] == "7.7.7", "版本退回 __init__.py 里的 __version__")
    check((output / "index.html").exists(), "元数据坏掉时网站照常生成")
    check(not build_site.verify_output(output), "生成的页面没有断链")


# --------------------------------------------------------------------------
# 检查项
# --------------------------------------------------------------------------


def build(package: Path, tmp: Path, name: str) -> tuple[dict, Path]:
    output = tmp / name
    data = build_site.generate(
        source_dir=package,
        output=output,
        readme_path=package.parent.parent / "README.md",
        installation_path=package.parent.parent / "INSTALLATION.md",
        # 人造包不带上仓库里真实的 tutorials/，免得混入无关内容
        tutorials_dir=package.parent.parent / "没有这个目录",
        quiet=True,
    )
    return data, output


def check_basic(data: dict, output: Path) -> None:
    api = json.loads((output / "api.json").read_text(encoding="utf-8"))

    print("\n包信息")
    check(api["package"]["name"] == "mypkg", "包名来自元数据")
    check(api["package"]["version"] == "9.9.9", "版本来自元数据")
    check(api["package"]["import_name"] == "mypkg", "导入名来自包目录")
    check(api["package"]["requires_python"] == ">=3.10", "requires-python 正确解析")
    check(
        api["package"]["urls"].get("Homepage") == "https://example.com/mypkg",
        "[project.urls] 表正确解析",
    )
    check(api["package"]["author"] == "Tester", "PEP 621 的 authors 正确解析")
    check(api["package"]["author_email"] == "tester@example.com", "作者邮箱正确解析")

    print("\n模块发现")
    names = [m["name"] for m in api["modules"]]
    check("core" in names, "识别到 core 模块")
    check("sub.extra" in names, "识别到嵌套子包模块 sub.extra")
    check("mypkg" in names, "包入口被识别为模块")
    check("broken" in names, "语法错误的文件仍被列出")

    print("\n函数提取")
    functions = {
        f["name"]: f
        for module in api["modules"]
        for f in module["functions"]
    }
    check("greet" in functions, "识别到函数 greet")
    check(
        "Return a greeting" in functions["greet"]["description"],
        "docstring 被用作函数说明（源码没有注释时）",
    )
    check(
        "This description comes from a docstring" in functions["greet"]["docstring"],
        "完整 docstring 保留在数据里",
    )
    check(functions["greet"]["params"][0]["annotation"] == "str", "参数类型注解解析")
    check(
        functions["greet"]["params"][0]["default"] in ('"world"', "'world'"),
        "默认值解析",
    )
    check(functions["greet"]["returns"] == "str", "返回类型解析")
    check(functions["fetch"]["is_async"], "async 函数被标记")
    check(functions["old_api"]["deprecated"], "@deprecated 装饰器被识别")
    check("_hidden_helper" in functions, "私有函数仍被提取（只是不导出）")
    check(
        functions["_hidden_helper"]["exported"] is False,
        "下划线开头的函数不被标记为导出",
    )
    check(functions["greet"]["exported"] is True, "导出的函数被标记")

    print("\n导出关系")
    exports = api["exports"]
    check("greet" in exports, "from .core import * 的名字被导出")
    check("sub_function" in exports, "子包的名字被导出")
    check("special" in exports, "显式 from .extra_only import special 被导出")
    check("_hidden_helper" not in exports, "私有名字不在导出列表里")

    print("\n数据表")
    tables = [table for module in api["modules"] for table in module["data"]]
    table_names = {t["name"] for t in tables}
    check("table" in table_names, "函数返回的字面量表被识别")
    check("SUB_CONSTANTS" in table_names, "模块级字面量表被识别")
    check(any(t["origin"] == "module" for t in tables), "模块级表被标注来源")

    print("\n容错")
    broken = next(m for m in api["modules"] if m["name"] == "broken")
    check(bool(broken["error"]), "语法错误被记录而不是抛异常")

    print("\n页面生成")
    for filename in ("index.html", "api.html", "about.html", "installation.html"):
        check((output / filename).exists(), f"生成了 {filename}")
    check(not (output / "modules.html").exists(), "不再生成模块页")
    check(not (output / "data.html").exists(), "不再生成数据页")
    installation = (output / "installation.html").read_text(encoding="utf-8")
    check(
        "pip install --upgrade mypkg" in installation,
        "安装页用真实的包名渲染 INSTALLATION.md",
    )
    index = (output / "index.html").read_text(encoding="utf-8")
    check("mypkg" in index, "首页出现包名")
    check("guan." not in index, "首页没有残留旧包名")
    check("import guan" not in index, "首页没有残留旧包的导入写法")
    check("pip install --upgrade mypkg" in index, "安装命令用真实的包名")
    check('href="installation.html"' in index, "首页有指向安装页的链接")
    api_html = (output / "api.html").read_text(encoding="utf-8")
    check("mypkg.greet" in api_html, "交叉引用使用真实的导入名")
    check("guan." not in api_html, "API 页没有残留旧包名")
    check(not build_site.verify_output(output), "生成的页面没有断链")


def check_edge(data: dict, output: Path) -> None:
    api = json.loads((output / "api.json").read_text(encoding="utf-8"))
    names = [m["name"] for m in api["modules"]]

    print("\n各种编码与文件形态")
    check("helper" in names, "空文件也被识别为模块")
    check("bom" in names, "带 BOM 的文件被识别")
    check("crlf" in names, "CRLF 换行的文件被识别")
    check("gbk" in names, "GBK 编码的文件被识别")
    check("my-module" in names, "文件名不是标识符时也能识别")

    functions = {
        f["name"]: f
        for module in api["modules"]
        for f in module["functions"]
    }
    check("bom_func" in functions, "BOM 文件里的函数被解析")
    check("crlf_func" in functions, "CRLF 文件里的函数被解析")
    check("gbk_func" in functions, "GBK 文件里的函数被解析")
    check(
        functions.get("gbk_func", {}).get("description") == "中文说明。",
        "GBK 文件里的中文说明没有乱码",
    )
    check("odd_func" in functions, "特殊文件名的模块里的函数被解析")
    check("inner" not in functions, "函数内部的嵌套函数不出现在 API 顶层")

    print("\n类")
    classes = {
        c["name"]: c
        for module in api["modules"]
        for c in module["classes"]
    }
    check("Widget" in classes, "识别到类")
    widget = classes.get("Widget", {})
    method_names = {m["name"] for m in widget.get("methods", [])}
    check("size" in method_names, "property 被当作方法记录")
    check("plain" in method_names, "普通方法被记录")
    check(
        any(m["name"] == "refresh" and m["is_async"] for m in widget.get("methods", [])),
        "类里的 async 方法被标记",
    )
    check("kind" in widget.get("attributes", []), "类属性被记录")

    print("\n相对导入")
    check("helper" in api["exports"], "from . import helper 被识别为导出")
    check("os" not in api["exports"], "第三方库（import os）不会被当成导出")
    check("js" not in api["exports"], "第三方库的别名（import json as js）不会被当成导出")

    print("\nREADME 与资源")
    about = (output / "about.html").read_text(encoding="utf-8")
    check("{{version}}" in about, "README 里的 {{version}} 没有被模板替换掉")
    check((output / "docs" / "logo.png").exists(), "README 引用的本地图片被复制过来")
    check("docs/logo.png" in about, "图片的相对路径被保留")
    check("<td></td>" in about, "表格缺失的单元格被补齐")
    check("checkbox" in about, "任务列表被正确渲染")
    check(not build_site.verify_output(output), "生成的页面没有断链")

    print("\n源码里的模板占位符")
    api_html = (output / "api.html").read_text(encoding="utf-8")
    check("{{version}}" in api_html, "源码里的 {{version}} 没有被替换掉")
    check(
        "<script>alert(1)</script>" not in api_html,
        "docstring 里的 HTML 被转义，不会被当成标签执行",
    )
    check("&lt;script&gt;" in api_html, "转义后的 HTML 实体出现在页面上")

    print("\n同名函数")
    duplicated = [
        module
        for module in api["modules"]
        if any(f["name"] == "plain_function" for f in module["functions"])
    ]
    check(len(duplicated) >= 1, "同名函数被分别记录")
    ids = re.findall(r'id="([^"]+)"', api_html)
    check(len(ids) == len(set(ids)), "页面里的 id 没有重复")
    check("plain.Container.plain" in ids, "同名方法与函数各有各的锚点")
    check("plain.plain" in ids, "模块级函数的锚点没有被方法覆盖")
    check("cls.Widget" in ids, "类本身也有锚点")

    print("\n交叉参考")
    use_widget = next(
        f
        for module in api["modules"]
        for f in module["functions"]
        if f["name"] == "use_widget"
    )
    check("Widget" in use_widget["calls"], "直接 import 进来的包内函数也被认出来")
    check('<a href="#cls.Widget">' in api_html, "交叉参考指向类自己的锚点")
    check(not build_site.verify_output(output), "生成的页面没有断链")


def check_i18n(data: dict, output: Path) -> None:
    print("\n中英文界面")
    check(not i18n.missing_keys(), "两种语言的文案都写全了")

    for filename in ("index.html", "api.html", "about.html", "installation.html"):
        if not (output / filename).exists():
            continue  # 中文页面里就没有的（比如没有 INSTALLATION.md），英文也不该有
        check(
            (output / "en" / filename).exists(),
            f"生成了 en/{filename}",
        )

    zh_index = (output / "index.html").read_text(encoding="utf-8")
    en_index = (output / "en" / "index.html").read_text(encoding="utf-8")

    check('<html lang="zh-CN">' in zh_index, "中文页面声明了语言")
    check('<html lang="en">' in en_index, "英文页面声明了语言")
    check(">首页<" in zh_index, "中文页面用中文导航")
    check(">Home<" in en_index, "英文页面用英文导航")
    check("模块一览" in zh_index, "中文页面的界面文字是中文")
    check("模块一览" not in en_index, "英文页面的界面文字是英文")
    check(">Modules<" in en_index, "英文页面有对应的英文标题")

    print("\n语言切换")
    check('href="en/"' in zh_index, "中文页面链到英文页面")
    check('href="../"' in en_index, "英文页面链回中文页面")
    zh_api = (output / "api.html").read_text(encoding="utf-8")
    en_api = (output / "en" / "api.html").read_text(encoding="utf-8")
    check('href="en/api.html"' in zh_api, "API 页也能切到另一种语言")
    check('href="../api.html"' in en_api, "英文 API 页能切回中文")

    print("\n地址里不带 index.html")
    zh_canonical = re.search(r'canonical" href="([^"]*)"', zh_index)
    en_canonical = re.search(r'canonical" href="([^"]*)"', en_index)
    check(
        bool(zh_canonical) and not zh_canonical.group(1).endswith("index.html"),
        "中文首页的规范地址不带 index.html",
    )
    check(
        bool(en_canonical) and not en_canonical.group(1).endswith("index.html"),
        "英文首页的规范地址不带 index.html",
    )
    check('href="./" class="is-active"' in zh_index, "首页链接指向目录")
    stub = output / "about" / "index.html"
    check(stub.exists(), "生成了不带 .html 的入口（/about）")
    if stub.exists():
        stub_text = stub.read_text(encoding="utf-8")
        check("../about.html" in stub_text, "入口跳到对应的 .html 页面")
        stub_canonical = re.search(r'canonical" href="([^"]*)"', stub_text)
        check(
            bool(stub_canonical) and stub_canonical.group(1).endswith("about.html"),
            "入口的规范地址指向 .html 页面",
        )

    print("\n文件内容保持原样")
    # README、源码里的说明不翻译：两种语言的页面里都应当原样出现
    zh_about = (output / "about.html").read_text(encoding="utf-8")
    en_about = (output / "en" / "about.html").read_text(encoding="utf-8")
    check("版本号是 {{version}}" in zh_about, "中文页面里是 README 原文")
    check("版本号是 {{version}}" in en_about, "英文页面里也是 README 原文")
    check(
        "A plain function." in en_api,
        "英文页面里保留源码里的英文说明",
    )
    check("The size." in en_api, "英文页面里保留 docstring 原文")

    print("\n两种语言的链接都是通的")
    check(not build_site.verify_output(output), "生成的页面没有断链")
    check((output / "en" / "static" / "style.css").exists(), "英文目录里有自己的样式")
    check((output / "en" / "docs" / "logo.png").exists(), "英文目录里也有 README 的图片")


def check_cli(tmp: Path) -> None:
    print("\n命令行与异常处理")
    try:
        build_site.generate(source_dir=tmp / "does-not-exist", output=tmp / "nope")
        check(False, "不存在的源码目录应当报错")
    except SystemExit:
        check(True, "不存在的源码目录会给出明确报错而不是崩溃")
    except Exception as error:  # noqa: BLE001
        check(False, f"不存在的源码目录应当报 SystemExit，实际是 {error!r}")

    try:
        build_site.generate(source_dir=tmp / "not-a-dir.txt", output=tmp / "nope2")
        check(False, "指向文件的源码路径应当报错")
    except SystemExit:
        check(True, "源码路径不是目录时会给出明确报错")
    except Exception as error:  # noqa: BLE001
        check(False, f"源码路径不是目录时应当报 SystemExit，实际是 {error!r}")

    print("\n断链检查本身有效")
    broken = tmp / "broken-site"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "index.html").write_text(
        '<a href="missing.html">x</a><a href="#nope">y</a>', encoding="utf-8"
    )
    problems = build_site.verify_output(broken)
    check(len(problems) == 2, "断链与失效锚点都能被查出来")


def check_empty(data: dict, output: Path) -> None:
    print("\n没有内容时的页面取舍")
    check(not (output / "api.html").exists(), "没有函数时不生成 API 参考页")
    check(not (output / "modules.html").exists(), "不再生成模块页")
    check(not (output / "data.html").exists(), "不再生成数据页")
    check((output / "index.html").exists(), "首页仍然生成")
    check(
        not (output / "installation.html").exists(),
        "没有 INSTALLATION.md 时不生成安装页",
    )
    index = (output / "index.html").read_text(encoding="utf-8")
    check("api.html" not in index, "首页没有链到不存在的 API 页")
    check(
        "installation.html" not in index,
        "首页没有链到不存在的安装页",
    )
    check("modules.html" not in index, "首页没有链到已经不存在的模块页")
    check("data.html" not in index, "首页没有链到已经不存在的数据页")
    check(not build_site.verify_output(output), "生成的页面没有断链")


def check_attr(data: dict, output: Path) -> None:
    api = json.loads((output / "api.json").read_text(encoding="utf-8"))
    print("\n版本号的 attr: 写法")
    check(api["package"]["version"] == "5.6.7", "attr: 形式的版本号被解析出来")
    check(
        api["package"]["description"] == "Version read from the package itself",
        "setup.cfg 里的描述被正确读取",
    )
    check(not build_site.verify_output(output), "生成的页面没有断链")


def check_class_only(data: dict, output: Path) -> None:
    print("\n只有类的时候")
    check((output / "api.html").exists(), "只有类时也会生成 API 参考页")
    api_html = (output / "api.html").read_text(encoding="utf-8")
    check("thing.Thing.go" in api_html, "类的方法有正确的锚点")
    check(not build_site.verify_output(output), "生成的页面没有断链")


TUTORIALS_README = """\
# Tutorials

Some examples.

## 一、日常场景 / Daily scenarios

| File | 说明 | Requirements |
| --- | --- | --- |
| `01_hello.py` | 打个招呼 | numpy |
"""

TUTORIALS_SCRIPT = """\
# 1. 打个招呼
#
# 来源 / Source: https://example.com/post
#
import numpy as np
print(np.zeros(3))
"""


def make_tutorials(root: Path) -> Path:
    """造一个 tutorials/ 目录。"""
    directory = root / "tutorials"
    (directory / "daily_scenarios").mkdir(parents=True, exist_ok=True)
    (directory / "README.md").write_text(TUTORIALS_README, encoding="utf-8")
    (directory / "daily_scenarios" / "01_hello.py").write_text(
        TUTORIALS_SCRIPT, encoding="utf-8"
    )
    return directory


def check_tutorials(tmp: Path) -> None:
    print("\n教程页")
    tutorials_dir = make_tutorials(tmp / "t")
    package = make_package(tmp / "t")
    output = tmp / "t" / "site-t"
    data = build_site.generate(
        source_dir=package,
        output=output,
        readme_path=tmp / "t" / "README.md",
        installation_path=tmp / "t" / "INSTALLATION.md",
        tutorials_dir=tutorials_dir,
        quiet=True,
    )

    check((output / "tutorials.html").exists(), "有 tutorials/ 时生成教程页")
    html = (output / "tutorials.html").read_text(encoding="utf-8")
    check("一、日常场景" in html, "分组标题来自 README 的章节")
    check("打个招呼" in html, "脚本标题与说明被收进来")
    check("np.zeros(3)" in html, "脚本源码被收进来")
    check(
        "href=\"tutorials/daily_scenarios/01_hello.py\"" in html,
        "页面给出脚本的下载地址",
    )
    check(
        (output / "tutorials" / "daily_scenarios" / "01_hello.py").exists(),
        "脚本被复制到站点里",
    )

    api = json.loads((output / "api.json").read_text(encoding="utf-8"))
    check(bool(api.get("tutorials", {}).get("files")), "api.json 里带上教程索引")
    check(
        "code" not in api["tutorials"]["groups"][0]["files"][0],
        "api.json 里不重复存整段源码",
    )
    check(data["tutorials"]["count"] == 1, "教程数量统计正确")
    check(not build_site.verify_output(output), "生成的页面没有断链")


def check_markdown() -> None:
    print("\nMarkdown 渲染")
    html = md_render.render("- a\n1. b\n")
    check("<ul>" in html and "<ol>" in html, "不同类型的列表分开渲染")
    html = md_render.render("- a\n  - b\n    - c\n")
    check(html.count("<ul>") == 3, "多层嵌套列表不会被当成代码块")
    html = md_render.render("[x](https://example.com/a_(b))")
    check('href="https://example.com/a_(b)"' in html, "链接地址里的括号不会被截断")
    html = md_render.render("see https://example.com/a_(b).")
    check('href="https://example.com/a_(b)"' in html, "裸露网址里的括号与句号处理正确")
    html = md_render.render("| a | b |\n|---|---|\n| 1 |\n")
    check("<td></td>" in html, "表格缺失的单元格被补齐")
    check("<blockquote>" in md_render.render("> 引用\n"), "引用块被渲染")
    check(
        "<em>x</em>" in md_render.render("*x*\n"),
        "斜体被渲染",
    )


# --------------------------------------------------------------------------


def run() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="site-selftest-"))
    try:
        print("== 人造包一：结构完全不同 ==")
        data, output = build(make_package(tmp / "a"), tmp / "a", "site-a")
        check_basic(data, output)

        print("\n== 人造包二：边角情形 ==")
        data, output = build(make_edge_package(tmp / "b"), tmp / "b", "site-b")
        check_edge(data, output)

        print("\n== 人造包三：什么都没有 ==")
        data, output = build(make_empty_package(tmp / "c"), tmp / "c", "site-c")
        check_empty(data, output)

        print("\n== 人造包四：元数据退化 ==")
        data, output = build(make_meta_package(tmp / "d"), tmp / "d", "site-d")
        check_meta(data, output)

        print("\n== 人造包五：attr: 形式的版本号 ==")
        data, output = build(make_attr_package(tmp / "e"), tmp / "e", "site-e")
        check_attr(data, output)

        print("\n== 人造包六：只有类 ==")
        data, output = build(make_class_only_package(tmp / "f"), tmp / "f", "site-f")
        check_class_only(data, output)

        print("\n== 自动发现 ==")
        discovered = extract.discover_package_dir([tmp / "a" / "src", tmp / "a"])
        check(
            discovered is not None and discovered.name == "mypkg",
            "自动发现包目录",
        )
        check(
            extract.discover_package_dir([Path(__file__).resolve().parent]) is None,
            "网站脚本目录不会被误认成包",
        )

        print("\n== 教程页 ==")
        check_tutorials(tmp)

        print("\n== 中英文界面 ==")
        data, output = build(make_edge_package(tmp / "g"), tmp / "g", "site-g")
        check_i18n(data, output)

        print("\n== Markdown ==")
        check_markdown()
        print("\n== 命令行与异常处理 ==")
        check_cli(tmp)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} 项检查未通过：")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("全部检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
