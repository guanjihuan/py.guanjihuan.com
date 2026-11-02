"""从包源码树中静态提取 API 数据。

本模块只做「静态解析」：读源码文本 + ast 解析，不 import 被解析的包。
因此不需要安装 numpy / torch / transformers 等重依赖，
也不会触发包内 statistics_of_guan_package() 的联网行为。

提取的内容全部来自源码本身，**不假设任何固定的包名、目录名、模块名或函数名**，
这样即使源码结构或内容大幅变动，网站仍然能自动识别并生成：

  * 包目录自动发现（支持 src/ 布局、任意包名、嵌套子包）
  * 模块名 / 模块头注释（形如 "# Module: basic_functions"），缺失时回退到模块名
  * 每个函数的签名、参数与默认值、装饰器、返回类型
  * 函数说明：优先取定义正上方紧邻的注释块，其次取 docstring
  * 分节注释（形如 "# --- 某某某 ---"）
  * 类及其方法、基类
  * 函数体内对包内其它函数的调用（用于生成交叉参考）
  * 弃用标记（函数体中出现 'deprecated' 提示，或 @deprecated 装饰器）
  * 字面量形式的数据表（函数返回的，或模块级定义的 dict / list）
  * 包的导出关系：__init__.py 中的 import * / 显式导入 / __all__
  * 解析失败的文件不会中断构建，只会被记录为错误
"""

from __future__ import annotations

import ast
import configparser
import json
import re
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# "# --- 标题 ---" 形式的分节注释
SECTION_RE = re.compile(r"^#\s*-{3,}\s*(.*?)\s*-{3,}\s*$")

# 模块头注释 "# Module: xxx"
MODULE_HEADER_RE = re.compile(r"^#\s*Module:\s*(.+?)\s*$")

# 弃用相关的关键字（装饰器名、警告文字等）
DEPRECATED_RE = re.compile(r"deprecat", re.IGNORECASE)


def _parse(text: str, filename: str) -> ast.Module:
    """解析源码，屏蔽被解析文件自身的 SyntaxWarning 等噪音。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(text, filename=filename)


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass
class Param:
    """函数的一个参数。"""

    name: str
    default: str | None = None  # 默认值的源码文本，None 表示无默认值
    annotation: str | None = None
    kind: str = "positional"  # positional / vararg / keyword_only / kwarg

    @property
    def required(self) -> bool:
        """是否必须传参（*args / **kwargs 不算）。"""
        return (
            self.default is None
            and self.kind in ("positional", "keyword_only")
        )


@dataclass
class Function:
    """一个函数（或方法）。"""

    name: str
    module: str
    signature: str
    params: list[Param] = field(default_factory=list)
    description: str = ""
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    lineno: int = 0
    end_lineno: int = 0
    source: str = ""
    section: str = ""
    deprecated: bool = False
    deprecated_note: str = ""
    calls: list[str] = field(default_factory=list)  # 该函数调用到的包内函数
    returns: str | None = None
    is_async: bool = False
    exported: bool = True  # 是否出现在包命名空间中
    owner: str = ""  # 所属类（方法才有）

    @property
    def anchor(self) -> str:
        return f"{self.module}.{self.name}"

    @property
    def required_params(self) -> list[Param]:
        return [p for p in self.params if p.required]


@dataclass
class Class:
    """一个类。"""

    name: str
    module: str
    description: str = ""
    docstring: str = ""
    bases: list[str] = field(default_factory=list)
    methods: list[Function] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    lineno: int = 0
    exported: bool = True


@dataclass
class DataTable:
    """源码中以字面量形式定义的数据表。"""

    name: str
    module: str
    kind: str  # dict / list
    description: str = ""
    entries: list[tuple[str, str]] = field(default_factory=list)  # (键, 值的源码文本)
    lineno: int = 0
    origin: str = "function"  # function（函数返回）/ module（模块级常量）


@dataclass
class Module:
    """一个 Python 模块。"""

    name: str
    file: str
    path: str = ""  # 相对包根目录的路径
    source_path: str = ""  # 磁盘上的绝对路径
    title: str = ""
    description: str = ""
    functions: list[Function] = field(default_factory=list)
    classes: list[Class] = field(default_factory=list)
    data: list[DataTable] = field(default_factory=list)
    line_count: int = 0
    is_package: bool = False  # 该文件所在目录是一个包（有 __init__.py）
    is_init: bool = False  # 该文件本身是不是 __init__.py
    is_entry: bool = False  # 是否是包入口 __init__.py
    error: str = ""  # 解析失败时的错误信息

    @property
    def anchor(self) -> str:
        return self.name


@dataclass
class PackageInfo:
    """包的基本信息，来自 pyproject.toml / setup.cfg / __init__.py。"""

    name: str = ""
    import_name: str = ""  # 导入时使用的名字（包目录名）
    version: str = ""
    description: str = ""
    summary: str = ""
    requires_python: str = ""
    url: str = ""
    urls: dict[str, str] = field(default_factory=dict)
    author: str = ""
    author_email: str = ""
    license: str = ""
    classifiers: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# 文本小工具
# --------------------------------------------------------------------------


def _read_text(path: Path) -> tuple[str, str]:
    """读源码文本，返回 (文本, 编码)。对编码问题尽量宽容。"""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"


def _first_paragraph(text: str) -> str:
    """取文本的第一段，压成一行。"""
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    return " ".join(line.strip() for line in parts[0].splitlines() if line.strip())


def _is_dunder(name: str) -> bool:
    """是否是 __xxx__ 形式的内部名字。"""
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _definition_start(node: ast.AST) -> int:
    """定义真正开始的行号：装饰器之上的那一行。

    带装饰器时 ast 的 lineno 指向 def 那一行，
    上方的注释块会被装饰器隔开，取说明文字时要以装饰器为起点。
    """
    start = getattr(node, "lineno", 0)
    for decorator in getattr(node, "decorator_list", []) or []:
        start = min(start, decorator.lineno)
    return start


def _comment_block_above(lines: list[str], lineno: int) -> str:
    """取函数/类定义正上方紧邻的注释块（自下而上收集连续的 # 注释行）。"""
    collected: list[str] = []
    index = lineno - 2  # lineno 为 1-based，转换成上一行的 0-based 下标
    while index >= 0:
        stripped = lines[index].strip()
        if stripped.startswith("#"):
            collected.append(stripped.lstrip("#").strip())
            index -= 1
        else:
            break
    collected.reverse()
    return " ".join(part for part in collected if part)


def _docstring_of(node: ast.AST) -> str:
    """取节点 docstring（没有则为空串）。"""
    try:
        return ast.get_docstring(node) or ""
    except TypeError:
        return ""


# --------------------------------------------------------------------------
# 签名与参数
# --------------------------------------------------------------------------


def _format_params(args: ast.arguments) -> list[Param]:
    """把 ast.arguments 转成参数列表。"""
    params: list[Param] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        params.append(
            Param(
                name=arg.arg,
                default=ast.unparse(default) if default is not None else None,
                annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                kind="positional",
            )
        )
    if args.vararg:
        params.append(
            Param(
                name="*" + args.vararg.arg,
                annotation=ast.unparse(args.vararg.annotation)
                if args.vararg.annotation
                else None,
                kind="vararg",
            )
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            Param(
                name=arg.arg,
                default=ast.unparse(default) if default is not None else None,
                annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                kind="keyword_only",
            )
        )
    if args.kwarg:
        params.append(
            Param(
                name="**" + args.kwarg.arg,
                annotation=ast.unparse(args.kwarg.annotation)
                if args.kwarg.annotation
                else None,
                kind="kwarg",
            )
        )
    return params


def _format_signature(
    name: str, args: ast.arguments, returns: ast.expr | None = None
) -> str:
    """还原出可读的函数签名文本。

    优先用 ast.unparse 直接还原参数表（能正确处理仅位置参数分隔符 "/"、
    关键字限定、注解与默认值），失败时退回手工拼接。
    """
    try:
        text = ast.unparse(args)
    except Exception:  # pragma: no cover - ast.unparse 极少失败
        text = _format_params_text(args)
    signature = f"{name}({text})"
    if returns is not None:
        try:
            signature += " -> " + ast.unparse(returns)
        except Exception:  # pragma: no cover
            pass
    return signature


def _format_params_text(args: ast.arguments) -> str:
    """手工拼接参数列表（ast.unparse 不可用时的退路）。"""
    parts: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += ": " + ast.unparse(arg.annotation)
        if default is not None:
            text += "=" + ast.unparse(default)
        parts.append(text)
    if args.vararg:
        text = "*" + args.vararg.arg
        if args.vararg.annotation is not None:
            text += ": " + ast.unparse(args.vararg.annotation)
        parts.append(text)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += ": " + ast.unparse(arg.annotation)
        if default is not None:
            text += "=" + ast.unparse(default)
        parts.append(text)
    if args.kwarg:
        text = "**" + args.kwarg.arg
        if args.kwarg.annotation is not None:
            text += ": " + ast.unparse(args.kwarg.annotation)
        parts.append(text)
    return ", ".join(parts)


# --------------------------------------------------------------------------
# 弃用、调用、数据表
# --------------------------------------------------------------------------


def _deprecation(node: ast.AST) -> tuple[bool, str]:
    """判断是否已弃用，返回 (是否弃用, 说明文字)。

    识别几种写法：
      * 装饰器名里含 deprecated，例如 @deprecated
      * 函数体里的字符串含 deprecated，例如提示信息
      * 出现 DeprecationWarning / PendingDeprecationWarning，
        无论是直接写名字还是写成 warnings.DeprecationWarning
    """
    note = ""
    for decorator in getattr(node, "decorator_list", []) or []:
        text = ast.unparse(decorator)
        if DEPRECATED_RE.search(text):
            return True, text
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if DEPRECATED_RE.search(child.value):
                note = child.value.strip()
                break
        elif isinstance(child, ast.Name) and DEPRECATED_RE.search(child.id):
            if "Warning" in child.id:
                return True, note or child.id
        elif isinstance(child, ast.Attribute) and DEPRECATED_RE.search(child.attr):
            if "Warning" in child.attr:
                return True, note or child.attr
    return False, note


def _package_aliases(tree: ast.Module, names: set[str]) -> set[str]:
    """找出源码里指向本包的别名，例如 import guan / import guan as g。"""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in names:
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in names:
                for alias in node.names:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _iter_calls(node: ast.AST):
    """遍历节点里的调用，但不下钻到嵌套的函数 / 类 / lambda 内部。

    嵌套函数体内的调用属于嵌套函数，不该算到外层函数头上。
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _iter_calls(child)


def _package_calls(
    node: ast.AST,
    aliases: set[str],
    direct_names: set[str] | None = None,
    shadowed: set[str] | None = None,
) -> list[str]:
    """收集函数体内出现的包内函数调用（去重且保持顺序）。

    两种形式都算：
      * 别名调用：``guan.foo()`` / ``core.foo()``，别名须在 aliases 里
      * 直接调用：``from .core import foo`` 之后的 ``foo()``，
        名字须在 direct_names 里，且没有被本模块的同名定义遮蔽
    """
    found: list[str] = []
    direct_names = direct_names or set()
    shadowed = shadowed or set()
    for child in _iter_calls(node):
        func = child.func
        name = ""
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in aliases
        ):
            name = func.attr
        elif (
            isinstance(func, ast.Name)
            and func.id in direct_names
            and func.id not in shadowed
        ):
            name = func.id
        if name and name not in found:
            found.append(name)
    return found


def _literal_value(node: ast.AST) -> Any:
    """尝试把 AST 节点求值成 Python 字面量，失败则返回 None。"""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _format_value(value: Any) -> str:
    """把字面量转成适合网页展示的文本。"""
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return repr(value)
    return str(value)


def _table_entries(value: Any) -> list[tuple[str, str]]:
    """把 dict / list 字面量转成 (键, 值) 列表。"""
    if isinstance(value, dict):
        return [(str(k), _format_value(v)) for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [(str(i), _format_value(v)) for i, v in enumerate(value)]
    return []


def _collect_data_tables(node: ast.FunctionDef, module_name: str) -> list[DataTable]:
    """从一个函数里提取以字面量形式写死的数据表。

    覆盖常见的几种写法：
      * 直接 return 一个 dict / list 字面量
      * 先建一个容器变量，再逐项 constants['x'] = ... 赋值，最后 return 该变量
      * 用 container.update({...}) 追加
      * return 一个元组，其中的元素是 dict / list
    """
    tables: list[DataTable] = []
    variables: dict[str, Any] = {}

    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = _literal_value(statement.value)
            if isinstance(target, ast.Name) and value is not None:
                variables[target.id] = value
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                container = variables.get(target.value.id)
                key = _literal_value(target.slice)
                item = _literal_value(statement.value)
                if isinstance(container, dict) and key is not None and item is not None:
                    container[key] = item
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            container = variables.get(statement.target.id)
            item = _literal_value(statement.value)
            if isinstance(container, list) and isinstance(item, list):
                container.extend(item)
                variables[statement.target.id] = container
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.args
            ):
                container = variables.get(call.func.value.id)
                patch = _literal_value(call.args[0])
                if call.func.attr == "update" and isinstance(container, dict):
                    if isinstance(patch, dict):
                        container.update(patch)
                elif call.func.attr == "append" and isinstance(container, list):
                    if patch is not None:
                        container.append(patch)
                elif call.func.attr == "extend" and isinstance(container, list):
                    if isinstance(patch, (list, tuple)):
                        container.extend(patch)
        elif isinstance(statement, ast.Return) and statement.value is not None:
            value = _literal_value(statement.value)
            if value is None and isinstance(statement.value, ast.Name):
                value = variables.get(statement.value.id)
            candidates: list[Any] = []
            if isinstance(value, (list, tuple)):
                candidates = [v for v in value if isinstance(v, (dict, list, tuple))]
            elif value is not None:
                candidates = [value]
            for candidate in candidates:
                if not isinstance(candidate, (dict, list, tuple)) or not candidate:
                    continue
                kind = "dict" if isinstance(candidate, dict) else "list"
                entries = _table_entries(candidate)
                if entries:
                    tables.append(
                        DataTable(
                            name=node.name,
                            module=module_name,
                            kind=kind,
                            entries=entries,
                            lineno=node.lineno,
                            origin="function",
                        )
                    )

    return tables


def _collect_module_tables(tree: ast.Module, module_name: str) -> list[DataTable]:
    """提取模块级以字面量定义的数据表（常量表）。"""
    tables: list[DataTable] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target = statement.target
            value_node = statement.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if _is_dunder(target.id):
            continue  # __all__ / __version__ 之类的内部名字不算数据表
        value = _literal_value(value_node)
        if not isinstance(value, (dict, list)) or not value:
            continue
        kind = "dict" if isinstance(value, dict) else "list"
        tables.append(
            DataTable(
                name=target.id,
                module=module_name,
                kind=kind,
                entries=_table_entries(value),
                lineno=statement.lineno,
                origin="module",
            )
        )
    return tables


# --------------------------------------------------------------------------
# 模块解析
# --------------------------------------------------------------------------


def _find_sections(lines: list[str]) -> list[tuple[int, str]]:
    """找出所有分节注释，返回 (行号, 标题)。"""
    sections: list[tuple[int, str]] = []
    for number, line in enumerate(lines, start=1):
        match = SECTION_RE.match(line.strip())
        if match:
            sections.append((number, match.group(1)))
    return sections


def _module_all(tree: ast.Module) -> set[str] | None:
    """取模块级的 __all__（没有则返回 None）。"""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = _literal_value(node.value)
                    if isinstance(value, (list, tuple, set)):
                        return {str(item) for item in value}
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                value = _literal_value(node.value) if node.value else None
                if isinstance(value, (list, tuple, set)):
                    return {str(item) for item in value}
    return None


def parse_module(path: Path, module_name: str, relative_path: str = "") -> Module:
    """解析单个模块文件。

    解析失败不会抛出异常，而是在返回的 Module 上记录 error，
    以免一个文件的语法问题导致整个网站构建失败。
    """
    module = Module(
        name=module_name,
        file=path.name,
        path=relative_path or path.name,
        source_path=str(path),
    )
    try:
        text, _ = _read_text(path)
    except OSError as exc:
        module.error = f"无法读取文件：{exc}"
        return module

    module.line_count = len(text.splitlines())
    try:
        tree = _parse(text, str(path))
    except SyntaxError as exc:
        module.error = f"语法错误：第 {exc.lineno} 行 {exc.msg}"
        return module

    lines = text.splitlines()

    for line in lines:
        header = MODULE_HEADER_RE.match(line.strip())
        if header:
            module.title = header.group(1)
            break

    module.description = _first_paragraph(_docstring_of(tree))
    if not module.title:
        module.title = module_name

    sections = _find_sections(lines)
    aliases = _package_aliases(tree, {module_name.split(".")[0]}) | {module_name}

    def section_for(lineno: int) -> str:
        current = ""
        for number, title in sections:
            if number < lineno:
                current = title
            else:
                break
        return current

    def describe(node: ast.AST) -> str:
        """函数说明：注释块优先，其次 docstring。

        注释块从装饰器往上找，这样「@装饰器」不会把说明文字挡掉。
        """
        comment = _comment_block_above(lines, _definition_start(node))
        if comment:
            return comment
        return _first_paragraph(_docstring_of(node))

    def build_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str = ""
    ) -> Function:
        end = node.end_lineno or node.lineno
        # 装饰器在 def 之上，源码片段要从装饰器开始
        start = _definition_start(node)
        source = "\n".join(lines[start - 1 : end])
        deprecated, note = _deprecation(node)
        docstring = _docstring_of(node)
        return Function(
            name=node.name,
            module=module_name,
            signature=_format_signature(node.name, node.args, node.returns),
            params=_format_params(node.args),
            description=describe(node),
            docstring=docstring,
            decorators=[ast.unparse(d) for d in node.decorator_list],
            lineno=node.lineno,
            end_lineno=end,
            source=source,
            section=section_for(node.lineno),
            deprecated=deprecated,
            deprecated_note=note,
            calls=_package_calls(node, aliases),
            returns=ast.unparse(node.returns) if node.returns else None,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            owner=owner,
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module.functions.append(build_function(node))
            module.data.extend(_collect_data_tables(node, module_name))
        elif isinstance(node, ast.ClassDef):
            klass = Class(
                name=node.name,
                module=module_name,
                description=describe(node),
                docstring=_docstring_of(node),
                bases=[ast.unparse(base) for base in node.bases],
                lineno=node.lineno,
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    klass.methods.append(build_function(item, owner=node.name))
                elif isinstance(item, ast.Assign) and len(item.targets) == 1:
                    if isinstance(item.targets[0], ast.Name):
                        klass.attributes.append(item.targets[0].id)
            module.classes.append(klass)

    module.data.extend(_collect_module_tables(tree, module_name))

    # 让数据表带上说明文字
    for table in module.data:
        if table.origin == "function":
            for function in module.functions:
                if function.name == table.name:
                    table.description = function.description
                    break

    return module


# --------------------------------------------------------------------------
# 包解析
# --------------------------------------------------------------------------


def module_name_for(path: Path, package_root: Path) -> str:
    """由文件路径推出模块名（相对包根目录，支持子包）。"""
    relative = path.relative_to(package_root)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _is_build_dir(path: Path) -> bool:
    """判断目录是不是网站构建脚本所在目录（那不是被发布的包）。"""
    return (path / "build_site.py").is_file() or (path / "extract.py").is_file()


def discover_package_dir(search_roots: list[Path]) -> Path | None:
    """在候选目录里找出包目录（含 __init__.py 的那一层）。

    依次检查每个候选根目录下的子目录；若根目录本身就是包，也算命中。
    这样仓库把源码放在 PyPI/src、src 或根目录都能被认出来。
    """
    for root in search_roots:
        root = Path(root)
        if not root.is_dir() or _is_build_dir(root):
            continue
        if (root / "__init__.py").is_file():
            return root
        candidates = [
            child
            for child in sorted(root.iterdir())
            if child.is_dir()
            and not child.name.startswith((".", "_"))
            and not _is_build_dir(child)
            and (child / "__init__.py").is_file()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            # 多个候选时，优先挑模块文件最多的那个
            return max(candidates, key=lambda d: len(list(d.rglob("*.py"))))
    # 退一步：支持没有 __init__.py 的命名空间包（PEP 420）
    for root in search_roots:
        root = Path(root)
        if not root.is_dir() or _is_build_dir(root):
            continue
        loose = [
            child
            for child in sorted(root.iterdir())
            if child.is_dir()
            and not child.name.startswith((".", "_"))
            and not _is_build_dir(child)
            and any(child.glob("*.py"))
        ]
        if loose:
            return max(loose, key=lambda d: len(list(d.rglob("*.py"))))
    return None


def _import_order(entry_module: Module) -> list[str]:
    """从包入口 __init__.py 解析导入顺序（from .xxx import * 等）。"""
    order: list[str] = []
    path = Path(entry_module.source_path or entry_module.path)
    try:
        text, _ = _read_text(path)
        tree = _parse(text, str(path))
    except (OSError, SyntaxError):
        return order
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            order.append(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level and node.names:
            for alias in node.names:
                order.append(alias.name)
    return order


_TREE_CACHE: dict[str, ast.Module | None] = {}


def _tree_of(module: Module) -> ast.Module | None:
    """取模块的 AST（带缓存，解析失败返回 None）。"""
    path = Path(module.source_path or module.path)
    key = str(path)
    if key in _TREE_CACHE:
        return _TREE_CACHE[key]
    tree: ast.Module | None = None
    try:
        text, _ = _read_text(path)
        tree = _parse(text, str(path))
    except (OSError, SyntaxError):
        tree = None
    _TREE_CACHE[key] = tree
    return tree


def _package_context(module: Module) -> list[str]:
    """模块在包内的相对位置（用于解析 from .xxx import）。"""
    if module.is_entry:
        return []
    if module.is_init:
        return module.name.split(".")
    parts = module.name.split(".")
    return parts[:-1]


def _resolve_relative(module: Module, level: int, name: str) -> str:
    """把 `from .xxx import y` 里的 xxx 解析成包内的完整模块名。"""
    context = _package_context(module)
    if level > 1:
        context = context[: max(0, len(context) - (level - 1))]
    if name:
        context = context + name.split(".")
    return ".".join(context)


def _public_names(
    module_name: str,
    by_name: dict[str, Module],
    stack: tuple[str, ...] = (),
) -> set[str]:
    """一个模块通过 `from .x import *` 会带出哪些名字。

    除了本模块定义的函数与类，还包括它自己再导入的名字
    （比如子包的 __init__.py 里 `from .extra import *`），
    这样包入口 `from .sub import *` 也能把子包里的函数带出来。
    """
    module = by_name.get(module_name)
    if module is None or module_name in stack:
        return set()
    tree = _tree_of(module)
    if tree is None:
        return set()

    explicit = _module_all(tree)
    if explicit is not None:
        return set(explicit)

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.module:
                    target = _resolve_relative(module, node.level, node.module)
                    names |= _public_names(target, by_name, stack + (module_name,))
                else:
                    # from . import mod：带出子模块本身
                    for alias in node.names:
                        if alias.name != "*":
                            names.add(alias.asname or alias.name)
            elif node.module:
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _module_constants(module: Module) -> set[str]:
    """模块顶层定义的常量名（大写或普通赋值）。"""
    tree = _tree_of(module)
    if tree is None:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _definition_index(modules: list[Module]) -> dict[str, str]:
    """名字 → 真正定义它的模块。

    同名时以先出现的模块为准；跨子包转发时也能指对真正定义的地方。
    """
    index: dict[str, str] = {}
    for module in modules:
        for function in module.functions:
            index.setdefault(function.name, module.name)
        for klass in module.classes:
            index.setdefault(klass.name, module.name)
    return index


def compute_exports(
    package_root: Path, modules: list[Module]
) -> tuple[dict[str, str], set[str]]:
    """计算包命名空间里可见的名字。

    返回 (名字 → 定义所在模块, 包级显式 __all__)。
    解析 __init__.py 里的几种导入写法：
      * from .mod import *        → 整个模块的公开名字
      * from .mod import name     → 单个名字
      * from .mod import name as x→ 别名
      * from . import mod         → 子模块本身

    只有包内真正定义出来的名字才算导出，第三方库（numpy、os 之类）
    即使在 __init__.py 里被 import，也不会出现在导出表里。
    """
    exports: dict[str, str] = {}
    explicit_all: set[str] = set()

    by_name = {module.name: module for module in modules}
    entry = next((m for m in modules if m.is_entry), None)
    if entry is None:
        return exports, explicit_all

    tree = _tree_of(entry)
    if tree is None:
        return exports, explicit_all

    # 包里真正定义出来的名字（排除 import 进来的第三方名字，如 numpy）
    owners = _definition_index(modules)
    defined_names: set[str] = set(owners)
    for module in modules:
        defined_names.update(_module_constants(module))

    def owner_of(name: str) -> str:
        return owners.get(name, entry.name)

    def is_public(name: str) -> bool:
        """下划线开头的名字不算公开 API。"""
        return bool(name) and not name.startswith("_")

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = _literal_value(node.value)
                    if isinstance(value, (list, tuple, set)):
                        explicit_all = {str(item) for item in value}
        elif isinstance(node, ast.ImportFrom):
            source = node.module or ""
            if node.level and not source:
                # from . import mod：把子模块本身挂到包命名空间上
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    target = _resolve_relative(entry, node.level, alias.name)
                    if target in by_name:
                        exports.setdefault(alias.asname or alias.name, target)
            elif node.level and source:
                source = _resolve_relative(entry, node.level, source)
                if source not in by_name:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        for name in _public_names(source, by_name):
                            if name in defined_names and is_public(name):
                                exports.setdefault(name, owner_of(name))
                    elif is_public(alias.asname or alias.name):
                        exports.setdefault(
                            alias.asname or alias.name, owner_of(alias.name)
                        )
            elif source:
                # from 包名 import xxx：只认包内真正定义过的名字
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if alias.name in defined_names and is_public(alias.asname or alias.name):
                        exports.setdefault(
                            alias.asname or alias.name, owner_of(alias.name)
                        )
        elif isinstance(node, ast.Import):
            # 直接 import 进来的多半是第三方库，只有包内定义的名字才算导出
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if alias.name.count(".") == 0 and name in defined_names and is_public(name):
                    exports.setdefault(name, owner_of(name))

    # 入口自身定义的函数、类与常量总是可见
    for function in entry.functions:
        if is_public(function.name):
            exports.setdefault(function.name, entry.name)
    for klass in entry.classes:
        if is_public(klass.name):
            exports.setdefault(klass.name, entry.name)
    for name in _module_constants(entry):
        if is_public(name):
            exports.setdefault(name, entry.name)

    # 注意：__all__ 只影响 `from 包 import *`，
    # 不影响 `包.名字` 的访问，所以这里不据此裁剪导出表。
    return exports, explicit_all


def parse_modules(source_dir: Path) -> list[Module]:
    """解析包目录下的所有模块（只做语法解析，不涉及导出关系）。

    递归处理子包；包入口 __init__.py 也作为一个模块返回（is_entry=True）。
    顺序按 __init__.py 中的导入顺序排列，其余模块按名字追加。
    """
    package_root = Path(source_dir)
    modules: list[Module] = []

    files = [
        p
        for p in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in p.parts and p.name != "setup.py"
    ]
    for path in files:
        relative = path.relative_to(package_root)
        name = module_name_for(path, package_root)
        if not name:
            # 包入口 __init__.py：模块名就是包名
            name = package_root.name
        module = parse_module(path, name, relative.as_posix())
        module.is_init = path.name == "__init__.py"
        module.is_entry = module.is_init and path.parent == package_root
        module.is_package = (path.parent / "__init__.py").is_file()
        if module.is_entry:
            module.title = module.title or package_root.name
        modules.append(module)

    # 排序：按 __init__.py 的导入顺序
    entry = next((m for m in modules if m.is_entry), None)
    order: list[str] = []
    if entry is not None:
        order = _import_order(entry)
    rank = {name: index for index, name in enumerate(order)}

    def sort_key(module: Module) -> tuple:
        if module.is_entry:
            return (1, 0, module.name)
        return (0, rank.get(module.name, len(rank)), module.name)

    modules.sort(key=sort_key)
    return modules


def apply_exports(modules: list[Module], exports: dict[str, str]) -> None:
    """按导出表标记每个函数 / 类是否出现在包命名空间里。"""
    for module in modules:
        for function in module.functions:
            function.exported = function.name in exports
        for klass in module.classes:
            klass.exported = klass.name in exports


def _direct_package_names(module: Module, by_name: dict[str, Module]) -> set[str]:
    """本模块用 `from .xxx import y` 直接引进来的包内名字。

    这些名字在函数体里是以裸名字调用的（``y()``），
    只靠「包名.xxx」的形式是找不到它们的。
    """
    tree = _tree_of(module)
    if tree is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            if name in by_name:
                continue  # 这是子模块本身，不是函数
            names.add(name)
    return names


def resolve_calls(
    modules: list[Module],
    package_root: Path,
    package_aliases: set[str] | None = None,
) -> None:
    """算出每个函数调用到的包内函数（用于生成交叉参考）。"""
    package_root = Path(package_root)
    by_name = {module.name: module for module in modules}

    # 包内的调用（交叉参考）应使用整个包的别名集合：
    # 包名本身、模块名、以及源码里显式写下的别名。
    all_aliases: set[str] = {package_root.name}
    all_aliases |= {m.name for m in modules}
    all_aliases |= {m.name.split(".")[0] for m in modules}
    if package_aliases:
        all_aliases |= package_aliases
    for module in modules:
        all_aliases |= _explicit_aliases(module)

    for module in modules:
        tree = _tree_of(module)
        if tree is None:
            continue
        direct = _direct_package_names(module, by_name)
        # 本模块自己定义的名字会遮住直接引入的同名函数
        shadowed = {f.name for f in module.functions}
        shadowed |= {c.name for c in module.classes}
        calls_by_function: dict[tuple[str, int], list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls_by_function[(node.name, node.lineno)] = _package_calls(
                    node, all_aliases, direct, shadowed
                )
        for function in module.functions:
            function.calls = calls_by_function.get(
                (function.name, function.lineno), function.calls
            )


def parse_package(
    source_dir: Path, package_aliases: set[str] | None = None
) -> list[Module]:
    """解析整个包目录，返回模块列表。

    解析、导出关系、交叉参考三步都做完了。
    """
    package_root = Path(source_dir)
    modules = parse_modules(package_root)
    exports, _ = compute_exports(package_root, modules)
    apply_exports(modules, exports)
    resolve_calls(modules, package_root, package_aliases)
    return modules


def _explicit_aliases(module: Module) -> set[str]:
    """源码里指向包内模块的别名。

    只认相对导入（``from .core import foo as f``、``from . import core as c``），
    第三方库的别名（``import numpy as np``）不算，
    否则 ``np.array(...)`` 会被误认成对包内 ``array`` 的调用。
    """
    tree = _tree_of(module)
    if tree is None:
        return set()
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            for alias in node.names:
                if alias.asname:
                    aliases.add(alias.asname)
    return aliases


# --------------------------------------------------------------------------
# 包元信息
# --------------------------------------------------------------------------


def _read_pyproject(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text, _ = _read_text(path)
        return tomllib.loads(text)
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return {}


def _read_setup_cfg(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    parser = configparser.ConfigParser()
    try:
        text, _ = _read_text(path)
        parser.read_string(text)
    except (OSError, ValueError, configparser.Error):
        return {}
    return {section: dict(parser[section]) for section in parser.sections()}


def _version_from_init(path: Path) -> str:
    """从 __init__.py 里找 __version__ = "..."。"""
    if not path.exists():
        return ""
    try:
        text, _ = _read_text(path)
        tree = _parse(text, str(path))
    except (OSError, SyntaxError):
        return ""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    value = _literal_value(node.value)
                    if isinstance(value, str):
                        return value
    return ""


def _module_attr_value(path: Path, attr: str) -> str:
    """在模块里找某个赋值（``attr = "..."``）的字面量值。"""
    try:
        text, _ = _read_text(path)
        tree = _parse(text, str(path))
    except (OSError, SyntaxError):
        return ""
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == attr:
                    value = _literal_value(node.value)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == attr
                and node.value is not None
            ):
                value = _literal_value(node.value)
        if isinstance(value, str):
            return value
    return ""


def _attr_value(package_dir: Path | None, spec: str) -> str:
    """解析 ``attr: 包名.__version__``，静态读出那个值。

    不 import 被解析的包，只按路径找文件读字面量。
    """
    if not package_dir:
        return ""
    spec = spec.strip()
    if "." not in spec:
        return ""
    module_part, _, attr = spec.rpartition(".")
    parts = module_part.split(".")
    candidates = []
    if module_part == package_dir.name:
        # 指的就是包自己
        candidates.append(package_dir / "__init__.py")
    candidates.append(package_dir.joinpath(*parts).with_suffix(".py"))
    candidates.append(package_dir.joinpath(*parts, "__init__.py"))
    for path in candidates:
        if path.is_file():
            value = _module_attr_value(path, attr)
            if value:
                return value
    return ""


def _resolve_version_directive(
    value: str, package_dir: Path | None, base_dir: Path | None
) -> str:
    """把 ``attr:`` / ``file:`` 形式的版本号写法解析成真正的值。

    setup.cfg 里常写 ``version = attr: pkg.__version__``，
    直接显示出来会很难看，这里静态地把它读出来。
    """
    value = value.strip()
    if value.startswith("attr:"):
        return _attr_value(package_dir, value[5:]) or value
    if value.startswith("file:"):
        name = value[5:].strip()
        if base_dir:
            path = Path(base_dir) / name
            if path.is_file():
                try:
                    text, _ = _read_text(path)
                except OSError:
                    return value
                first = text.strip().splitlines()[0].strip() if text.strip() else ""
                if first:
                    return first
    return value


def parse_package_info(
    pyproject: Path,
    setup_cfg: Path | None = None,
    init_file: Path | None = None,
    package_dir: Path | None = None,
    import_name: str = "",
) -> PackageInfo:
    """读取包信息。

    同时兼容三种来源：PEP 621 的 [project] 表、setuptools 的 [metadata] 表、
    以及 setup.cfg。本仓库的元数据写在 setup.cfg 的 [metadata] 段里。
    """
    sources: list[dict[str, Any]] = []

    data = _read_pyproject(Path(pyproject))
    project = data.get("project")
    if isinstance(project, dict):
        sources.append(project)
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        sources.append(metadata)

    cfg = _read_setup_cfg(Path(setup_cfg)) if setup_cfg else {}
    for section in ("metadata", "options"):
        if section in cfg:
            sources.append(cfg[section])

    def pick(*keys: str, default: str = "") -> str:
        for source in sources:
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
        return default

    urls: dict[str, str] = {}
    for source in sources:
        raw = source.get("project_urls") or source.get("urls")
        if not raw:
            continue
        if isinstance(raw, dict):  # pyproject 的 [project.urls] 表
            for key, value in raw.items():
                if isinstance(value, str):
                    urls.setdefault(str(key), value)
            continue
        raw = str(raw)
        if raw.strip().startswith("{"):
            try:
                urls = {str(k): str(v) for k, v in json.loads(raw).items()}
                break
            except (ValueError, TypeError):
                continue
        for line in raw.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                urls.setdefault(key.strip(), value.strip())

    classifiers: list[str] = []
    for source in sources:
        value = source.get("classifiers")
        if isinstance(value, list):
            classifiers = [str(item) for item in value]
            break
        if isinstance(value, str) and value.strip():
            classifiers = [line.strip() for line in value.splitlines() if line.strip()]
            break

    author = pick("author", "maintainer")
    author_email = pick("author_email", "maintainer_email")
    if not author or not author_email:
        # PEP 621 的 authors / maintainers 是 [{name, email}, ...] 列表
        for source in sources:
            for key in ("authors", "maintainers"):
                people = source.get(key)
                if not isinstance(people, list):
                    continue
                for person in people:
                    if not isinstance(person, dict):
                        continue
                    if not author and person.get("name"):
                        author = str(person["name"])
                    if not author_email and person.get("email"):
                        author_email = str(person["email"])

    license_text = pick("license")
    if not license_text:
        for source in sources:
            value = source.get("license")
            if isinstance(value, dict):
                license_text = str(value.get("text") or value.get("file") or "")
                if license_text:
                    break

    version = pick("version")
    if version:
        base_dir = Path(setup_cfg).parent if setup_cfg else None
        version = _resolve_version_directive(version, package_dir, base_dir)
    if not version and init_file is not None:
        version = _version_from_init(Path(init_file))

    name = pick("name")
    if not name and package_dir is not None:
        name = Path(package_dir).name
    if not import_name and package_dir is not None:
        import_name = Path(package_dir).name

    description = pick("description", "summary")
    if description.startswith("file:"):
        # 描述指向一个文件时（比如 `description = file: README.md`），
        # 取那个文件的第一段文字，而不是把 "file: ..." 摆在网站上
        base_dir = Path(setup_cfg).parent if setup_cfg else None
        if base_dir:
            path = base_dir / description[5:].strip()
            if path.is_file():
                try:
                    text, _ = _read_text(path)
                except OSError:
                    text = ""
                description = _first_paragraph(text) or description

    return PackageInfo(
        name=name,
        import_name=import_name or name,
        version=version,
        description=description,
        summary=pick("summary"),
        requires_python=pick("requires-python", "requires_python", "python_requires"),
        url=pick("url", "home-page", "homepage"),
        urls=urls,
        author=author,
        author_email=author_email,
        license=license_text,
        classifiers=classifiers,
    )


# --------------------------------------------------------------------------
# 序列化
# --------------------------------------------------------------------------


def _function_to_dict(function: Function) -> dict[str, Any]:
    """把函数（或方法）序列化成统一的字典结构。"""
    return {
        "name": function.name,
        "signature": function.signature,
        "description": function.description,
        "docstring": function.docstring,
        "decorators": function.decorators,
        "lineno": function.lineno,
        "end_lineno": function.end_lineno,
        "section": function.section,
        "deprecated": function.deprecated,
        "deprecated_note": function.deprecated_note,
        "calls": function.calls,
        "returns": function.returns,
        "is_async": function.is_async,
        "exported": function.exported,
        "owner": function.owner,
        "params": [
            {
                "name": p.name,
                "default": p.default,
                "annotation": p.annotation,
                "kind": p.kind,
                "required": p.required,
            }
            for p in function.params
        ],
        "source": function.source,
    }


def to_dict(
    modules: list[Module],
    info: PackageInfo,
    exports: dict[str, str] | None = None,
) -> dict[str, Any]:
    """整理成可直接序列化为 JSON 的结构。"""
    exported_names = set(exports or {})
    return {
        "package": {
            "name": info.name,
            "import_name": info.import_name or info.name,
            "version": info.version,
            "description": info.description,
            "summary": info.summary,
            "requires_python": info.requires_python,
            "url": info.url,
            "urls": info.urls,
            "author": info.author,
            "author_email": info.author_email,
            "license": info.license,
            "classifiers": info.classifiers,
        },
        "stats": {
            "modules": len(modules),
            "functions": sum(len(m.functions) for m in modules),
            "classes": sum(len(m.classes) for m in modules),
            "lines": sum(m.line_count for m in modules),
            "data_tables": sum(len(m.data) for m in modules),
            "exported_functions": sum(
                1
                for m in modules
                for f in m.functions
                if f.exported or f.name in exported_names
            ),
        },
        "exports": dict(sorted((exports or {}).items())),
        "modules": [
            {
                "name": m.name,
                "file": m.file,
                "path": m.path,
                "title": m.title,
                "description": m.description,
                "line_count": m.line_count,
                "is_package": m.is_package,
                "is_entry": m.is_entry,
                "error": m.error,
                "data": [
                    {
                        "name": d.name,
                        "kind": d.kind,
                        "description": d.description,
                        "lineno": d.lineno,
                        "origin": d.origin,
                        "entries": [{"key": k, "value": v} for k, v in d.entries],
                    }
                    for d in m.data
                ],
                "classes": [
                    {
                        "name": c.name,
                        "description": c.description,
                        "docstring": c.docstring,
                        "bases": c.bases,
                        "lineno": c.lineno,
                        "exported": c.exported,
                        "attributes": list(c.attributes),
                        "methods": [_function_to_dict(f) for f in c.methods],
                    }
                    for c in m.classes
                ],
                "functions": [_function_to_dict(f) for f in m.functions],
            }
            for m in modules
        ],
    }
