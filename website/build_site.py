"""生成软件包的网站。

用法：
    python website/build_site.py                     # 输出到 dist/site
    python website/build_site.py -o build/site      # 指定输出目录
    python website/build_site.py --source PyPI/src/guan

网站的内容有四个来源，都不是手工维护的：
  * 包源码（静态解析，见 extract.py）—— API 参考
  * 仓库根目录的 README.md（渲染，见 md_render.py）—— 关于、用法、引用
  * 仓库根目录的 INSTALLATION.md（渲染，见 md_render.py）—— 安装
  * 仓库根目录的 tutorials/（读取，见 tutorials.py）—— 教程与示例
因此不会与实际发布的包脱节。

构建过程不假设任何固定的包名、目录名或函数名：
  * 包目录自动发现（PyPI/src、src 等常见布局都能找到）
  * 页面与导航按实际有内容与否动态生成
  * 首页示例代码从源码里真实的函数挑出来，而不是写死

站点有中英文两套界面（见 i18n.py）：中文在根目录，英文在 en/ 子目录。
只有网站框架（导航、按钮、标题这些）会翻译，文件内容一律保持原样。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# 允许 `python -m website.build_site` 与 `python website/build_site.py` 两种调用方式
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract  # noqa: E402
import i18n  # noqa: E402
import md_render  # noqa: E402
import tutorials  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PYPI_DIR = ROOT / "PyPI"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# 源码可能放在这些地方，按顺序找
SOURCE_CANDIDATES = (
    PYPI_DIR / "src",
    ROOT / "src",
    ROOT,
)

# 教程目录（示例脚本）可能放在这些地方，按顺序找
TUTORIALS_CANDIDATES = (
    ROOT / "tutorials",
    PYPI_DIR / "tutorials",
)


def e(text: object) -> str:
    """HTML 转义。"""
    return html.escape(str(text), quote=True)


def display_name(package: dict) -> str:
    """站点上展示的名字（品牌名）。

    PyPI 上的包名与导入名必须是小写的（pip install guan、import guan），
    但网站标题、页头品牌这类地方用大写更醒目——所以展示名与包名分开。
    展示名仍由包名推导，不写死具体的名字。
    """
    name = package.get("display_name") or package["name"]
    if name.islower() and name.isalnum():
        return name.upper()
    return name


def load_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def render(template: str, **context: object) -> str:
    """极简模板替换：{{ key }} 形式。

    只扫描模板本身一次，替换进去的内容不会被二次替换——
    否则源码或 README 里出现 {{version}} 之类会被误改。
    """
    return re.sub(
        r"\{\{\s*(\w+)\s*\}\}",
        lambda match: str(context.get(match.group(1), match.group(0))),
        template,
    )


# --------------------------------------------------------------------------
# 页面定义
# --------------------------------------------------------------------------


@dataclass
class PageSpec:
    """一个页面的定义。

    页面是否出现在导航里，由 available 决定——这样源码里没有函数时，
    「API 参考」页会自动消失，而不是留下一个空页面。

    标题与导航文字都用 i18n.py 里的文案键，两种语言各写一份；
    title_fn 用于标题里要带包名这类动态内容的情况。
    """

    key: str
    filename: str
    label_key: str
    title_key: str
    builder: Callable[[dict], str]
    description_fn: Callable[[dict], str]
    title_fn: Callable[[dict], str] | None = None
    available: Callable[[dict], bool] | None = None

    def page_title(self, data: dict) -> str:
        if self.title_fn is not None:
            return self.title_fn(data)
        return data["t"](self.title_key)

    def description(self, data: dict) -> str:
        return self.description_fn(data)


def _has_functions(data: dict) -> bool:
    """包里有没有可展示的函数（类的方法也算）。"""
    if data["stats"]["functions"]:
        return True
    return any(module["classes"] for module in data["modules"])


def _has_installation(data: dict) -> bool:
    """有没有安装说明文档可渲染。"""
    return Path(data["meta"]["installation_file"]).exists()


def _has_tutorials(data: dict) -> bool:
    """tutorials/ 目录里有没有可展示的示例脚本。"""
    tutorials_data = data.get("tutorials") or {}
    return bool(tutorials_data.get("files"))


PAGE_SPECS: list[PageSpec] = [
    PageSpec(
        key="index",
        filename="index.html",
        label_key="nav_home",
        title_key="page_home",
        title_fn=lambda data: (
            f"{display_name(data['package'])} — "
            f"{data['package']['description'] or data['package'].get('summary') or ''}"
        ).strip(" —"),
        builder=lambda data: build_home(data),
        description_fn=lambda data: data["t"](
            "page_home", name=display_name(data["package"])
        ),
    ),
    PageSpec(
        key="installation",
        filename="installation.html",
        label_key="nav_installation",
        title_key="page_installation",
        builder=lambda data: build_installation(data),
        description_fn=lambda data: data["t"](
            "desc_installation", name=display_name(data["package"])
        ),
        available=_has_installation,
    ),
    PageSpec(
        key="api",
        filename="api.html",
        label_key="nav_api",
        title_key="page_api",
        builder=lambda data: build_api(data),
        description_fn=lambda data: data["t"](
            "desc_api", name=display_name(data["package"])
        ),
        available=_has_functions,
    ),
    PageSpec(
        key="tutorials",
        filename="tutorials.html",
        label_key="nav_tutorials",
        title_key="page_tutorials",
        builder=lambda data: build_tutorials(data),
        description_fn=lambda data: data["t"](
            "desc_tutorials", name=display_name(data["package"])
        ),
        available=_has_tutorials,
    ),
    PageSpec(
        key="about",
        filename="about.html",
        label_key="nav_about",
        title_key="page_about",
        builder=lambda data: build_readme(data),
        description_fn=lambda data: data["t"](
            "desc_about", name=display_name(data["package"])
        ),
    ),
]


def active_pages(data: dict) -> list[PageSpec]:
    """返回实际要生成的页面列表。"""
    pages = []
    for spec in PAGE_SPECS:
        if spec.available is not None and not spec.available(data):
            continue
        pages.append(spec)
    return pages


def lang_url(lang: str, filename: str, from_lang: str) -> str:
    """另一种语言里同一个页面的相对路径（相对于当前页面）。

    中文页面在根目录，英文页面在 en/ 子目录，所以两个方向要分别处理。
    """
    target_dir = i18n.LANG_DIRS[lang]
    current_dir = i18n.LANG_DIRS[from_lang]
    if target_dir == current_dir:
        return filename
    if not target_dir:
        return "../" + filename
    if not current_dir:
        return target_dir + "/" + filename
    return "../" + target_dir + "/" + filename


def lang_dir(output: Path, lang: str) -> Path:
    """某种语言的页面所在目录。"""
    directory = i18n.LANG_DIRS[lang]
    return output / directory if directory else output


def absolute_url(base: str, path: str) -> str:
    """把站点内的相对路径拼成绝对地址。

    base 是站点根地址（如 https://py.guanjihuan.com），path 是页面里的相对路径
    （如 "api.html" 或 "../api.html"）。搜索引擎要求 hreflang 与 canonical
    用绝对地址，所以这里把 ".." 之类的片段解析掉再拼。
    不知道根地址时原样返回，页面至少还能用相对地址互相跳转。
    """
    if not base:
        return path
    parts: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if parts:
                parts.pop()
            continue
        parts.append(segment)
    return base + "/" + "/".join(parts)


def site_base_url(package: dict, override: str | None = None) -> str:
    """站点根地址。

    优先用命令行给的 --base-url；没给就从包元数据里的主页地址推。
    末尾的斜杠去掉，免得拼出 "https://x.com//api.html" 这种双斜杠。
    """
    candidate = override or package.get("url") or ""
    return candidate.strip().rstrip("/")



def _lang_context(data: dict, lang: str, pages: list[PageSpec]) -> dict:
    """按语言准备一份数据：文案翻译器，加上该语言的导航标签。"""
    t = i18n.translator(lang)
    meta = dict(data["meta"])
    meta["page_links"] = [
        {"key": spec.key, "filename": spec.filename, "label": t(spec.label_key)}
        for spec in pages
        if spec.key != "index"
    ]
    return {**data, "meta": meta, "t": t, "lang": lang}


def nav_html(pages: list[PageSpec], active_key: str, t) -> str:
    """生成导航条（顶栏与页脚共用）。"""
    items = []
    for spec in pages:
        class_attr = "is-active" if spec.key == active_key else ""
        items.append(
            f'<a href="{spec.filename}" class="{class_attr}">{e(t(spec.label_key))}</a>'
        )
    return "\n      ".join(items)


def page(
    template: str,
    *,
    title: str,
    description: str,
    content: str,
    info: dict,
    stats: dict,
    pages: list[PageSpec],
    active: str,
    lang: str,
    filename: str,
    site_url: str = "",
    extra_js: str = "",
) -> str:
    """套上 base.html 外壳。"""
    package = info
    t = i18n.translator(lang)
    other = next(item for item in i18n.LANGS if item != lang)
    return render(
        template,
        # 标题与描述会进到 <title> 和 meta 里，必须转义
        # 首页标题已经以展示名开头，后面不再重复追加「· 展示名」
        title=e(
            title
            if title.startswith(display_name(package))
            else f"{title} · {display_name(package)}"
        ),
        description=e(description),
        content=content,
        version=package["version"],
        package_name=display_name(package),
        import_name=package.get("import_name") or package["name"],
        nav_items=nav_html(pages, active, t),
        foot_nav_items=nav_html(pages, active, t),
        module_count=stats["modules"],
        function_count=stats["functions"],
        class_count=stats["classes"],
        line_count=stats["lines"],
        lang=i18n.HTML_LANG[lang],
        lang_switch=lang_url(other, filename, lang),
        lang_switch_label=i18n.LANG_NAMES_FULL[other],
        lang_switch_title=t("switch_language"),
        # 无障碍名称里必须包含按钮上看得见的文字，否则读屏软件会读出
        # 一个跟界面上的标签对不上的名字
        lang_switch_aria=t("switch_to_lang", lang=i18n.LANG_NAMES_FULL[other]),
        lang_switch_hreflang=i18n.HTML_LANG[other],
        # hreflang 与 canonical 用绝对地址——搜索引擎只认绝对的
        alt_zh=absolute_url(site_url, lang_url("zh", filename, lang)),
        alt_en=absolute_url(site_url, lang_url("en", filename, lang)),
        # x-default 指向默认语言（中文）的那一份
        alt_default=absolute_url(site_url, lang_url(i18n.DEFAULT_LANG, filename, lang)),
        canonical=absolute_url(site_url, lang_url(lang, filename, lang)),
        skip_link=t("skip"),
        theme_label=t("theme_toggle"),
        menu_label=t("menu_open"),
        to_top_label=t("to_top"),
        foot_meta=t(
            "foot_stats",
            version=package["version"],
            modules=stats["modules"],
            functions=stats["functions"],
            lines=stats["lines"],
        ),
        site_i18n=json.dumps(t.js_all(), ensure_ascii=False),
        extra_js=extra_js,
    )


# --------------------------------------------------------------------------
# 首页
# --------------------------------------------------------------------------


def page_filename(data: dict, key: str) -> str:
    """取某个页面的文件名；页面不存在时返回空串。"""
    for link in data["meta"].get("page_links", []):
        if link.get("key") == key:
            return link.get("filename", "")
    return ""


def build_home(data: dict) -> str:
    package = data["package"]
    stats = data["stats"]
    modules = data["modules"]
    import_name = package.get("import_name") or package["name"]
    t = data["t"]

    api_page = page_filename(data, "api")
    cards = []
    for module in modules:
        count = e(t("functions_count", count=len(module["functions"])))
        # 没有 API 页时（包里还没有函数），卡片不做成链接
        if api_page:
            cards.append(
                f"""<a class="module-card" href="{e(api_page)}#{e(module['name'])}">
  <span class="module-name">{e(module['name'])}</span>
  <span class="module-count">{count}</span>
</a>"""
            )
        else:
            cards.append(
                f"""<div class="module-card">
  <span class="module-name">{e(module['name'])}</span>
  <span class="module-count">{count}</span>
</div>"""
            )

    # 首页按钮跟着实际生成的页面走，避免链到不存在的页面
    buttons = []
    for index, link in enumerate(data["meta"].get("page_links", [])):
        style = "btn" if index == 0 else "btn btn-ghost"
        buttons.append(f'<a class="{style}" href="{e(link["filename"])}">{e(link["label"])}</a>')
    hero_links = "\n      ".join(buttons)

    return f"""
<section class="hero">
  <div class="hero-text">
    <p class="eyebrow">{e(t("hero_eyebrow"))}</p>
    <h1>{e(display_name(package))}</h1>
    <p class="lead">{e(package['description'] or package.get('summary') or '')}</p>
    <div class="install">
      <code>pip install --upgrade {e(package['name'])}</code>
      <button class="copy" type="button" data-copy="pip install --upgrade {e(package['name'])}">{e(t("copy"))}</button>
    </div>
    <p class="hero-meta">{e(t("hero_version"))} <strong>v{e(package['version'])}</strong>
       · {e(t("hero_python", version=package['requires_python'] or '3.6+'))}</p>
    <p class="hero-links">
      {hero_links}
    </p>
  </div>
</section>

<section class="stats">
  <div class="stat"><span class="stat-value">{stats['modules']}</span><span class="stat-label">{e(t("stat_modules"))}</span></div>
  <div class="stat"><span class="stat-value">{stats['functions']}</span><span class="stat-label">{e(t("stat_functions"))}</span></div>
  <div class="stat"><span class="stat-value">{stats['lines']:,}</span><span class="stat-label">{e(t("stat_lines"))}</span></div>
</section>

<section class="section">
  <div class="section-head">
    <h2>{e(t("section_modules"))}</h2>
    {f'<a class="more" href="{e(api_page)}">{e(t("all_api"))}</a>' if api_page else ""}
  </div>
  <div class="module-grid">
{chr(10).join(cards)}
  </div>
</section>

<section class="section">
  <div class="split">
    <div class="panel">
      <h2>{e(t("section_package"))}</h2>
      <table class="params">
        <tbody>
          <tr><td>{e(t("field_display_name"))}</td><td><code>{e(display_name(package))}</code></td></tr>
          <tr><td>{e(t("field_pypi_name"))}</td><td><code>{e(package['name'])}</code></td></tr>
          <tr><td>{e(t("field_import_name"))}</td><td><code>{e(import_name)}</code></td></tr>
          <tr><td>{e(t("field_version"))}</td><td><code>{e(package['version'])}</code></td></tr>
          <tr><td>{e(t("field_python"))}</td><td><code>{e(package['requires_python'] or '>=3.6')}</code></td></tr>
          <tr><td>{e(t("field_homepage"))}</td><td><a href="{e(package.get('url') or '#')}">{e(package.get('url') or '—')}</a></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>
"""


# --------------------------------------------------------------------------
# API 参考
# --------------------------------------------------------------------------


def _function_block(
    function: dict,
    module_name: str,
    home_of: dict[str, str],
    import_name: str,
    t,
    anchor: str | None = None,
) -> str:
    params = function["params"]
    param_rows = []
    for param in params:
        default = param["default"]
        default_html = (
            f'<code class="param-default">{e(default)}</code>'
            if default
            else f'<span class="muted">{e(t("param_required"))}</span>'
        )
        annotation = (
            f'<span class="param-type">{e(param["annotation"])}</span>'
            if param["annotation"]
            else ""
        )
        param_rows.append(
            f'<tr><td><code>{e(param["name"])}</code></td><td>{annotation}</td><td>{default_html}</td></tr>'
        )
    params_html = (
        f"""<table class="params">
<thead><tr><th>{e(t("param_name"))}</th><th>{e(t("param_type"))}</th><th>{e(t("param_default"))}</th></tr></thead>
<tbody>
{chr(10).join(param_rows)}
</tbody>
</table>"""
        if param_rows
        else ""
    )

    badges = []
    if function.get("deprecated"):
        badges.append(f'<span class="badge badge-warn">{e(t("badge_deprecated"))}</span>')
    if not function.get("exported", True):
        badges.append(f'<span class="badge badge-muted">{e(t("badge_not_exported"))}</span>')
    if function.get("is_async"):
        badges.append('<span class="badge">async</span>')
    if function.get("decorators"):
        badges.append('<span class="badge">@' + e(function["decorators"][0]) + "</span>")
    badges_html = " ".join(badges)

    # 源码里的分节注释（形如 "# --- 某某某 ---"）作为小标签显示
    section_html = (
        f'<p class="func-section">{e(function["section"])}</p>'
        if function.get("section")
        else ""
    )

    calls_html = ""
    if function.get("calls"):
        links = []
        for call in function["calls"]:
            target = home_of.get(call)
            if target:
                links.append(
                    f'<a href="#{e(target)}"><code>{e(import_name)}.{e(call)}</code></a>'
                )
            else:
                links.append(f"<code>{e(import_name)}.{e(call)}</code>")
        calls_html = f'<p class="calls">{e(t("calls"))}{", ".join(links)}</p>'

    doc_html = _docstring_html(function)
    anchor_id = anchor or f"{module_name}.{function['name']}"
    owner = function.get("owner") or ""
    shown_name = f"{owner}.{function['name']}" if owner else function["name"]
    return f"""
<article class="func" id="{e(anchor_id)}">
  {section_html}
  <h3><code class="func-name">{e(shown_name)}</code>{badges_html}</h3>
  <p class="sig"><code>{e(function['signature'])}</code></p>
  {f'<p class="desc">{e(function["description"])}</p>' if function["description"] else ""}
  {doc_html}
  {params_html}
  {calls_html}
  <details class="source">
    <summary>{e(t("view_source"))}</summary>
    <pre class="code"><code>{e(function['source'])}</code></pre>
  </details>
</article>
"""


def _docstring_html(function: dict) -> str:
    """docstring 除了首段（已作为简短说明）之外的其余段落。"""
    docstring = (function.get("docstring") or "").strip()
    if not docstring:
        return ""
    paragraphs = [p.strip() for p in docstring.split("\n\n") if p.strip()]
    rest = paragraphs[1:]
    if not rest:
        return ""
    body = "".join(f"<p>{e(p)}</p>" for p in rest)
    return f'<div class="docstring">{body}</div>'


def build_api(data: dict) -> str:
    package = data["package"]
    modules = data["modules"]
    import_name = package.get("import_name") or package["name"]
    t = data["t"]

    # 函数名 → 该函数在页面上的锚点，用于生成交叉参考链接
    # 类的锚点是「模块.类」，类的方法是「模块.类.方法」，
    # 与模块级函数的「模块.函数」不一样
    home_of: dict[str, str] = {}
    for module in modules:
        for function in module["functions"]:
            home_of.setdefault(
                function["name"], f"{module['name']}.{function['name']}"
            )
        for klass in module["classes"]:
            home_of.setdefault(
                klass["name"], f"{module['name']}.{klass['name']}"
            )
            for method in klass["methods"]:
                home_of.setdefault(
                    method["name"], f"{module['name']}.{klass['name']}.{method['name']}"
                )

    nav_items = []
    body = []
    for module in modules:
        nav_items.append(
            f'<li><a href="#{e(module["name"])}"><span class="nav-name">{e(module["name"])}</span>'
            f'<span class="nav-count">{len(module["functions"])}</span></a></li>'
        )
        functions = "\n".join(
            _function_block(function, module["name"], home_of, import_name, t)
            for function in module["functions"]
        )
        classes_html = ""
        if module["classes"]:
            class_blocks = []
            for klass in module["classes"]:
                methods = "\n".join(
                    _function_block(
                        method,
                        module["name"],
                        home_of,
                        import_name,
                        t,
                        anchor=f"{module['name']}.{klass['name']}.{method['name']}",
                    )
                    for method in klass["methods"]
                )
                bases = (
                    f'<p class="muted">{e(t("base_classes"))}{", ".join(f"<code>{e(b)}</code>" for b in klass["bases"])}</p>'
                    if klass.get("bases")
                    else ""
                )
                class_blocks.append(
                    f"""<div class="class-block" id="{e(module['name'])}.{e(klass['name'])}">
  <h3 class="class-name">class {e(klass['name'])}</h3>
  {f'<p class="desc">{e(klass["description"])}</p>' if klass["description"] else ""}
  {bases}
  {methods}
</div>"""
                )
            classes_html = (
                f'<h3 class="group-title">{e(t("api_classes_title"))}</h3>\n'
                + "\n".join(class_blocks)
            )

        error_html = (
            f'<p class="badge badge-warn">{e(t("parse_error"))}：{e(module["error"])}</p>'
            if module.get("error")
            else ""
        )

        # 只有真的有函数时才写「函数」这个分组标题，
        # 否则像包根目录那种空模块会留下一对空标题和空网格
        functions_html = (
            f'<h3 class="group-title">{e(t("api_functions_title"))}</h3>\n'
            f'  <div class="func-grid">\n{functions}\n  </div>'
            if module["functions"]
            else ""
        )

        body.append(
            f"""<section class="module" id="{e(module['name'])}">
  <h2 class="module-title">{e(module['name'])}</h2>
  <p class="module-meta">{e(t("module_meta", lines=module['line_count'], count=len(module['functions'])))}</p>
  {error_html}
  {classes_html}
  {functions_html}
</section>"""
        )

    return f"""
<header class="page-head">
  <h1>{e(t("page_api"))}</h1>
</header>
<div class="api-layout">
  <aside class="api-nav">
    <p class="api-nav-title">{e(t("api_modules_title"))}</p>
    <ul>
{chr(10).join(nav_items)}
    </ul>
  </aside>
  <div class="api-body">
{chr(10).join(body)}
  </div>
</div>
"""


# --------------------------------------------------------------------------
# 关于页
# --------------------------------------------------------------------------


def _markdown_doc_page(
    data: dict,
    *,
    title: str,
    lead: str,
    source_path: str,
    file_key: str,
    show_toc: bool = True,
) -> str:
    """把一份 Markdown 文档渲染成页面内容。"""
    t = data["t"]
    doc_path = Path(data["meta"][file_key])
    if not doc_path.exists():
        return (
            f'<header class="page-head"><h1>{e(title)}</h1></header>'
            f'<p class="muted">{e(t("file_not_found", path=source_path))}</p>'
        )
    try:
        markdown = doc_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        try:
            markdown = doc_path.read_text(encoding="gbk")
        except (OSError, UnicodeDecodeError):
            markdown = ""
    rendered, toc = md_render.render_document(markdown)

    toc_html = ""
    if toc and show_toc:
        items = "\n".join(
            f'<li class="toc-l{entry["level"]}">'
            f'<a href="#{e(entry["id"])}">{e(entry["text"])}</a></li>'
            for entry in toc
        )
        toc_html = f"""
<aside class="toc">
  <p class="toc-title">{e(t("toc_title"))}</p>
  <ul>
{items}
  </ul>
</aside>
"""

    lead_html = f'  <p class="lead">{lead}</p>\n' if lead else ""
    layout_class = "doc-layout" if toc_html else "doc-layout doc-layout-flat"

    return f"""
<header class="page-head">
  <h1>{e(title)}</h1>
{lead_html}</header>
<div class="{layout_class}">
{toc_html}
  <article class="markdown">
{rendered}
  </article>
</div>
"""


def build_readme(data: dict) -> str:
    """把 README.md 渲染成页面内容。"""
    return _markdown_doc_page(
        data,
        title=data["t"]("page_about"),
        lead="",
        source_path=data["meta"]["readme_path"],
        file_key="readme_file",
    )


def build_installation(data: dict) -> str:
    """把 INSTALLATION.md 渲染成「安装」页。"""
    return _markdown_doc_page(
        data,
        title=data["t"]("page_installation"),
        lead="",
        source_path=data["meta"]["installation_path"],
        file_key="installation_file",
        show_toc=False,
    )


# --------------------------------------------------------------------------
# 教程页
# --------------------------------------------------------------------------


def _tutorial_card(item: dict, t) -> str:
    """一个示例脚本的卡片：标题、说明、源码。"""
    description = item.get("description") or ""
    description_html = (
        f'<p class="tut-desc">{md_render.render_inline(description)}</p>'
        if description
        else ""
    )

    source_link = (
        f' · <a href="{e(item["source"])}" rel="noopener">{e(t("view_original"))}</a>'
        if item.get("source")
        else ""
    )

    return f"""
<article class="tut-card" id="{e(item['anchor'])}">
  <div class="tut-card-head">
    <h3 class="tut-title">{e(item['title'])}</h3>
    <span class="tut-lines">{e(t("lines", count=item.get('lines', 0)))}</span>
  </div>
  {description_html}
  <details class="source">
    <summary>{e(t("view_source"))}</summary>
    <pre class="code"><code>{e(item['code'])}</code></pre>
  </details>
  <p class="tut-links">
    <a href="{e(item['path'])}" download="{e(Path(item['path']).name)}"
       data-download="{e(Path(item['path']).name)}">{e(t("download_script"))}</a>{source_link}
  </p>
</article>
"""


def build_tutorials(data: dict) -> str:
    """把 tutorials/ 目录里的示例脚本渲染成「教程」页。"""
    tutorials_data = data.get("tutorials") or {}
    groups = tutorials_data.get("groups", [])
    t = data["t"]

    nav_items = []
    body = []
    for group in groups:
        # 根目录下的脚本归为「开始使用」这一组，分组名按语言给
        title = group["title"]
        subtitle_text = group.get("subtitle") or ""
        if not group.get("key"):
            title = t("tut_root_title")
            subtitle_text = t("tut_root_subtitle")
        nav_items.append(
            f'<li><a href="#{e(group["anchor"])}">'
            f'<span class="nav-name">{e(title)}</span>'
            f'<span class="nav-count">{len(group["files"])}</span></a></li>'
        )
        cards = "\n".join(_tutorial_card(item, t) for item in group["files"])
        subtitle = (
            f'<p class="tut-group-sub">{e(subtitle_text)}</p>'
            if subtitle_text
            else ""
        )
        body.append(
            f"""<section class="tut-group" id="{e(group['anchor'])}">
  <h2 class="tut-group-title">{e(title)}</h2>
  {subtitle}
  <div class="tut-cards">
{cards}
  </div>
</section>"""
        )

    return f"""
<header class="page-head">
  <h1>{e(t("page_tutorials"))}</h1>
</header>
<div class="tut-layout">
  <aside class="toc tut-nav">
    <p class="toc-title">{e(t("tut_groups_title"))}</p>
    <ul>
{chr(10).join(nav_items)}
    </ul>
  </aside>
  <div class="tut-body">
{chr(10).join(body)}
  </div>
</div>
"""


# --------------------------------------------------------------------------
# 构建
# --------------------------------------------------------------------------


def _relative(path: Path) -> str:
    """尽量用相对仓库根目录的路径展示。"""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _metadata_dir(source_dir: Path) -> Path:
    """在源码目录附近找放 pyproject.toml / setup.cfg 的目录。"""
    candidates = [
        source_dir,
        source_dir.parent,
        source_dir.parent.parent,
        PYPI_DIR,
        ROOT,
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() or (candidate / "setup.cfg").exists():
            return candidate
    return source_dir


def _readme_file() -> Path:
    """找 README.md（先仓库根目录，再包目录）。"""
    for candidate in (ROOT / "README.md", PYPI_DIR / "README.md"):
        if candidate.exists():
            return candidate
    return ROOT / "README.md"


def _installation_file() -> Path:
    """找 INSTALLATION.md（先仓库根目录，再包目录）。"""
    for candidate in (ROOT / "INSTALLATION.md", PYPI_DIR / "INSTALLATION.md"):
        if candidate.exists():
            return candidate
    return ROOT / "INSTALLATION.md"


def copy_local_assets(
    output: Path, base_dirs: list[Path]
) -> tuple[list[str], list[str]]:
    """把页面里引用的本地文件（图片、文档等）复制到输出目录。

    README 里可能写 `![logo](docs/logo.png)` 或 `[说明](docs/notes.md)`，
    这些文件不在网站目录里，需要按相对路径复制过去，否则图片会裂开、
    链接会断掉。相对路径依次按 base_dirs 里的目录去找。

    站点有多个语言目录，每套页面里的引用都是相对于自己所在目录的，
    所以每个语言目录都要备一份这些文件。
    返回 (已复制, 找不到的图片)。
    """
    copied: list[str] = []
    missing: list[str] = []
    for page_file in sorted(output.rglob("*.html")):
        page_dir = page_file.parent
        text = page_file.read_text(encoding="utf-8")
        for kind, pattern in (("src", r'src="([^"]+)"'), ("href", r'href="([^"]+)"')):
            for raw in re.findall(pattern, text):
                src = html.unescape(raw).split("#")[0].split("?")[0]
                if not src or re.match(
                    r"^(?:https?:|data:|//|mailto:|javascript:)", src
                ):
                    continue
                if src.startswith("/"):
                    continue  # 站点根路径，交给服务器处理
                if (page_dir / src).exists():
                    continue  # 已经是网站自带的资源（如 static/style.css）
                source = _find_local_file(src, base_dirs)
                if source is None:
                    if kind == "src":
                        missing.append(src)
                    continue
                target = (page_dir / src).resolve()
                # 不允许复制到输出目录之外
                try:
                    target.relative_to(output.resolve())
                except ValueError:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(src)
    return copied, missing


def _find_local_file(name: str, base_dirs: list[Path]) -> Path | None:
    """在若干个基准目录里找本地文件。"""
    for base in base_dirs:
        candidate = (Path(base) / name).resolve()
        if candidate.is_file():
            return candidate
    return None


def clean_output(output: Path) -> None:
    """清空输出目录，只留下本次要生成的东西。

    先删文件、再按层级从深到浅删目录：某个文件被占用时，
    不影响其余文件的重建；实在删不掉的目录最后再整体试一次。
    """
    if not output.exists():
        return
    paths = sorted(output.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for path in paths:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        except OSError:
            pass
    for path in paths:
        if path.is_dir():
            try:
                shutil.rmtree(path)
            except OSError:
                pass


def verify_output(output: Path) -> list[str]:
    """检查生成的页面：内部链接指向的文件与锚点都必须存在。

    源码一旦大改，很容易出现「函数改名后旧锚点失效」这类问题，
    构建时顺手查一遍，可以第一时间发现。

    链接都是相对于页面自己所在目录的，所以每个语言目录都要单独查。
    """
    problems: list[str] = []
    pages = sorted(output.rglob("*.html"))
    anchors: dict[str, set[str]] = {}
    for page_file in pages:
        text = page_file.read_text(encoding="utf-8")
        key = page_file.relative_to(output).as_posix()
        anchors[key] = set(re.findall(r'id="([^"]+)"', text))

    for page_file in pages:
        text = page_file.read_text(encoding="utf-8")
        key = page_file.relative_to(output).as_posix()
        for href in re.findall(r'href="([^"]+)"', text):
            href = html.unescape(href)
            if href.startswith(("http://", "https://", "mailto:", "data:", "javascript:")):
                continue
            if href.startswith("#"):
                anchor = urllib.parse.unquote(href[1:])
                if anchor and anchor not in anchors[key]:
                    problems.append(f"{key}：锚点 {href} 不存在")
                continue
            target, _, fragment = href.partition("#")
            target = urllib.parse.unquote(target)
            if not target:
                continue
            target_path = (page_file.parent / target).resolve()
            if not target_path.exists():
                problems.append(f"{key}：链接 {href} 指向的文件不存在")
                continue
            if fragment and target.endswith(".html"):
                try:
                    target_key = target_path.relative_to(output.resolve()).as_posix()
                except ValueError:
                    continue
                if urllib.parse.unquote(fragment) not in anchors.get(target_key, set()):
                    problems.append(f"{key}：链接 {href} 的锚点不存在")
        # 页面里引用的本地文件（图片等）也要在
        for src in re.findall(r'src="([^"]+)"', text):
            src = html.unescape(src).split("#")[0].split("?")[0]
            if not src or src.startswith(("http:", "https:", "data:", "//", "/")):
                continue
            if not (page_file.parent / urllib.parse.unquote(src)).exists():
                problems.append(f"{key}：引用的文件 {src} 不存在")
    return problems


def generate(
    source_dir: Path | None = None,
    output: Path = ROOT / "dist" / "site",
    readme_path: Path | None = None,
    installation_path: Path | None = None,
    tutorials_dir: Path | None = None,
    base_url: str | None = None,
    quiet: bool = False,
) -> dict:
    """生成网站，返回提取到的数据字典。

    source_dir       包源码目录；为 None 时自动发现
    output           网站输出目录
    readme_path      README.md 路径，用于生成「关于」页面
    installation_path INSTALLATION.md 路径，用于生成「安装」页面
    tutorials_dir    示例脚本目录；为 None 时自动发现，没有就不生成「教程」页
    base_url         站点根地址；为 None 时从包元数据里的主页地址推
    """
    if source_dir is None:
        source_dir = extract.discover_package_dir(list(SOURCE_CANDIDATES))
    if source_dir is None:
        raise SystemExit(
            "找不到包源码目录。请用 --source 指定，例如 --source PyPI/src/<包名>"
        )
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"包源码目录不存在或不是目录：{source_dir}")

    output = Path(output).resolve()
    clean_output(output)
    output.mkdir(parents=True, exist_ok=True)

    # 界面文案必须两种语言都写全，缺了就直接报错，
    # 免得构建出「一半中文一半英文」的页面
    incomplete = i18n.missing_keys()
    if incomplete:
        raise SystemExit(
            "界面文案不完整：\n  " + "\n  ".join(incomplete)
        )

    # 解析、导出关系、交叉参考分三步做，每一步只用做一次
    modules = extract.parse_modules(source_dir)
    metadata_dir = _metadata_dir(source_dir)
    info = extract.parse_package_info(
        metadata_dir / "pyproject.toml",
        metadata_dir / "setup.cfg",
        source_dir / "__init__.py",
        package_dir=source_dir,
        import_name=source_dir.name,
    )
    exports, _ = extract.compute_exports(source_dir, modules)
    extract.apply_exports(modules, exports)
    extract.resolve_calls(modules, source_dir)
    data = extract.to_dict(modules, info, exports)

    readme = Path(readme_path) if readme_path else _readme_file()
    installation = Path(installation_path) if installation_path else _installation_file()
    data["meta"] = {
        "source_path": _relative(source_dir),
        "readme_path": _relative(readme),
        "readme_file": str(readme),
        "installation_path": _relative(installation),
        "installation_file": str(installation),
        "metadata_dir": _relative(metadata_dir),
    }
    # 教程：tutorials/ 目录里的示例脚本，没有这个目录就不生成「教程」页
    if tutorials_dir is None:
        tutorials_dir = next(
            (path for path in TUTORIALS_CANDIDATES if path.is_dir()), None
        )
    if tutorials_dir is not None:
        tutorials_dir = Path(tutorials_dir).resolve()
        data["tutorials"] = tutorials.collect(tutorials_dir)
        data["meta"]["tutorials_path"] = _relative(tutorials_dir)

    # 站点根地址：hreflang、canonical、sitemap 都要用绝对地址
    site_url = site_base_url(data["package"], base_url)
    data["meta"]["site_url"] = site_url

    stats = data["stats"]
    shell = load_template("base.html")
    pages = active_pages(data)

    # 每种语言生成一套页面：中文在根目录，英文在 en/ 子目录。
    # 页面里的链接都相对于自己所在目录，所以每个目录都是自洽的。
    for lang in i18n.LANGS:
        context = _lang_context(data, lang, pages)
        directory = lang_dir(output, lang)
        directory.mkdir(parents=True, exist_ok=True)
        for spec in pages:
            content = spec.builder(context)
            html_text = page(
                shell,
                title=spec.page_title(context),
                description=spec.description(context),
                content=content,
                info=data["package"],
                stats=stats,
                pages=pages,
                active=spec.key,
                lang=lang,
                filename=spec.filename,
                site_url=site_url,
            )
            (directory / spec.filename).write_text(html_text, encoding="utf-8")

    # 机器可读的完整 API 数据
    # 对外发布的产物里只放相对路径，不泄露本机的绝对路径
    api_data = dict(data)
    api_data["meta"] = {
        **data["meta"],
        "readme_file": data["meta"]["readme_path"],
        "installation_file": data["meta"]["installation_path"],
    }
    if "tutorials" in api_data:
        # 源码正文太大，索引里只留标题、说明这些摘要信息
        api_data["tutorials"] = tutorials.without_code(api_data["tutorials"])
    (output / "api.json").write_text(
        json.dumps(api_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 静态资源：每个语言目录里都放一份
    for lang in i18n.LANGS:
        static_output = lang_dir(output, lang) / "static"
        static_output.mkdir(parents=True, exist_ok=True)
        for asset in STATIC_DIR.glob("*"):
            if asset.is_file():
                shutil.copy2(asset, static_output / asset.name)

    # README / INSTALLATION / 教程里引用的本地图片、文档与脚本
    base_dirs = [readme.parent, installation.parent, ROOT]
    if tutorials_dir is not None:
        base_dirs.append(Path(tutorials_dir).parent)
    copied, missing = copy_local_assets(output, base_dirs)

    errors = [m for m in modules if m.error]
    problems = verify_output(output)
    if not quiet:
        print(f"网站已生成：{output}")
        print(
            f"  {stats['modules']} 个模块 · {stats['functions']} 个函数 "
            f"· {stats['classes']} 个类 · {stats['lines']} 行源码"
        )
        print(
            "  语言："
            + " · ".join(
                f"{i18n.LANG_NAMES[lang]}（{lang_dir(output, lang).relative_to(output) or '.'}）"
                for lang in i18n.LANGS
            )
        )
        print(f"  页面：{', '.join(spec.filename for spec in pages)}")
        if data.get("tutorials"):
            groups = data["tutorials"]["groups"]
            names = "、".join(f"{group['title']}（{len(group['files'])}）" for group in groups)
            print(f"  教程：{data['tutorials']['count']} 个示例脚本 — {names}")
        for module in errors:
            print(f"  ! {module.path}：{module.error}")
        for name in missing:
            print(f"  ! 找不到图片资源：{name}")
        for problem in problems:
            print(f"  ! {problem}")
        if problems:
            print(f"  （共 {len(problems)} 处链接问题）")
        if copied:
            print(f"  已复制 {len(copied)} 个本地资源：{', '.join(sorted(set(copied)))}")

    # 供调用方判断构建是否干净
    data["problems"] = problems
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="生成软件包的网站")
    parser.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "dist" / "site"),
        help="网站输出目录（默认 dist/site）",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="包源码目录（默认自动发现，例如 PyPI/src/<包名>）",
    )
    parser.add_argument(
        "--readme",
        default=None,
        help="README.md 路径（默认仓库根目录的 README.md）",
    )
    parser.add_argument(
        "--installation",
        default=None,
        help="INSTALLATION.md 路径（默认仓库根目录的 INSTALLATION.md）",
    )
    parser.add_argument(
        "--tutorials",
        default=None,
        help="示例脚本目录（默认仓库根目录的 tutorials/，没有则不生成教程页）",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="只生成网站，不打印检查结果",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="发现断链或解析错误时以非零状态退出（用于 CI）",
    )
    args = parser.parse_args()

    data = generate(
        source_dir=Path(args.source) if args.source else None,
        output=Path(args.output),
        readme_path=Path(args.readme) if args.readme else None,
        installation_path=Path(args.installation) if args.installation else None,
        tutorials_dir=Path(args.tutorials) if args.tutorials else None,
        quiet=args.quiet,
    )

    if args.strict:
        problems = list(data.get("problems") or [])
        problems += [
            f"源码解析失败：{module.path}"
            for module in data.get("modules", [])
            if module.get("error")
        ]
        if problems:
            print(f"严格模式下发现 {len(problems)} 个问题，构建失败。")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
