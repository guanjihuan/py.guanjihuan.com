"""站点界面的中英文文案。

只有「网站框架」的部分需要翻译：导航、按钮、页面标题、页脚统计等等。
文档正文（README.md、INSTALLATION.md）、源码里的说明与注释、教程脚本
都保持原样，不在这里翻译——它们是文件内容，不是界面。

站点是静态的，所以不做运行时切换，而是构建时生成两套页面：
  * 中文在站点根目录（index.html）
  * 英文在 en/ 子目录（en/index.html）
两套页面里的文件引用都相对于各自所在的目录，所以每个语言目录都是自洽的。

新增文案时，zh 与 en 两边都要写；缺哪一边，构建时会直接报错，
免得网站上线后有一半界面还是另一种语言。
"""

from __future__ import annotations

# 站点支持的语言，按顺序构建
LANGS: tuple[str, ...] = ("zh", "en")

# 默认语言（首页用的语言，也是没有指定语言时的回退）
DEFAULT_LANG = "zh"

# 每种语言的页面放在站点根目录下的哪个子目录
LANG_DIRS: dict[str, str] = {
    "zh": "",
    "en": "en",
}

# <html lang="..."> 与 hreflang 用的取值
HTML_LANG: dict[str, str] = {
    "zh": "zh-CN",
    "en": "en",
}

# 语言在界面上的名字（切换按钮上显示的）
LANG_NAMES: dict[str, str] = {
    "zh": "中",
    "en": "En",
}

# 语言切换按钮上显示的完整语言名。
# 用完整名字而不是「中 / En」，是因为无障碍规则要求：
# 按钮上看得见的文字必须包含在它给读屏软件的无障碍名称里。
# 「En」配「切换语言」对不上，写成「English」配「切换到 English 界面」就没问题。
LANG_NAMES_FULL: dict[str, str] = {
    "zh": "中文",
    "en": "English",
}

# --------------------------------------------------------------------------
# 界面文案
# --------------------------------------------------------------------------

STRINGS: dict[str, dict[str, str]] = {
    "zh": {
        # 顶栏与页脚
        "skip": "跳到主要内容",
        "theme_toggle": "切换深浅色主题",
        "menu_open": "打开菜单",
        "menu_close": "关闭菜单",
        "to_top": "回到顶部",
        "foot_stats": "{version} · {modules} 个模块 · {functions} 个函数 · {lines} 行源码",
        "switch_language": "切换语言",
        "switch_to_lang": "切换到 {lang} 界面",
        # 导航
        "nav_home": "首页",
        "nav_installation": "安装",
        "nav_api": "API 参考",
        "nav_tutorials": "教程",
        "nav_about": "关于",
        # 页面标题与描述
        "page_home": "{name} 官方网站",
        "page_installation": "安装",
        "page_api": "API 参考",
        "page_tutorials": "教程",
        "page_about": "关于",
        "desc_installation": "{name} 的安装与使用说明",
        "desc_api": "{name} 的完整 API 参考",
        "desc_tutorials": "{name} 的教程与示例代码",
        "desc_about": "关于 {name}",
        # 首页
        "hero_eyebrow": "Python package",
        "hero_version": "当前版本",
        "hero_python": "支持 Python {version}",
        "copy": "复制",
        "stat_modules": "模块",
        "stat_functions": "函数",
        "stat_lines": "行源码",
        "section_modules": "模块一览",
        "all_api": "全部 API →",
        "section_package": "包信息",
        "field_display_name": "展示名",
        "field_pypi_name": "PyPI 名称",
        "field_import_name": "导入名",
        "field_version": "版本",
        "field_python": "Python",
        "field_homepage": "主页",
        # API 参考
        "api_modules_title": "模块",
        "api_functions_title": "函数",
        "api_classes_title": "类",
        "param_name": "参数",
        "param_type": "类型",
        "param_default": "默认值",
        "param_required": "必填",
        "badge_deprecated": "已弃用",
        "badge_not_exported": "未导出",
        "parse_error": "解析失败",
        "view_source": "查看源码",
        "calls": "调用了：",
        "base_classes": "基类：",
        "module_meta": "{lines} 行 · {count} 个函数",
        # 关于 / 安装 / 教程
        "toc_title": "目录",
        "tut_groups_title": "分组",
        "tut_root_title": "开始使用",
        "tut_root_subtitle": "Getting started",
        "file_not_found": "未找到 {path}。",
        "download_script": "下载脚本",
        "view_original": "查看原文",
        "lines": "{count} 行",
        "functions_count": "{count} 个函数",
    },
    "en": {
        # 顶栏与页脚
        "skip": "Skip to main content",
        "theme_toggle": "Toggle light and dark theme",
        "menu_open": "Open menu",
        "menu_close": "Close menu",
        "to_top": "Back to top",
        "foot_stats": (
            "{version} · {modules} modules · {functions} functions · {lines} lines of code"
        ),
        "switch_language": "Switch language",
        "switch_to_lang": "Switch to {lang} interface",
        # 导航
        "nav_home": "Home",
        "nav_installation": "Install",
        "nav_api": "API Reference",
        "nav_tutorials": "Tutorials",
        "nav_about": "About",
        # 页面标题与描述
        "page_home": "{name} official site",
        "page_installation": "Installation",
        "page_api": "API Reference",
        "page_tutorials": "Tutorials",
        "page_about": "About",
        "desc_installation": "How to install and use {name}",
        "desc_api": "The complete API reference of {name}",
        "desc_tutorials": "Tutorials and example code for {name}",
        "desc_about": "About {name}",
        # 首页
        "hero_eyebrow": "Python package",
        "hero_version": "Current version",
        "hero_python": "Supports Python {version}",
        "copy": "Copy",
        "stat_modules": "Modules",
        "stat_functions": "Functions",
        "stat_lines": "Lines of code",
        "section_modules": "Modules",
        "all_api": "All API →",
        "section_package": "Package info",
        "field_display_name": "Display name",
        "field_pypi_name": "PyPI name",
        "field_import_name": "Import name",
        "field_version": "Version",
        "field_python": "Python",
        "field_homepage": "Homepage",
        # API 参考
        "api_modules_title": "Modules",
        "api_functions_title": "Functions",
        "api_classes_title": "Classes",
        "param_name": "Parameter",
        "param_type": "Type",
        "param_default": "Default",
        "param_required": "Required",
        "badge_deprecated": "Deprecated",
        "badge_not_exported": "Not exported",
        "parse_error": "Parse error",
        "view_source": "View source",
        "calls": "Calls: ",
        "base_classes": "Base classes: ",
        "module_meta": "{lines} lines · {count} functions",
        # 关于 / 安装 / 教程
        "toc_title": "Contents",
        "tut_groups_title": "Groups",
        "tut_root_title": "Getting started",
        "tut_root_subtitle": "Start here",
        "file_not_found": "{path} not found.",
        "download_script": "Download script",
        "view_original": "View original",
        "lines": "{count} lines",
        "functions_count": "{count} functions",
    },
}

# --------------------------------------------------------------------------
# 脚本里用到的文案
# --------------------------------------------------------------------------
# site.js 在运行时才生成一些按钮（复制按钮、筛选框……），
# 这些文字没法在构建时写进 HTML，只能由页面把文案传给脚本。

JS_STRINGS: dict[str, dict[str, str]] = {
    "zh": {
        "copied": "已复制",
        "copy": "复制",
        "copy_code": "复制代码",
        "copy_anchor": "复制本函数的链接",
        "theme_to_light": "切换到浅色主题",
        "theme_to_dark": "切换到深色主题",
        "menu_open": "打开菜单",
        "menu_close": "关闭菜单",
        "filter_placeholder": "在本页筛选函数…",
        "filter_label": "筛选函数",
        "filter_clear": "清除",
        "filter_count": "显示 {shown} / {total} 个函数",
    },
    "en": {
        "copied": "Copied",
        "copy": "Copy",
        "copy_code": "Copy code",
        "copy_anchor": "Copy link to this function",
        "theme_to_light": "Switch to light theme",
        "theme_to_dark": "Switch to dark theme",
        "menu_open": "Open menu",
        "menu_close": "Close menu",
        "filter_placeholder": "Filter functions on this page…",
        "filter_label": "Filter functions",
        "filter_clear": "Clear",
        "filter_count": "Showing {shown} / {total} functions",
    },
}


class Translator:
    """按语言取文案的小工具。

    t("nav_api") 取文案；t("lines", count=3) 取带占位符的文案。
    找不到的键原样返回，方便一眼看出漏了哪一条。
    """

    def __init__(self, lang: str) -> None:
        if lang not in STRINGS:
            raise ValueError(f"没有这种语言：{lang}")
        self.lang = lang
        self._strings = STRINGS[lang]
        self._js = JS_STRINGS[lang]

    def __call__(self, key: str, **kwargs: object) -> str:
        text = self._strings.get(key)
        if text is None:
            # 漏写时退回默认语言，再退回键名本身
            text = STRINGS[DEFAULT_LANG].get(key, key)
        if not kwargs:
            return text
        return text.format(**kwargs)

    def js(self, key: str, **kwargs: object) -> str:
        text = self._js.get(key)
        if text is None:
            text = JS_STRINGS[DEFAULT_LANG].get(key, key)
        if not kwargs:
            return text
        return text.format(**kwargs)

    def js_all(self) -> dict[str, str]:
        """整份脚本文案，供页面传给 site.js。"""
        return dict(self._js)


def translator(lang: str) -> Translator:
    return Translator(lang)


def missing_keys() -> list[str]:
    """检查各语言的文案是否齐全，返回缺失的键。

    构建前跑一次，缺哪边就补哪边——比上线后才发现强。
    """
    problems: list[str] = []
    reference = set(STRINGS[DEFAULT_LANG])
    for lang, strings in STRINGS.items():
        for key in sorted(reference - set(strings)):
            problems.append(f"{lang} 缺少文案 {key}")
        for key in sorted(set(strings) - reference):
            problems.append(f"{lang} 多出文案 {key}")
    js_reference = set(JS_STRINGS[DEFAULT_LANG])
    for lang, strings in JS_STRINGS.items():
        for key in sorted(js_reference - set(strings)):
            problems.append(f"{lang} 缺少脚本文案 {key}")
        for key in sorted(set(strings) - js_reference):
            problems.append(f"{lang} 多出脚本文案 {key}")
    return problems
