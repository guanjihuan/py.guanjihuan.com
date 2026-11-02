"""审计生成的站点：链接、HTML 结构、SEO、可访问性。

用法：
    python website/audit_site.py [站点目录]
默认审计 website/dist/site。
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
SITE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "website" / "dist" / "site"

problems: list[str] = []
notes: list[str] = []


def report(kind: str, page: str, message: str) -> None:
    problems.append(f"[{kind}] {page}: {message}")


HTML_PAGES = sorted(SITE.rglob("*.html"))
all_files = {p.relative_to(SITE).as_posix() for p in SITE.rglob("*") if p.is_file()}

# ---------------------------------------------------------------------------
# 1. 链接检查
# ---------------------------------------------------------------------------
ATTR_RE = re.compile(r"""\b(href|src)\s*=\s*(["'])(.*?)\2""", re.S | re.I)
ID_RE = re.compile(r"""\bid\s*=\s*(["'])(.*?)\1""", re.S | re.I)

page_ids: dict[str, set[str]] = {}
for page in HTML_PAGES:
    text = page.read_text(encoding="utf-8")
    page_ids[page.relative_to(SITE).as_posix()] = {m.group(2) for m in ID_RE.finditer(text)}

broken_links = 0
external_links: Counter[str] = Counter()

for page in HTML_PAGES:
    rel = page.relative_to(SITE).as_posix()
    text = page.read_text(encoding="utf-8")
    for match in ATTR_RE.finditer(text):
        attr, raw = match.group(1).lower(), match.group(3).strip()
        if not raw or raw.startswith(("javascript:", "data:", "mailto:", "tel:", "#")):
            if raw.startswith("#"):
                anchor = unquote(raw[1:])
                if anchor and anchor not in page_ids[rel]:
                    report("锚点", rel, f"指向本页不存在的 id: {raw}")
            continue
        parsed = urlparse(raw)
        if parsed.scheme in ("http", "https"):
            external_links[parsed.netloc] += 1
            continue
        if parsed.netloc:
            report("链接", rel, f"未知协议/域名: {raw}")
            continue
        target = unquote(parsed.path)
        if not target:
            continue
        base = page.parent
        dest = (base / target).resolve()
        try:
            dest_rel = dest.relative_to(SITE.resolve())
        except ValueError:
            report("链接", rel, f"指向站点外部: {raw}")
            continue
        if target.endswith("/"):
            dest = dest / "index.html"
        if not dest.exists():
            report("链接", rel, f"目标不存在: {raw}")
            broken_links += 1
        elif parsed.fragment:
            frag = unquote(parsed.fragment)
            dest_rel = dest_rel.as_posix()
            if dest_rel in page_ids and frag not in page_ids[dest_rel]:
                report("锚点", rel, f"{raw} -> 目标页无 id: {frag}")

# ---------------------------------------------------------------------------
# 2. HTML 结构检查
# ---------------------------------------------------------------------------
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}
BLOCK = {"html", "head", "body", "div", "main", "header", "footer", "nav",
         "section", "article", "aside", "ul", "ol", "li", "table", "tbody",
         "thead", "tr", "td", "th", "form", "p", "h1", "h2", "h3", "h4", "h5",
         "h6", "pre", "blockquote", "details", "summary", "figure", "picture",
         "video", "audio", "dl", "dt", "dd", "hr", "fieldset", "option",
         "select", "textarea", "button", "nav", "template", "svg", "canvas",
         "iframe", "span", "label", "code", "a", "strong", "em", "b", "i",
         "small", "sub", "sup", "br", "img", "table", "caption", "colgroup",
         "style", "script", "noscript"}

for page in HTML_PAGES:
    rel = page.relative_to(SITE).as_posix()
    text = page.read_text(encoding="utf-8")

    # 重定向页（不带 .html 的入口）只有一个跳转，不需要完整的页面结构
    if 'http-equiv="refresh"' in text:
        continue

    # doctype
    if not text.lstrip().lower().startswith("<!doctype html"):
        report("结构", rel, "缺少 <!DOCTYPE html>")

    # lang 属性
    m = re.search(r"<html\b([^>]*)>", text, re.I)
    if not m:
        report("结构", rel, "缺少 <html> 标签")
    elif not re.search(r"\blang\s*=", m.group(1), re.I):
        report("SEO", rel, "<html> 缺少 lang 属性")

    # title
    if not re.search(r"<title>.+?</title>", text, re.S | re.I):
        report("SEO", rel, "缺少非空 <title>")

    # meta description
    md = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', text, re.S | re.I)
    if not md:
        report("SEO", rel, "缺少 meta description")
    elif not md.group(1).strip():
        report("SEO", rel, "meta description 为空")

    # viewport
    if not re.search(r'<meta\s+name=["\']viewport["\']', text, re.I):
        report("SEO", rel, "缺少 viewport meta")

    # 标题层级：h1 应唯一
    h1s = re.findall(r"<h1\b", text, re.I)
    if len(h1s) != 1:
        report("结构", rel, f"h1 数量为 {len(h1s)}（应为 1）")

    # 标题层级不跳级
    levels = [int(m) for m in re.findall(r"<h([1-6])\b", text, re.I)]
    for prev, cur in zip(levels, levels[1:]):
        if cur > prev + 1:
            report("结构", rel, f"标题层级跳跃 h{prev} -> h{cur}")

    # 标签配对（简单栈式检查，忽略 void 与 script/style 内容）
    stripped = re.sub(r"<(script|style)\b.*?</\1>", "", text, flags=re.S | re.I)
    stack: list[str] = []
    for m in re.finditer(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)\b[^>]*?(/?)>", stripped):
        closing, tag, selfclose = m.group(1), m.group(2).lower(), m.group(3)
        if tag in VOID or selfclose:
            continue
        if closing:
            if not stack:
                report("结构", rel, f"多余的 </{tag}>")
            elif stack[-1] != tag:
                if tag in stack:
                    while stack and stack[-1] != tag:
                        report("结构", rel, f"标签未闭合: <{stack.pop()}>")
                    stack.pop()
                else:
                    report("结构", rel, f"标签错配: </{tag}> 对应 <{stack[-1]}>")
                    stack.pop()
            else:
                stack.pop()
        else:
            stack.append(tag)
    for leftover in reversed(stack):
        report("结构", rel, f"标签未闭合: <{leftover}>")

    # 重复 id
    ids = [m.group(2) for m in ID_RE.finditer(text)]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        report("结构", rel, f"重复 id: {dupes[:5]}")

    # 图片 alt
    for m in re.finditer(r"<img\b[^>]*>", text, re.I | re.S):
        if not re.search(r"\balt\s*=", m.group(0), re.I):
            report("可访问性", rel, f"img 缺少 alt: {m.group(0)[:70]}")

    # 链接文本
    for m in re.finditer(r"<a\b([^>]*)>(.*?)</a>", text, re.S | re.I):
        attrs, inner = m.group(1), m.group(2)
        if 'aria-label="' in attrs or "aria-label='" in attrs:
            continue
        text_only = re.sub(r"<[^>]+>", "", inner).strip()
        if not text_only and not re.search(r"\bdownload\b", attrs, re.I):
            report("可访问性", rel, f"空链接文本: {attrs[:70]}")

    # 按钮缺 type
    for m in re.finditer(r"<button\b[^>]*>", text, re.I):
        if not re.search(r"\btype\s*=", m.group(0), re.I):
            report("结构", rel, "button 缺少 type 属性")

    # 硬编码语言
    if 'lang="en"' in text.lower() and "en/" in rel:
        pass

# ---------------------------------------------------------------------------
# 3. 跨页面语言检查
# ---------------------------------------------------------------------------
for page in HTML_PAGES:
    rel = page.relative_to(SITE).as_posix()
    text = page.read_text(encoding="utf-8")
    m = re.search(r"<html\b([^>]*)>", text, re.I)
    lang = re.search(r'lang=["\']([^"\']+)["\']', m.group(1) if m else "")
    expected = "en" if rel.startswith("en/") else "zh-CN"
    if lang and lang.group(1) != expected:
        report("i18n", rel, f"lang={lang.group(1)} 期望 {expected}")

# 中英文页面数量配平
zh_pages = {p.relative_to(SITE).as_posix() for p in HTML_PAGES if not p.relative_to(SITE).as_posix().startswith("en/")}
en_pages = {p.relative_to(SITE).as_posix()[3:] for p in HTML_PAGES if p.relative_to(SITE).as_posix().startswith("en/")}
if zh_pages != en_pages:
    only_zh = sorted(zh_pages - en_pages)
    only_en = sorted(en_pages - zh_pages)
    if only_zh:
        report("i18n", "-", f"只有中文的页面: {only_zh}")
    if only_en:
        report("i18n", "-", f"只有英文的页面: {only_en}")

# ---------------------------------------------------------------------------
# 4. 静态资源
# ---------------------------------------------------------------------------
for page in HTML_PAGES:
    rel = page.relative_to(SITE).as_posix()
    text = page.read_text(encoding="utf-8")
    for m in re.finditer(r'<script\b[^>]*\bsrc=["\']([^"\']+)["\']', text, re.I):
        src = urlparse(m.group(1)).path
        p = (page.parent / unquote(src)).resolve()
        if not p.exists():
            report("资源", rel, f"脚本不存在: {m.group(1)}")

# 孤立的静态资源
referenced: set[str] = set()
for page in HTML_PAGES:
    text = page.read_text(encoding="utf-8")
    for m in ATTR_RE.finditer(text):
        raw = m.group(3).strip()
        if raw.startswith(("http", "mailto:", "javascript:", "data:", "#")):
            continue
        target = unquote(urlparse(raw).path)
        if target:
            referenced.add(((page.parent / target).resolve()).as_posix())
for f in SITE.rglob("*"):
    if f.is_file() and f.suffix.lower() in (".css", ".js", ".jpg", ".png", ".svg", ".ico"):
        if f.resolve().as_posix() not in referenced:
            notes.append(f"未被引用的静态文件: {f.relative_to(SITE).as_posix()}")

# ---------------------------------------------------------------------------
# 5. api.json
# ---------------------------------------------------------------------------
api_json = SITE / "api.json"
if api_json.exists():
    try:
        data = json.loads(api_json.read_text(encoding="utf-8"))
        notes.append(f"api.json 解析正常，顶层键: {list(data)[:8] if isinstance(data, dict) else type(data)}")
    except Exception as exc:
        report("api.json", "-", f"解析失败: {exc}")

# ---------------------------------------------------------------------------
# 6. 控制台输出
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"审计目录: {SITE}")
print(f"页面数: {len(HTML_PAGES)}  文件数: {len(all_files)}")
print("=" * 60)
if problems:
    print(f"\n发现 {len(problems)} 个问题：\n")
    for p in problems:
        print("  " + p)
else:
    print("\n未发现问题。")
if notes:
    print(f"\n提示 ({len(notes)}):")
    for n in notes:
        print("  - " + n)
if external_links:
    print("\n外部域名引用:")
    for domain, count in external_links.most_common():
        print(f"  {domain}: {count}")
print()
sys.exit(1 if problems else 0)
