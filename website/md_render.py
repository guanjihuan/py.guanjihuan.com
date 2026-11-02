"""把 Markdown 转成 HTML。

构建后端只依赖标准库，所以这里自带一个精简的 Markdown 渲染器，
覆盖 README 里常见与可能出现的写法：

  * 标题（# ~ ######）与 setext 标题（=== / ---）
  * 段落
  * 无序列表（+ - *）与有序列表（1. 1)），支持嵌套与任务列表
  * 围栏代码块（```）与缩进代码块（4 空格）
  * 行内代码、**粗体**、*斜体*、~~删除线~~
  * [文字](链接)、![图片](链接)、引用式链接 [文字][ref] 与 [ref]: 地址
  * 裸露的 URL、邮箱、<url> 自动链接
  * 引用块（>）
  * 分隔线（---）
  * 表格（GitHub 风格）
"""

from __future__ import annotations

import html
import re

# --------------------------------------------------------------------------
# 行内
# --------------------------------------------------------------------------

CODE_SPAN_RE = re.compile(r"`([^`]+)`")
# 链接地址里允许出现一层成对的括号，例如维基百科风格的链接
_URL = r"[^()\s]*(?:\([^()]*\)[^()\s]*)*"
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((" + _URL + r")(?:\s+\"[^\"]*\")?\)")
LINK_RE = re.compile(r"\[([^\]]*)\]\((" + _URL + r")(?:\s+\"[^\"]*\")?\)")
REF_LINK_RE = re.compile(r"\[([^\]]*)\]\[([^\]]*)\]")
AUTOLINK_RE = re.compile(r"https?://[^\s<>\"'），,。]+")
ANGLE_AUTOLINK_RE = re.compile(r"<((?:https?|mailto):[^>\s]+)>")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
ITALIC_RE = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
STRIKE_RE = re.compile(r"~~(.+?)~~")
HR_RE = re.compile(r"^(\s*)([-*_])(\s*\2){2,}\s*$")

# 行内链接引用定义： [ref]: https://...
REF_DEF_RE = re.compile(r"^ {0,3}\[([^\]]+)\]:\s*(\S+)(?:\s+\"[^\"]*\")?\s*$")

# 列表项：允许只有一个符号（空列表项），但符号与文字之间必须有空格
LIST_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<bullet>[-*+]|\d+[.)])(?:[ \t]+(?P<text>.*))?$"
)

_PLACEHOLDER = "\x00{}\x00"


def _stash(store: list[str], html_text: str) -> str:
    """把已渲染好的片段存起来，返回占位符。"""
    store.append(html_text)
    return _PLACEHOLDER.format(len(store) - 1)


def _restore(text: str, store: list[str]) -> str:
    for index, value in enumerate(store):
        text = text.replace(_PLACEHOLDER.format(index), value)
    return text


def render_inline(text: str, references: dict[str, str] | None = None) -> str:
    """渲染行内标记。"""
    store: list[str] = []
    references = references or {}

    # 行内代码：内容原样转义，不参与后续的强调处理
    def code_sub(match: re.Match) -> str:
        return _stash(store, f"<code>{html.escape(match.group(1))}</code>")

    text = CODE_SPAN_RE.sub(code_sub, text)

    # 图片：![alt](url)
    def image_sub(match: re.Match) -> str:
        alt = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        return _stash(store, f'<img src="{url}" alt="{alt}">')

    text = IMAGE_RE.sub(image_sub, text)

    # 引用式链接 [文字][ref]（ref 为空时用文字本身作 ref）
    def ref_sub(match: re.Match) -> str:
        label = match.group(1)
        ref = (match.group(2) or label).strip().lower()
        url = references.get(ref)
        if not url:
            return match.group(0)
        return _stash(
            store,
            f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>',
        )

    text = REF_LINK_RE.sub(ref_sub, text)

    # 链接：[文字](地址)
    def link_sub(match: re.Match) -> str:
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        return _stash(store, f'<a href="{url}">{label}</a>')

    text = LINK_RE.sub(link_sub, text)

    # <url> 自动链接
    def angle_sub(match: re.Match) -> str:
        url = match.group(1)
        return _stash(
            store,
            f'<a href="{html.escape(url, quote=True)}">{html.escape(url)}</a>',
        )

    text = ANGLE_AUTOLINK_RE.sub(angle_sub, text)

    # 裸露的 URL 与邮箱
    def url_sub(match: re.Match) -> str:
        url = match.group(0)
        trailing = ""
        # 句末标点不属于链接；成对括号之外的右括号也不属于
        while url and url[-1] in ".,;:!?":
            trailing = url[-1] + trailing
            url = url[:-1]
        while url.endswith(")") and url.count(")") > url.count("("):
            trailing = ")" + trailing
            url = url[:-1]
        return _stash(
            store,
            f'<a href="{html.escape(url, quote=True)}">{html.escape(url)}</a>{trailing}',
        )

    text = AUTOLINK_RE.sub(url_sub, text)

    def mail_sub(match: re.Match) -> str:
        address = match.group(0)
        return _stash(
            store,
            f'<a href="mailto:{html.escape(address, quote=True)}">'
            f"{html.escape(address)}</a>",
        )

    text = EMAIL_RE.sub(mail_sub, text)

    # 剩下的普通文本转义
    text = html.escape(text)

    # 强调
    text = STRIKE_RE.sub(r"<del>\1</del>", text)
    text = BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = ITALIC_RE.sub(r"<em>\1</em>", text)

    return _restore(text, store)


# --------------------------------------------------------------------------
# 块级
# --------------------------------------------------------------------------


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    return all(
        part.strip().replace(":", "").replace("-", "") == ""
        for part in stripped.strip("|").split("|")
    )


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _starts_block(lines: list[str], index: int) -> bool:
    """判断该行是否是一个新块的开头（用于终止段落）。"""
    line = lines[index]
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return True
    if re.match(r"^#{1,6}\s", stripped):
        return True
    if stripped.startswith(">"):
        return True
    if re.match(r"^[-*+]\s", stripped):
        return True
    if re.match(r"^\d+[.)]\s", stripped):
        return True
    if line.startswith("    ") and line.strip():
        return True
    if HR_RE.match(line):
        return True
    if _is_setext_underline(stripped):
        return True
    if index + 1 < len(lines) and "|" in stripped and _is_table_separator(lines[index + 1]):
        return True
    return False


def _is_setext_underline(stripped: str) -> bool:
    """是否是 setext 标题的下划线（=== 或 ---）。"""
    if not stripped:
        return False
    if set(stripped) == {"="}:
        return True
    # 单个 "-" 可能是空列表项，因此要求至少两个
    return set(stripped) == {"-"} and len(stripped) >= 2


def _render_table(
    lines: list[str], index: int, references: dict[str, str]
) -> tuple[str, int]:
    """渲染 GitHub 风格表格，返回 (html, 下一行下标)。"""
    header = _split_table_row(lines[index])
    index += 2  # 跳过表头与分隔行
    rows: list[list[str]] = []
    while index < len(lines) and "|" in lines[index] and lines[index].strip():
        rows.append(_split_table_row(lines[index]))
        index += 1

    out = ['<div class="table-wrap"><table class="md-table">']
    out.append("<thead><tr>")
    for cell in header:
        out.append(f"<th>{render_inline(cell, references)}</th>")
    out.append("</tr></thead><tbody>")
    width = len(header)
    for row in rows:
        out.append("<tr>")
        # 补齐缺失的单元格，避免表格错位
        cells = list(row[:width]) + [""] * max(0, width - len(row))
        for cell in cells:
            out.append(f"<td>{render_inline(cell, references)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out), index


def _dedent(lines: list[str]) -> list[str]:
    """去掉一组行的公共缩进。"""
    indents = [_indent_of(line) for line in lines if line.strip()]
    if not indents:
        return list(lines)
    common = min(indents)
    return [line[common:] if line.strip() else "" for line in lines]


def _render_list(
    lines: list[str],
    index: int,
    toc: list[dict],
    used: set[str],
    references: dict[str, str],
) -> tuple[str, int]:
    """渲染有序 / 无序列表（支持嵌套列表与任务列表），返回 (html, 下一行下标)。"""
    match = LIST_ITEM_RE.match(lines[index])
    if not match:
        return "", index
    ordered = bool(re.match(r"^\d+[.)]$", match.group("bullet")))
    base_indent = _indent_of(lines[index])

    items: list[list[str]] = []  # 每个列表项的内容行
    current: list[str] = []
    i = index

    def is_same_kind(line: str) -> bool:
        item = LIST_ITEM_RE.match(line)
        if not item:
            return False
        if _indent_of(line) > base_indent:
            return False
        return bool(re.match(r"^\d+[.)]$", item.group("bullet"))) == ordered

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            # 空行：其后若仍是本列表则继续，否则列表结束
            look = i + 1
            while look < len(lines) and not lines[look].strip():
                look += 1
            if look < len(lines):
                item = LIST_ITEM_RE.match(lines[look])
                if item and _indent_of(lines[look]) <= base_indent and is_same_kind(lines[look]):
                    current.append("")
                    i = look
                    continue
                if item and _indent_of(lines[look]) > base_indent:
                    current.append("")
                    i = look
                    continue
            break

        item = LIST_ITEM_RE.match(line)

        if is_same_kind(line):
            if current:
                items.append(current)
            current = [(item.group("text") or "") if item else ""]
            i += 1
            continue

        if HR_RE.match(line) and _indent_of(line) <= base_indent:
            # 分隔线结束当前列表
            break

        if item and _indent_of(line) > base_indent:
            # 嵌套列表：整块收进当前项
            if current:
                current.append(line)
            i += 1
            continue

        if item and _indent_of(line) <= base_indent:
            # 同级的另一种列表（例如无序列表之后紧跟有序列表）：另起一个列表
            break

        if current:
            current.append(line.strip())  # 懒续行
            i += 1
            continue

        break

    if current:
        items.append(current)

    tag = "ol" if ordered else "ul"
    out = [f"<{tag}>"]
    for item_lines in items:
        content = _dedent(item_lines)
        text = content[0] if content else ""
        # 嵌套内容按它自己的缩进单独去掉一层，
        # 否则更深一层的列表会被当成缩进代码块
        inner = _dedent(content[1:]) if len(content) > 1 else []
        checkbox = ""
        task = re.match(r"^\[([ xX])\]\s+(.*)$", text)
        if task:
            checked = " checked" if task.group(1).lower() == "x" else ""
            checkbox = f'<input type="checkbox" disabled{checked}> '
            text = task.group(2)
        body = checkbox + render_inline(text, references)
        if any(line.strip() for line in inner):
            nested = _render("\n".join(inner), toc, used, references)
            body += nested
        out.append(f"<li>{body}</li>")
    out.append(f"</{tag}>")
    return "\n".join(out), i


def render(markdown: str) -> str:
    """把 Markdown 全文转成 HTML。"""
    html_text, _ = render_document(markdown)
    return html_text


def _slug(text: str, used: set[str]) -> str:
    """由标题文字生成唯一锚点。"""
    base = re.sub(r"[^\w一-鿿]+", "-", text.strip().lower()).strip("-")
    if not base:
        base = "section"
    slug = base
    counter = 2
    while slug in used:
        slug = f"{base}-{counter}"
        counter += 1
    used.add(slug)
    return slug


def render_document(markdown: str) -> tuple[str, list[dict]]:
    """渲染 Markdown，同时返回目录（标题级别、标题文字、锚点）。

    目录用于在宽屏上给正文旁边配一个可跳转的侧栏。
    """
    toc: list[dict] = []
    used: set[str] = set()
    references = _collect_references(markdown)
    html_text = _render(markdown, toc, used, references)
    return html_text, toc


def _collect_references(markdown: str) -> dict[str, str]:
    """收集全文里的链接引用定义 [ref]: url。"""
    references: dict[str, str] = {}
    in_fence = False
    for line in markdown.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = REF_DEF_RE.match(line)
        if match:
            references[match.group(1).strip().lower()] = match.group(2)
    return references


def _render(
    markdown: str,
    toc: list[dict],
    used: set[str],
    references: dict[str, str] | None = None,
) -> str:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if references is None:
        references = _collect_references(markdown)
    out: list[str] = []
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        # 围栏代码块
        fence = re.match(r"^(`{3,}|~{3,})\s*(\S*)\s*$", stripped)
        if fence:
            marker = fence.group(1)[0]
            language = fence.group(2)
            index += 1
            code: list[str] = []
            while index < total:
                closing = lines[index].strip()
                if closing.startswith(marker * 3) and set(closing) <= {marker, " "}:
                    index += 1
                    break
                code.append(lines[index])
                index += 1
            class_attr = f' class="language-{html.escape(language)}"' if language else ""
            body = html.escape("\n".join(code))
            out.append(f"<pre class=\"code\"><code{class_attr}>{body}</code></pre>")
            continue

        # 缩进代码块
        if line.startswith("    ") and line.strip():
            code = []
            while index < total and (
                lines[index].startswith("    ") or not lines[index].strip()
            ):
                if not lines[index].strip() and (
                    index + 1 >= total or not lines[index + 1].startswith("    ")
                ):
                    break
                code.append(lines[index][4:])
                index += 1
            out.append(
                '<pre class="code"><code>' + html.escape("\n".join(code)) + "</code></pre>"
            )
            continue

        # 链接引用定义：单独成行，不渲染
        if REF_DEF_RE.match(line):
            index += 1
            continue

        # ATX 标题
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).rstrip("#").strip()
            anchor = _slug(re.sub(r"`|\*\*|\*", "", text), used)
            if level in (2, 3):
                toc.append({"level": level, "text": re.sub(r"`|\*\*|\*", "", text), "id": anchor})
            out.append(
                f'<h{level} id="{html.escape(anchor, quote=True)}">'
                f"{render_inline(text, references)}</h{level}>"
            )
            index += 1
            continue

        # 分隔线
        if HR_RE.match(line):
            out.append("<hr>")
            index += 1
            continue

        # 表格
        if "|" in stripped and index + 1 < total and _is_table_separator(lines[index + 1]):
            table_html, index = _render_table(lines, index, references)
            out.append(table_html)
            continue

        # 引用块
        if stripped.startswith(">"):
            quoted = []
            while index < total and lines[index].strip().startswith(">"):
                quoted.append(lines[index].strip().lstrip(">").strip())
                index += 1
            out.append(
                "<blockquote>"
                + _render("\n".join(quoted), toc, used, references)
                + "</blockquote>"
            )
            continue

        # 列表
        if LIST_ITEM_RE.match(stripped):
            list_html, index = _render_list(lines, index, toc, used, references)
            out.append(list_html)
            continue

        # 段落（若整段之后紧跟 === 或 ---，其实是 setext 标题）
        parts: list[str] = []
        while index < total and not _starts_block(lines, index):
            parts.append(lines[index].strip())
            index += 1
        text = " ".join(parts)
        underline = lines[index].strip() if index < total else ""
        if parts and _is_setext_underline(underline):
            level = 1 if set(underline) == {"="} else 2
            anchor = _slug(re.sub(r"`|\*\*|\*", "", text), used)
            if level in (2, 3):
                toc.append({"level": level, "text": re.sub(r"`|\*\*|\*", "", text), "id": anchor})
            out.append(
                f'<h{level} id="{html.escape(anchor, quote=True)}">'
                f"{render_inline(text, references)}</h{level}>"
            )
            index += 1
            continue
        out.append("<p>" + render_inline(text, references) + "</p>")

    return "\n".join(out)
