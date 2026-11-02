"""读取 tutorials/ 目录，为网站提供「教程」页的内容。

教程内容不手工维护：分组、标题、说明、依赖都从 tutorials/ 下的脚本
与其 README.md 里读出来。仓库里新增一个示例脚本，网站自动跟着更新。

信息来自两处，互相补充：

  * 每个脚本开头的注释——标题、来源链接、一句话说明
  * tutorials/README.md 里的表格——更详细的说明与依赖说明

两处都没有时，退回到按文件名与 import 推断，保证页面不为空。
"""

from __future__ import annotations

import re
from pathlib import Path

# 「# 来源 / Source: https://...」
SOURCE_RE = re.compile(r"来源\s*/\s*Source\s*[:：]\s*(\S+)")
# README 表格里的一行：| `01_ai_chat.py` | 说明 | 依赖 |
TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")
# 脚本里的 import，用来猜依赖
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

# 根目录下的脚本（不在子目录里）单独归为一组
ROOT_GROUP_TITLE = "开始使用"
ROOT_GROUP_SUBTITLE = "Getting started"


def _read_text(path: Path) -> str:
    """读文本文件，编码不对时退回 latin-1，绝不抛异常。"""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        try:
            return path.read_text(encoding="gbk")
        except (OSError, UnicodeDecodeError):
            return ""


def _split_heading(heading: str) -> tuple[str, str]:
    """把「一、日常场景 / Daily scenarios」拆成中文标题与英文副标题。"""
    parts = re.split(r"\s*/\s*", heading.strip(), maxsplit=1)
    title = parts[0].strip()
    subtitle = parts[1].strip() if len(parts) > 1 else ""
    return title, subtitle


def parse_readme(readme_path: Path) -> dict:
    """解析 tutorials/README.md。

    返回：
      files     文件名 -> {desc, req, section}
      sections  [(标题, 副标题, [文件名...])]
    """
    files: dict[str, dict] = {}
    sections: list[tuple[str, str, list[str]]] = []
    if not readme_path.exists():
        return {"files": files, "sections": sections}

    section_title = ""
    section_subtitle = ""
    section_files: list[str] = []

    def flush() -> None:
        if section_title or section_files:
            sections.append((section_title, section_subtitle, list(section_files)))

    for line in _read_text(readme_path).splitlines():
        if line.startswith("## "):
            flush()
            section_title, section_subtitle = _split_heading(line[3:])
            section_files = []
            continue
        match = TABLE_ROW_RE.match(line.strip())
        if not match:
            continue
        name, desc, req = match.groups()
        files[name] = {
            "desc": desc.strip(),
            "req": req.strip(),
            "section": section_title,
        }
        section_files.append(name)
    flush()
    return {"files": files, "sections": sections}


def parse_script(path: Path) -> dict:
    """读脚本开头的注释块。

    约定：文件开头连续的注释里，第一行是标题，含「来源 / Source」的行是出处，
    其余行是补充说明。注释块在遇到第一行代码时结束。
    """
    title = ""
    source = ""
    notes: list[str] = []

    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        text = stripped.lstrip("#").strip()
        if not text:
            continue
        match = SOURCE_RE.search(text)
        if match:
            source = match.group(1)
            continue
        if not title:
            title = text
            continue
        notes.append(text)
    return {"title": title, "source": source, "notes": notes}


def detect_requirements(code: str) -> list[str]:
    """从 import 语句推断依赖（README 里没写时用）。"""
    names = []
    for name in IMPORT_RE.findall(code):
        if name in ("guan", "os", "sys", "time", "json", "re", "math", "cmath", "random"):
            continue
        if name not in names:
            names.append(name)
    return names


def _short_title(description: str) -> str:
    """从一段说明里取出适合当标题的短句。

    README 的说明常常是「try except 简写：`@guan.try_decorator`」这样，
    标题取第一个短句就够了，剩下的留给说明。
    """
    text = description.strip()
    if not text:
        return ""
    head = re.split(r"[：:，,。;；]", text, maxsplit=1)[0].strip()
    head = head.replace("`", "").strip()
    return head or text.replace("`", "").strip()


def _requirements_from_cell(cell: str) -> list[str]:
    """把 README 表格里的依赖单元格拆成列表。

    只按逗号、顿号、分号拆——单元格里的斜杠常常是路径或模型名的一部分，
    拆开反而看不懂。
    """
    if not cell or cell.strip() in {"-", "—", "无", "N/A"}:
        return []
    parts = re.split(r"[,，、;；]", cell)
    return [part.strip(" `·") for part in parts if part.strip(" `·")]


def collect(tutorials_dir: Path) -> dict:
    """收集 tutorials 目录下的全部示例脚本。

    返回值可以直接放进 api.json，也可以直接渲染成页面。
    """
    tutorials_dir = Path(tutorials_dir).resolve()
    if not tutorials_dir.is_dir():
        return {"dir": tutorials_dir.name, "readme": "", "groups": [], "files": [], "count": 0}

    readme = tutorials_dir / "README.md"
    readme_info = parse_readme(readme)

    scripts = sorted(
        path for path in tutorials_dir.rglob("*.py") if path.is_file()
    )

    groups: dict[str, dict] = {}
    files: list[dict] = []

    for path in scripts:
        relative = path.relative_to(tutorials_dir)
        group_key = relative.parts[0] if len(relative.parts) > 1 else ""
        info = parse_script(path)
        code = _read_text(path)
        entry = readme_info["files"].get(path.name, {})

        # 标题与说明都来自脚本自身：首行注释是标题，其余注释是说明。
        # README 的表格只用来补标题与依赖，不再重复它写的那段说明。
        title = info["title"] or _short_title(entry.get("desc", "")) or relative.name
        description = "；".join(info["notes"])

        requirements = _requirements_from_cell(entry.get("req", ""))
        if not requirements:
            requirements = detect_requirements(code)

        # 链接用相对仓库根目录的路径，便于站点里直接下载脚本
        repo_relative = (Path(tutorials_dir.name) / relative).as_posix()
        anchor = f"{group_key or 'root'}-{relative.stem}".replace("_", "-")

        item = {
            "key": relative.as_posix(),
            "path": repo_relative,
            "group": group_key,
            "title": title,
            "description": description,
            "requirements": requirements,
            "source": info["source"],
            "anchor": anchor,
            "lines": len(code.splitlines()),
            "code": code,
        }
        files.append(item)

        group = groups.setdefault(
            group_key,
            {
                "key": group_key,
                "title": ROOT_GROUP_TITLE if group_key == "" else group_key.replace("_", " "),
                "subtitle": ROOT_GROUP_SUBTITLE if group_key == "" else "",
                "anchor": f"group-{group_key or 'root'}",
                "files": [],
            },
        )
        group["files"].append(item)

    # 分组标题优先用 README 里的章节标题
    for key, group in groups.items():
        if not key:
            continue
        best: tuple[str, str, int] | None = None
        for title, subtitle, names in readme_info["sections"]:
            overlap = sum(
                1 for item in group["files"] if Path(item["key"]).name in names
            )
            if overlap and (best is None or overlap > best[2]):
                best = (title, subtitle, overlap)
        if best:
            group["title"], group["subtitle"] = best[0], best[1]

    ordered = sorted(
        groups.values(),
        key=lambda group: (group["key"] != "", group["title"]),
    )

    return {
        "dir": tutorials_dir.name,
        "readme": (tutorials_dir / "README.md").name if readme.exists() else "",
        "groups": ordered,
        "files": files,
        "count": len(files),
    }


def without_code(data: dict) -> dict:
    """去掉源码的副本，用于放进 api.json（索引里用不着整段代码）。"""
    return {
        "dir": data.get("dir", ""),
        "count": data.get("count", 0),
        "groups": [
            {
                "key": group["key"],
                "title": group["title"],
                "subtitle": group.get("subtitle", ""),
                "anchor": group["anchor"],
                "files": [
                    {key: value for key, value in item.items() if key != "code"}
                    for item in group["files"]
                ],
            }
            for group in data.get("groups", [])
        ],
        "files": [
            {key: value for key, value in item.items() if key != "code"}
            for item in data.get("files", [])
        ],
    }
