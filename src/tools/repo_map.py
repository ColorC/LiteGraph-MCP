# -*- coding: utf-8 -*-
"""
Repo Map 生成器 (类似 Aider)

使用 Tree-sitter 解析代码文件，提取类名、方法签名等结构信息。
返回精简的代码结构视图，供 Agent 快速理解代码组织。

支持语言:
- C# (.cs)
- Lua (.lua)
- Python (.py)

支持远程文件访问:
- 通过 Windows 文件桥接服务访问 Git 文件
"""

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

# 尝试导入 tree_sitter 和相关语言包
HAS_TREE_SITTER = False
_TS_LANGUAGES = {}

try:
    from tree_sitter import Language, Parser
    from grep_ast import filename_to_lang

    # 加载语言包
    try:
        import tree_sitter_lua as _tslua
        _TS_LANGUAGES['lua'] = Language(_tslua.language())
    except ImportError:
        pass
    try:
        import tree_sitter_c_sharp as _tscs
        _TS_LANGUAGES['csharp'] = Language(_tscs.language())
        _TS_LANGUAGES['c#'] = _TS_LANGUAGES['csharp']
    except ImportError:
        pass
    try:
        import tree_sitter_python as _tspy
        _TS_LANGUAGES['python'] = Language(_tspy.language())
    except ImportError:
        pass

    if _TS_LANGUAGES:
        HAS_TREE_SITTER = True
        logger.info(f"tree-sitter 可用，支持语言: {list(_TS_LANGUAGES.keys())}")
    else:
        logger.warning("tree-sitter 已安装但无语言包")
except ImportError:
    logger.warning("tree-sitter 或 grep_ast 未安装，Repo Map 将使用正则备用方案")


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class CodeTag:
    """代码标签"""
    file_path: str        # 文件相对路径
    line_number: int      # 行号
    name: str             # 名称 (类名、方法名等)
    kind: str             # 类型 (class, method, function 等)
    signature: str = ""   # 签名 (参数列表等)
    return_type: str = ""
    parameters: List[Dict[str, str]] = field(default_factory=list)
    parent_symbol: str = ""
    access_modifier: str = ""


@dataclass
class FileSignature:
    """文件签名"""
    file_path: str
    language: str
    classes: List[CodeTag] = field(default_factory=list)
    functions: List[CodeTag] = field(default_factory=list)
    methods: List[CodeTag] = field(default_factory=list)

    def _format_signature(self, tag: CodeTag, include_signature_details: bool = True) -> str:
        if include_signature_details and tag.signature:
            return tag.signature

        if tag.signature:
            sig = tag.signature
            if ':' in sig:
                sig = sig.split(':', 1)[1]
            elif '.' in sig and not sig.startswith('def '):
                sig = sig.rsplit('.', 1)[1]
            return sig

        return f"{tag.name}()"

    def _symbol_relevance(self, query: str, tag: CodeTag) -> float:
        if not query:
            return 1.0
        q_tokens = [t for t in re.split(r"[^\w\u4e00-\u9fff]+", query.lower()) if t]
        if not q_tokens:
            return 1.0

        haystack = " ".join([
            tag.name.lower(),
            tag.signature.lower(),
            tag.parent_symbol.lower(),
            tag.return_type.lower(),
            " ".join((p.get("name", "") + " " + p.get("type", "")).lower() for p in tag.parameters),
        ])
        hits = sum(1 for t in q_tokens if t in haystack)
        return hits / max(1, len(q_tokens))

    def format(
        self,
        include_classes: bool = True,
        include_signature_details: bool = True,
        max_methods_per_class: int = 10,
        max_group_methods: int = 15,
        max_functions: int = 20,
        max_standalone_methods: int = 10,
        max_symbols_per_file: int = 30,
        query: str = "",
        order: str = "top_down",
    ) -> str:
        """格式化为 Repo Map 输出"""
        lines = [f"{self.file_path}:"]

        classes = sorted(self.classes, key=lambda t: t.line_number)
        functions = sorted(self.functions, key=lambda t: t.line_number)
        methods = sorted(self.methods, key=lambda t: t.line_number)

        def _rerank_and_clip(symbols: List[CodeTag]) -> List[CodeTag]:
            if not symbols:
                return symbols
            if not query:
                return symbols[:max_symbols_per_file] if max_symbols_per_file > 0 else symbols

            ranked = sorted(
                symbols,
                key=lambda t: (-self._symbol_relevance(query, t), t.line_number)
            )
            if max_symbols_per_file > 0:
                ranked = ranked[:max_symbols_per_file]

            if order == "top_down":
                return sorted(ranked, key=lambda t: t.line_number)
            return ranked

        if query and order == "score_only":
            classes = sorted(classes, key=lambda t: (-self._symbol_relevance(query, t), t.line_number))

        methods = _rerank_and_clip(methods)
        functions = _rerank_and_clip(functions)

        grouped_methods: Dict[str, List[CodeTag]] = {}
        standalone_methods = []
        for m in methods:
            if m.parent_symbol:
                grouped_methods.setdefault(m.parent_symbol, []).append(m)
            elif ':' in m.name:
                cls_name = m.name.split(':', 1)[0]
                grouped_methods.setdefault(cls_name, []).append(m)
            elif '.' in m.name:
                cls_name = m.name.rsplit('.', 1)[0]
                grouped_methods.setdefault(cls_name, []).append(m)
            else:
                standalone_methods.append(m)

        if include_classes and classes:
            for cls in classes:
                lines.append(f"  class {cls.name}")
                cls_methods = grouped_methods.get(cls.name, [])
                if not cls_methods:
                    cls_methods = [m for m in methods if m.line_number > cls.line_number][:max_methods_per_class]
                for method in cls_methods[:max_methods_per_class]:
                    lines.append(f"    + {self._format_signature(method, include_signature_details)}")
                if len(cls_methods) > max_methods_per_class:
                    lines.append(f"    ... (+{len(cls_methods) - max_methods_per_class} more)")

        if grouped_methods and not classes:
            for cls_name, cls_methods in sorted(grouped_methods.items(), key=lambda kv: kv[0]):
                lines.append(f"  class {cls_name}")
                for m in cls_methods[:max_group_methods]:
                    lines.append(f"    + {self._format_signature(m, include_signature_details)}")
                if len(cls_methods) > max_group_methods:
                    lines.append(f"    ... (+{len(cls_methods) - max_group_methods} more)")

        if functions:
            for func in functions[:max_functions]:
                lines.append(f"  fn {self._format_signature(func, include_signature_details)}")
            if len(functions) > max_functions:
                lines.append(f"  ... (+{len(functions) - max_functions} more functions)")

        if standalone_methods and not classes:
            for m in standalone_methods[:max_standalone_methods]:
                lines.append(f"  + {self._format_signature(m, include_signature_details)}")

        return "\n".join(lines)


@dataclass
class RepoMapResult:
    """Repo Map 结果"""
    files: List[FileSignature] = field(default_factory=list)
    total_classes: int = 0
    total_functions: int = 0
    total_methods: int = 0

    def format(self) -> str:
        """格式化为输出字符串"""
        if not self.files:
            return "No code structure found."

        parts = []
        for file_sig in self.files:
            parts.append(file_sig.format())
            parts.append("")  # 空行分隔

        return "\n".join(parts)


# =============================================================================
# RepoMapGenerator
# =============================================================================

class RepoMapGenerator:
    """
    Repo Map 生成器

    使用 Tree-sitter 或正则表达式提取代码结构
    支持本地文件和远程文件（通过 Windows 桥接服务）
    """

    def __init__(
        self,
        root_path: str,
        use_cache: bool = True,
        cache_db_path: Optional[str] = None,
        remote_bridge_url: Optional[str] = None
    ):
        """初始化生成器

        Args:
            root_path: 代码根目录
            use_cache: 是否使用缓存
            cache_db_path: 缓存数据库路径
            remote_bridge_url: Windows 桥接服务 URL (e.g. http://localhost:8765)
        """
        self.root = Path(root_path)
        self.use_cache = use_cache
        self.remote_bridge_url = remote_bridge_url

        # 初始化缓存
        if use_cache and cache_db_path:
            self.cache_db_path = Path(cache_db_path)
            self._init_cache()
        else:
            self.cache_db_path = None
            self._cache = {}

        # 初始化 Tree-sitter
        if HAS_TREE_SITTER:
            self._init_tree_sitter()
        else:
            self.parser = None

    def _init_cache(self):
        """初始化 SQLite 缓存"""
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_signatures (
                file_path TEXT PRIMARY KEY,
                mtime REAL,
                signature_json TEXT
            )
        """)
        conn.commit()
        conn.close()
        self._db_conn = conn

    def _init_tree_sitter(self):
        """初始化 Tree-sitter 解析器"""
        self.parsers = {}
        for lang_name, lang_obj in _TS_LANGUAGES.items():
            try:
                self.parsers[lang_name] = Parser(lang_obj)
            except Exception as e:
                logger.warning(f"Tree-sitter {lang_name} 初始化失败：{e}")

    def _get_file_mtime(self, file_path: Path) -> float:
        """获取文件修改时间"""
        try:
            return file_path.stat().st_mtime
        except FileNotFoundError:
            return 0

    def _get_cached_signature(self, file_path: str) -> Optional[FileSignature]:
        """从缓存获取签名"""
        if not self.use_cache:
            return None

        if self.cache_db_path:
            cursor = self._db_conn.cursor()
            cursor.execute(
                "SELECT signature_json FROM file_signatures WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return self._deserialize_signature(data)
        else:
            return self._cache.get(file_path)

        return None

    def _cache_signature(self, file_path: str, signature: FileSignature):
        """缓存签名"""
        if not self.use_cache:
            return

        if self.cache_db_path:
            cursor = self._db_conn.cursor()
            data = self._serialize_signature(signature)
            cursor.execute(
                """INSERT OR REPLACE INTO file_signatures
                   (file_path, mtime, signature_json) VALUES (?, ?, ?)""",
                (file_path, self._get_file_mtime(Path(file_path)), json.dumps(data))
            )
            self._db_conn.commit()
        else:
            self._cache[file_path] = signature

    def _serialize_signature(self, sig: FileSignature) -> dict:
        """序列化签名为 dict"""
        def _tag_to_dict(t: CodeTag) -> Dict[str, Any]:
            return {
                "file_path": t.file_path,
                "line_number": t.line_number,
                "name": t.name,
                "kind": t.kind,
                "signature": t.signature,
                "return_type": t.return_type,
                "parameters": t.parameters,
                "parent_symbol": t.parent_symbol,
                "access_modifier": t.access_modifier,
            }

        return {
            "file_path": sig.file_path,
            "language": sig.language,
            "classes": [_tag_to_dict(t) for t in sig.classes],
            "functions": [_tag_to_dict(t) for t in sig.functions],
            "methods": [_tag_to_dict(t) for t in sig.methods],
        }

    def _deserialize_signature(self, data: dict) -> FileSignature:
        """从 dict 反序列化签名"""
        sig = FileSignature(
            file_path=data["file_path"],
            language=data["language"]
        )
        for t in data.get("classes", []):
            sig.classes.append(CodeTag(**t))
        for t in data.get("functions", []):
            sig.functions.append(CodeTag(**t))
        for t in data.get("methods", []):
            sig.methods.append(CodeTag(**t))
        return sig

    @staticmethod
    def _split_params(raw_params: str) -> List[str]:
        params = []
        cur = []
        depth = 0
        for ch in raw_params:
            if ch in "([<":
                depth += 1
            elif ch in ")]>":
                depth = max(0, depth - 1)
            if ch == ',' and depth == 0:
                p = "".join(cur).strip()
                if p:
                    params.append(p)
                cur = []
            else:
                cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            params.append(tail)
        return params

    @classmethod
    def _parse_parameters(cls, params_text: str) -> List[Dict[str, str]]:
        text = (params_text or "").strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        if not text:
            return []

        parsed = []
        for token in cls._split_params(text):
            part = token.strip()
            if not part:
                continue

            default = ""
            if "=" in part:
                left, default = part.split("=", 1)
                part = left.strip()
                default = default.strip()

            # Python style: *args, **kwargs, name: Type
            py_style = re.match(r'^(?P<name>\*{0,2}[A-Za-z_]\w*)\s*:\s*(?P<ptype>.+)$', part)
            if py_style:
                name = py_style.group("name").strip()
                ptype = py_style.group("ptype").strip()
                parsed.append({"name": name, "type": ptype, "default": default})
                continue

            # C#/Lua/general fallback: [modifiers/type] name
            pieces = part.split()
            if len(pieces) >= 2:
                name = pieces[-1].strip()
                ptype = " ".join(pieces[:-1]).strip()
            else:
                name = part.strip()
                ptype = ""

            parsed.append({"name": name, "type": ptype, "default": default})

        return parsed

    def _extract_tags_tree_sitter(self, file_path: Path, lang: str) -> List[CodeTag]:
        """使用 Tree-sitter AST 遍历提取标签"""
        if not HAS_TREE_SITTER:
            return []

        parser = self.parsers.get(lang.lower())
        if not parser:
            return []

        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        return self._extract_tags_from_code(str(file_path), lang, code)

    def _extract_tags_from_code(self, file_path_str: str, lang: str, code: str) -> List[CodeTag]:
        """从代码字符串中使用 tree-sitter AST 提取标签"""
        parser = self.parsers.get(lang.lower())
        if not parser:
            return []

        tree = parser.parse(bytes(code, "utf-8"))
        tags = []

        if lang.lower() in ("lua",):
            self._walk_lua(tree.root_node, file_path_str, tags)
        elif lang.lower() in ("c#", "csharp"):
            self._walk_csharp(tree.root_node, file_path_str, tags)
        elif lang.lower() == "python":
            self._walk_python(tree.root_node, file_path_str, tags)

        return tags

    def _walk_lua(self, node, file_path: str, tags: List[CodeTag]):
        """遍历 Lua AST"""
        if node.type == "function_declaration":
            name_node = None
            params_node = None
            is_local = False
            for child in node.children:
                if child.type == "local":
                    is_local = True
                elif child.type in ("method_index_expression", "dot_index_expression"):
                    name_node = child
                elif child.type == "identifier":
                    name_node = child
                elif child.type == "parameters":
                    params_node = child

            if name_node:
                name = name_node.text.decode("utf-8")
                params = params_node.text.decode("utf-8") if params_node else "()"
                line = name_node.start_point[0] + 1

                if ':' in name or '.' in name:
                    kind = "method"
                else:
                    kind = "function"

                tags.append(CodeTag(
                    file_path=file_path,
                    line_number=line,
                    name=name,
                    kind=kind,
                    signature=f"{name}{params}",
                    parameters=self._parse_parameters(params),
                    parent_symbol=name.split(":", 1)[0] if ":" in name else (name.rsplit(".", 1)[0] if "." in name else ""),
                ))
            return  # 不递归进函数体

        for child in node.children:
            self._walk_lua(child, file_path, tags)

    def _walk_csharp(self, node, file_path: str, tags: List[CodeTag], current_class: str = ""):
        """遍历 C# AST"""
        if node.type == "class_declaration":
            name = ""
            base = ""
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8")
                elif child.type == "base_list":
                    base = child.text.decode("utf-8").strip(": ").strip()

            sig = f"class {name}"
            if base:
                sig += f" : {base}"

            tags.append(CodeTag(
                file_path=file_path,
                line_number=node.start_point[0] + 1,
                name=name,
                kind="class",
                signature=sig,
                parent_symbol=name,
            ))

            # 递归进 declaration_list
            for child in node.children:
                if child.type == "declaration_list":
                    self._walk_csharp(child, file_path, tags, current_class=name)
            return

        if node.type == "interface_declaration":
            name = ""
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8")
            tags.append(CodeTag(
                file_path=file_path,
                line_number=node.start_point[0] + 1,
                name=name,
                kind="class",
                signature=f"interface {name}",
                parent_symbol=name,
            ))
            for child in node.children:
                if child.type == "declaration_list":
                    self._walk_csharp(child, file_path, tags, current_class=name)
            return

        if node.type == "method_declaration":
            name = ""
            ret_type = ""
            params = ""
            modifiers = []
            for child in node.children:
                if child.type == "modifier":
                    modifiers.append(child.text.decode("utf-8"))
                elif child.type == "identifier":
                    name = child.text.decode("utf-8")
                elif child.type in ("predefined_type", "generic_name", "nullable_type",
                                     "array_type", "identifier"):
                    if not name:  # 返回类型在 name 之前
                        ret_type = child.text.decode("utf-8")
                elif child.type == "parameter_list":
                    params = child.text.decode("utf-8")

            # 如果 ret_type 没拿到，从第一行提取
            if not ret_type:
                first_line = node.text.decode("utf-8").split("\n")[0]
                ret_type = first_line.split(name)[0].strip().split()[-1] if name in first_line else ""

            sig = f"{ret_type} {name}{params}".strip()
            tags.append(CodeTag(
                file_path=file_path,
                line_number=node.start_point[0] + 1,
                name=name,
                kind="method",
                signature=sig,
                return_type=ret_type,
                parameters=self._parse_parameters(params),
                parent_symbol=current_class,
                access_modifier=" ".join(modifiers),
            ))
            return

        if node.type == "property_declaration":
            name = ""
            prop_type = ""
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8")
                elif child.type in ("predefined_type", "generic_name", "nullable_type", "array_type"):
                    prop_type = child.text.decode("utf-8")
            if name:
                tags.append(CodeTag(
                    file_path=file_path,
                    line_number=node.start_point[0] + 1,
                    name=name,
                    kind="method",  # 归类为 method 以便显示
                    signature=f"{prop_type} {name} {{ get; set; }}",
                    return_type=prop_type,
                    parent_symbol=current_class,
                ))
            return

        for child in node.children:
            self._walk_csharp(child, file_path, tags, current_class)

    def _walk_python(self, node, file_path: str, tags: List[CodeTag], depth: int = 0, current_class: str = ""):
        """遍历 Python AST"""
        if node.type == "class_definition":
            name = ""
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8")
                    break
            tags.append(CodeTag(
                file_path=file_path,
                line_number=node.start_point[0] + 1,
                name=name,
                kind="class",
                signature=node.text.decode("utf-8").split("\n")[0][:100],
                parent_symbol=name,
            ))
            # 递归进 body
            for child in node.children:
                if child.type == "block":
                    self._walk_python(child, file_path, tags, depth + 1, current_class=name)
            return

        if node.type == "function_definition":
            name = ""
            params = ""
            ret = ""
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8")
                elif child.type == "parameters":
                    params = child.text.decode("utf-8")
                elif child.type == "type":
                    ret = child.text.decode("utf-8")

            sig = f"{name}{params}"
            if ret:
                sig += f" -> {ret}"

            kind = "method" if depth > 0 else "function"
            tags.append(CodeTag(
                file_path=file_path,
                line_number=node.start_point[0] + 1,
                name=name,
                kind=kind,
                signature=sig,
                return_type=ret,
                parameters=self._parse_parameters(params),
                parent_symbol=current_class if kind == "method" else "",
            ))
            return  # 不递归进函数体

        for child in node.children:
            self._walk_python(child, file_path, tags, depth, current_class=current_class)

    def _extract_tags_regex_from_lines(self, file_path_str: str, lang: str, lines: List[str]) -> List[CodeTag]:
        """从已有的行列表中使用正则提取标签（供远程文件使用）"""
        return self._do_extract_tags_regex(file_path_str, lang, lines)

    def _extract_tags_regex(self, file_path: Path, lang: str) -> List[CodeTag]:
        """使用正则表达式提取标签 (备用方案)"""
        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        lines = code.split("\n")
        return self._do_extract_tags_regex(str(file_path), lang, lines)

    def _do_extract_tags_regex(self, file_path_str: str, lang: str, lines: List[str]) -> List[CodeTag]:
        """正则提取标签的核心实现"""
        tags: List[CodeTag] = []
        current_class = ""
        class_indent = -1

        if lang in ("c#", "csharp"):
            class_pattern = re.compile(
                r'^\s*(?P<mods>(?:(?:public|private|protected|internal|static|sealed|abstract|partial|new)\s+)*)'
                r'(?P<kind>class|interface)\s+(?P<name>[A-Za-z_]\w*)'
                r'(?P<tail>\s*:\s*[^\{]+)?'
            )
            method_pattern = re.compile(
                r'^\s*(?P<mods>(?:(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed|new|extern|unsafe|partial)\s+)*)'
                r'(?P<ret>[A-Za-z_][\w<>,\.\?\[\]\s]*)\s+'
                r'(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)'
                r'\s*(?:where\s+.+)?\s*(?:\{|=>|;)$'
            )
            property_pattern = re.compile(
                r'^\s*(?P<mods>(?:(?:public|private|protected|internal|static|virtual|override|abstract|sealed|new)\s+)*)'
                r'(?P<ret>[A-Za-z_][\w<>,\.\?\[\]\s]*)\s+'
                r'(?P<name>[A-Za-z_]\w*)\s*\{\s*(?:get|set|init)'
            )

            brace_depth = 0
            for i, line in enumerate(lines):
                class_match = class_pattern.search(line)
                if class_match:
                    kind = class_match.group("kind")
                    class_name = class_match.group("name")
                    tail = (class_match.group("tail") or "").strip()
                    signature = f"{kind} {class_name}{tail}".strip()
                    tags.append(CodeTag(
                        file_path=file_path_str,
                        line_number=i + 1,
                        name=class_name,
                        kind="class",
                        signature=signature,
                        parent_symbol=class_name,
                    ))
                    current_class = class_name

                method_match = method_pattern.search(line.strip())
                if method_match:
                    ret_type = re.sub(r"\s+", " ", method_match.group("ret").strip())
                    method_name = method_match.group("name")
                    params = method_match.group("params").strip()
                    modifiers = re.sub(r"\s+", " ", (method_match.group("mods") or "").strip())
                    sig = f"{ret_type} {method_name}({params})"
                    tags.append(CodeTag(
                        file_path=file_path_str,
                        line_number=i + 1,
                        name=method_name,
                        kind="method",
                        signature=sig,
                        return_type=ret_type,
                        parameters=self._parse_parameters(params),
                        parent_symbol=current_class,
                        access_modifier=modifiers,
                    ))

                prop_match = property_pattern.search(line.strip())
                if prop_match:
                    ret_type = re.sub(r"\s+", " ", prop_match.group("ret").strip())
                    prop_name = prop_match.group("name")
                    modifiers = re.sub(r"\s+", " ", (prop_match.group("mods") or "").strip())
                    tags.append(CodeTag(
                        file_path=file_path_str,
                        line_number=i + 1,
                        name=prop_name,
                        kind="method",
                        signature=f"{ret_type} {prop_name} {{ get; set; }}",
                        return_type=ret_type,
                        parent_symbol=current_class,
                        access_modifier=modifiers,
                    ))

                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    current_class = ""
                    brace_depth = max(brace_depth, 0)

        elif lang == "python":
            class_pattern = re.compile(r'^\s*class\s+([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:')
            func_pattern = re.compile(
                r'^\s*def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:'
            )

            class_stack: List[Tuple[str, int]] = []
            for i, raw_line in enumerate(lines):
                if not raw_line.strip() or raw_line.strip().startswith("#"):
                    continue

                indent = len(raw_line) - len(raw_line.lstrip(" "))
                while class_stack and indent <= class_stack[-1][1]:
                    class_stack.pop()

                class_match = class_pattern.search(raw_line)
                if class_match:
                    cls_name = class_match.group(1)
                    tags.append(CodeTag(
                        file_path=file_path_str,
                        line_number=i + 1,
                        name=cls_name,
                        kind="class",
                        signature=raw_line.strip()[:180],
                        parent_symbol=cls_name,
                    ))
                    class_stack.append((cls_name, indent))
                    continue

                func_match = func_pattern.search(raw_line)
                if func_match:
                    fn_name = func_match.group(1)
                    params = (func_match.group(2) or "").strip()
                    ret = (func_match.group(3) or "").strip()
                    in_class = bool(class_stack)
                    kind = "method" if in_class else "function"
                    sig = f"{fn_name}({params})"
                    if ret:
                        sig += f" -> {ret}"
                    tags.append(CodeTag(
                        file_path=file_path_str,
                        line_number=i + 1,
                        name=fn_name,
                        kind=kind,
                        signature=sig,
                        return_type=ret,
                        parameters=self._parse_parameters(params),
                        parent_symbol=class_stack[-1][0] if in_class else "",
                    ))

        elif lang == "lua":
            class_assign_pattern = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*\{')
            func_pattern = re.compile(r'^\s*(?:local\s+)?function\s+([\w.]+(?::[\w]+)?)\s*\(([^)]*)\)')

            for i, line in enumerate(lines):
                class_match = class_assign_pattern.search(line)
                if class_match:
                    cls_name = class_match.group(1)
                    tags.append(CodeTag(
                        file_path=file_path_str,
                        line_number=i + 1,
                        name=cls_name,
                        kind="class",
                        signature=f"{cls_name} = {{}}",
                        parent_symbol=cls_name,
                    ))

                match = func_pattern.search(line)
                if match:
                    full_name = match.group(1)
                    params = match.group(2).strip()
                    parent_symbol = ""
                    if ':' in full_name:
                        parent_symbol = full_name.split(':', 1)[0]
                    elif '.' in full_name:
                        parent_symbol = full_name.rsplit('.', 1)[0]

                    kind = "method" if parent_symbol else "function"
                    tags.append(CodeTag(
                        file_path=file_path_str,
                        line_number=i + 1,
                        name=full_name,
                        kind=kind,
                        signature=f"{full_name}({params})",
                        parameters=self._parse_parameters(params),
                        parent_symbol=parent_symbol,
                    ))

        return tags

    def generate_for_file(self, file_path: str) -> Optional[FileSignature]:
        """为单个文件生成签名列表

        Args:
            file_path: 文件路径 (相对或绝对)

        Returns:
            FileSignature 或 None
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.root / path

        # 获取相对路径
        try:
            rel_path = str(path.relative_to(self.root))
        except ValueError:
            rel_path = str(path)

        # 检查缓存
        cached = self._get_cached_signature(rel_path)
        if cached:
            return cached

        # 检查文件是否在本地
        if path.exists():
            # 本地文件，直接解析
            return self._generate_local(path, rel_path)
        elif self.remote_bridge_url:
            # 远程文件，通过桥接服务
            return self._generate_remote(rel_path)
        else:
            logger.warning(f"文件不存在且无远程桥接服务：{path}")
            return None

    def _generate_local(self, path: Path, rel_path: str) -> Optional[FileSignature]:
        """生成本地文件的签名"""
        # 检测语言
        lang = filename_to_lang(str(path)) if HAS_TREE_SITTER else self._detect_language(path)
        if not lang:
            return None

        # 提取标签：优先 tree-sitter，回退正则
        tags = []
        if HAS_TREE_SITTER and self.parsers.get(lang.lower()):
            tags = self._extract_tags_tree_sitter(path, lang)

        if not tags:
            tags = self._extract_tags_regex(path, lang)

        if not tags:
            return None

        # 组织签名
        sig = FileSignature(
            file_path=rel_path,
            language=lang
        )

        for tag in tags:
            if tag.kind == "class":
                sig.classes.append(tag)
            elif tag.kind == "function":
                sig.functions.append(tag)
            elif tag.kind == "method":
                sig.methods.append(tag)

        # 缓存
        self._cache_signature(rel_path, sig)

        return sig

    def _normalize_remote_path(self, rel_path: str) -> str:
        """标准化远程路径，去掉图数据库中的 record/ 前缀"""
        # 图数据库存的路径格式: src/main/Binary/...
        # 桥接服务 root 是 D:\Git\main，目录结构: Client/Binary/...
        for prefix in ["record/", "Record/"]:
            if rel_path.startswith(prefix):
                return rel_path[len(prefix):]
        return rel_path

    def _generate_remote(self, rel_path: str) -> Optional[FileSignature]:
        """通过远程桥接服务获取文件内容，本地解析生成签名"""
        try:
            # 标准化路径
            remote_path = self._normalize_remote_path(rel_path)

            # 通过桥接服务读取文件内容（优先 /read_file，兼容旧版 /api/read_file）
            payload = {'path': remote_path}
            response = requests.post(
                f"{self.remote_bridge_url}/read_file",
                json=payload,
                timeout=10
            )
            if response.status_code == 404:
                response = requests.post(
                    f"{self.remote_bridge_url}/api/read_file",
                    json=payload,
                    timeout=10
                )

            if response.status_code != 200:
                logger.warning(f"远程读取失败 {remote_path}: HTTP {response.status_code}")
                return None

            data = response.json()
            if not data.get('success'):
                logger.warning(f"远程读取失败 {remote_path}: {data.get('error')}")
                return None

            content = data.get('content', '')
            if not content:
                return None

            # 检测语言
            lang = self._detect_language(Path(rel_path))
            if HAS_TREE_SITTER:
                detected = filename_to_lang(rel_path)
                if detected:
                    lang = detected
            if not lang:
                return None

            # 优先 tree-sitter AST 解析，回退正则
            tags = []
            if HAS_TREE_SITTER and self.parsers.get(lang.lower()):
                tags = self._extract_tags_from_code(rel_path, lang, content)

            if not tags:
                lines = content.splitlines()
                tags = self._extract_tags_regex_from_lines(rel_path, lang, lines)

            if not tags:
                return None

            # 组织签名
            sig = FileSignature(file_path=rel_path, language=lang)
            for tag in tags:
                if tag.kind == "class":
                    sig.classes.append(tag)
                elif tag.kind == "function":
                    sig.functions.append(tag)
                elif tag.kind == "method":
                    sig.methods.append(tag)

            # 缓存
            self._cache_signature(rel_path, sig)
            return sig

        except requests.RequestException as e:
            logger.error(f"远程桥接服务请求失败：{e}")
            return None
        except Exception as e:
            logger.error(f"远程生成异常：{e}")
            return None

    def _detect_language(self, path: Path) -> Optional[str]:
        """检测文件语言 (备用方案)"""
        ext = path.suffix.lower()
        lang_map = {
            ".cs": "c#",
            ".py": "python",
            ".lua": "lua",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
        }
        return lang_map.get(ext)

    def generate_for_directory(
        self,
        directory: str = ".",
        pattern: str = "**/*",
        max_files: int = 100
    ) -> RepoMapResult:
        """为目录生成 Repo Map

        Args:
            directory: 目录路径 (相对或绝对)
            pattern: 文件模式
            max_files: 最大文件数

        Returns:
            RepoMapResult
        """
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.root / directory

        result = RepoMapResult()

        # 收集文件
        files_to_process = []
        for ext in ["*.cs", "*.py", "*.lua"]:
            files_to_process.extend(dir_path.glob(f"**/{ext}"))

        # 限制数量
        files_to_process = files_to_process[:max_files]

        for file_path in files_to_process:
            try:
                sig = self.generate_for_file(str(file_path))
                if sig and (sig.classes or sig.functions or sig.methods):
                    result.files.append(sig)
                    result.total_classes += len(sig.classes)
                    result.total_functions += len(sig.functions)
                    result.total_methods += len(sig.methods)
            except Exception as e:
                logger.debug(f"处理文件失败 {file_path}: {e}")

        # 按路径排序
        result.files.sort(key=lambda f: f.file_path)

        return result


# =============================================================================
# RepoMapTool - Agent 工具封装
# =============================================================================

class RepoMapTool:
    """
    Repo Map 工具

    供 Agent 调用的接口
    """

    name = "repo_map"
    description = (
        "生成代码仓库的结构视图，显示类、函数、方法的分布。"
        "适用于快速理解代码组织、查找入口点。"
        "支持本地文件和远程文件（通过 Windows 桥接服务）。"
    )

    def __init__(
        self,
        root_path: Optional[str] = None,
        remote_bridge_url: Optional[str] = None
    ):
        """初始化工具

        Args:
            root_path: 代码根目录，默认为当前工作目录
            remote_bridge_url: Windows 桥接服务 URL (e.g. http://localhost:8765)
        """
        if root_path is None:
            root_path = os.getcwd()
        self.root_path = root_path

        # 从环境变量获取远程桥接 URL
        if remote_bridge_url is None:
            remote_bridge_url = os.environ.get('WINDOWS_FILE_BRIDGE_URL')

        self.remote_bridge_url = remote_bridge_url
        self.generator = RepoMapGenerator(
            root_path=root_path,
            remote_bridge_url=remote_bridge_url
        )

    def to_schema(self) -> Dict[str, Any]:
        """返回工具 Schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "目录路径 (相对或绝对)",
                            "default": "."
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "文件模式 (e.g. '**/*.cs')",
                            "default": "**/*"
                        },
                        "max_files": {
                            "type": "integer",
                            "description": "最大文件数",
                            "default": 50
                        }
                    }
                }
            }
        }

    def __call__(
        self,
        directory: str = ".",
        file_pattern: str = "**/*",
        max_files: int = 50
    ) -> str:
        """执行 Repo Map 生成

        如果 file_pattern 是具体文件路径（包含扩展名），则生成单文件 Repo Map
        否则生成目录 Repo Map
        """
        try:
            # 检查是否是单个文件
            if file_pattern and ('.' in file_pattern.split('/')[-1]):
                # 单文件模式
                sig = self.generator.generate_for_file(file_pattern)
                if sig:
                    result = RepoMapResult()
                    result.files.append(sig)
                    result.total_classes = len(sig.classes)
                    result.total_functions = len(sig.functions)
                    result.total_methods = len(sig.methods)
                    return result.format()
                else:
                    return f"无法生成 Repo Map：{file_pattern}"
            else:
                # 目录模式
                result = self.generator.generate_for_directory(
                    directory=directory,
                    pattern=file_pattern,
                    max_files=max_files
                )
                return result.format()
        except Exception as e:
            logger.error(f"Repo Map 生成失败：{e}")
            import traceback
            traceback.print_exc()
            return f"生成失败：{e}"


# =============================================================================
# 便捷函数
# =============================================================================

def generate_repo_map(
    root_path: str,
    directory: str = ".",
    max_files: int = 50,
    remote_bridge_url: Optional[str] = None
) -> str:
    """便捷函数：生成 Repo Map"""
    tool = RepoMapTool(root_path=root_path, remote_bridge_url=remote_bridge_url)
    return tool(directory=directory, max_files=max_files)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Repo Map 生成器测试")
    print("=" * 60)

    # 测试当前目录
    root = "/home/developer/projects/open_graph_agent/open_graph_agent/src"
    tool = RepoMapTool(root_path=root)

    print("\n生成 src 目录的 Repo Map:")
    print("-" * 40)
    result = tool(directory=".", max_files=20)
    print(result)
