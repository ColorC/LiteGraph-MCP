# -*- coding: utf-8 -*-
"""
代码分片器 (Code Chunker)

使用 tree-sitter AST 将代码文件按语义边界切分为片段。
支持 Lua (.lua) 和 C# (.cs)。

分片规则:
- Lua: 每个 function_declaration 作为独立 chunk，文件头部合并为 file_header
- C#: 类体 ≤200 行整体作为 chunk，>200 行拆分为方法级 chunk
- 最小 50 行规则: 小 chunk 与相邻 chunk 合并
- Fallback: 不支持的语言按 100 行窗口滑窗切分
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# tree-sitter 初始化
_TS_PARSERS = {}

try:
    from tree_sitter import Language, Parser

    try:
        import tree_sitter_lua as _tslua
        _TS_PARSERS["lua"] = Parser(Language(_tslua.language()))
    except ImportError:
        pass
    try:
        import tree_sitter_c_sharp as _tscs
        _TS_PARSERS["csharp"] = Parser(Language(_tscs.language()))
    except ImportError:
        pass
except ImportError:
    logger.warning("tree-sitter 未安装，将使用滑窗 fallback")


MIN_CHUNK_LINES = 50
MAX_CLASS_LINES_FOR_WHOLE = 200
FALLBACK_WINDOW = 100
FALLBACK_OVERLAP = 20


@dataclass
class CodeChunk:
    """代码片段"""
    symbol_type: str      # "class" / "method" / "function" / "file_header" / "file_chunk"
    symbol_name: str      # 完整限定名
    start_line: int       # 1-based
    end_line: int         # 1-based, inclusive
    code: str             # 实际代码文本
    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def _detect_lang(file_path: str) -> Optional[str]:
    """从文件扩展名检测语言"""
    ext = Path(file_path).suffix.lower()
    return {".lua": "lua", ".cs": "csharp"}.get(ext)


def _get_lines(code: str, start: int, end: int) -> str:
    """提取代码行 (1-based, inclusive)"""
    lines = code.split("\n")
    return "\n".join(lines[start - 1 : end])


# =========================================================================
# Lua 分片
# =========================================================================

def _chunk_lua(code: str, file_path: str) -> List[CodeChunk]:
    """用 tree-sitter 对 Lua 文件分片"""
    parser = _TS_PARSERS.get("lua")
    if not parser:
        return []

    tree = parser.parse(bytes(code, "utf-8"))
    lines = code.split("\n")
    total_lines = len(lines)
    chunks: List[CodeChunk] = []

    # 收集所有顶层 function_declaration 的范围
    func_ranges = []  # (start_line, end_line, name, node)

    for child in tree.root_node.children:
        if child.type == "function_declaration":
            start = child.start_point[0] + 1  # 1-based
            end = child.end_point[0] + 1

            # 提取函数名
            name = ""
            for sub in child.children:
                if sub.type in ("method_index_expression", "dot_index_expression", "identifier"):
                    name = sub.text.decode("utf-8")
                    break
            func_ranges.append((start, end, name or f"__anon_{start}"))

    if not func_ranges:
        # 没有函数定义，整个文件作为一个 chunk
        return [CodeChunk("file_chunk", Path(file_path).stem, 1, total_lines, code)]

    # 文件头部: 第 1 行到第一个函数之前
    first_func_start = func_ranges[0][0]
    if first_func_start > 1:
        header_code = _get_lines(code, 1, first_func_start - 1)
        if header_code.strip():
            chunks.append(CodeChunk(
                "file_header", f"{Path(file_path).stem}:__header",
                1, first_func_start - 1, header_code
            ))

    # 每个函数作为独立 chunk
    for start, end, name in func_ranges:
        func_code = _get_lines(code, start, end)
        kind = "method" if ":" in name or "." in name else "function"
        chunks.append(CodeChunk(kind, name, start, end, func_code))

    # 文件尾部: 最后一个函数之后
    last_func_end = func_ranges[-1][1]
    if last_func_end < total_lines:
        tail_code = _get_lines(code, last_func_end + 1, total_lines)
        if tail_code.strip():
            chunks.append(CodeChunk(
                "file_header", f"{Path(file_path).stem}:__tail",
                last_func_end + 1, total_lines, tail_code
            ))

    return chunks
# =========================================================================
# C# 分片
# =========================================================================

def _chunk_csharp(code: str, file_path: str) -> List[CodeChunk]:
    """用 tree-sitter 对 C# 文件分片"""
    parser = _TS_PARSERS.get("csharp")
    if not parser:
        return []

    tree = parser.parse(bytes(code, "utf-8"))
    lines = code.split("\n")
    total_lines = len(lines)
    chunks: List[CodeChunk] = []

    def _find_classes(node):
        """递归查找所有 class/struct/interface 声明"""
        results = []
        for child in node.children:
            if child.type in ("class_declaration", "struct_declaration", "interface_declaration"):
                results.append(child)
            elif child.type in ("namespace_declaration", "compilation_unit"):
                results.extend(_find_classes(child))
            # 也搜索 namespace body
            elif child.type == "declaration_list":
                results.extend(_find_classes(child))
        return results

    class_nodes = _find_classes(tree.root_node)

    if not class_nodes:
        # 没有类定义，整个文件作为 chunk
        return [CodeChunk("file_chunk", Path(file_path).stem, 1, total_lines, code)]

    for cls_node in class_nodes:
        cls_start = cls_node.start_point[0] + 1
        cls_end = cls_node.end_point[0] + 1
        cls_lines = cls_end - cls_start + 1

        # 提取类名
        cls_name = ""
        for child in cls_node.children:
            if child.type == "identifier":
                cls_name = child.text.decode("utf-8")
                break

        if cls_lines <= MAX_CLASS_LINES_FOR_WHOLE:
            # 整个类作为一个 chunk
            cls_code = _get_lines(code, cls_start, cls_end)
            chunks.append(CodeChunk("class", cls_name, cls_start, cls_end, cls_code))
        else:
            # 拆分为方法级 chunk
            # 提取类声明行作为上下文前缀
            cls_decl_line = _get_lines(code, cls_start, cls_start)

            for child in cls_node.children:
                if child.type == "declaration_list":
                    for member in child.children:
                        if member.type == "method_declaration":
                            m_start = member.start_point[0] + 1
                            m_end = member.end_point[0] + 1
                            m_name = ""
                            for sub in member.children:
                                if sub.type == "identifier":
                                    m_name = sub.text.decode("utf-8")
                                    break
                            m_code = cls_decl_line + "\n...\n" + _get_lines(code, m_start, m_end)
                            chunks.append(CodeChunk(
                                "method", f"{cls_name}.{m_name}",
                                m_start, m_end, m_code
                            ))

    return chunks


# =========================================================================
# 合并小 chunk + Fallback + 公共入口
# =========================================================================

def _merge_small_chunks(chunks: List[CodeChunk], code: str) -> List[CodeChunk]:
    """合并小于 MIN_CHUNK_LINES 的相邻 chunk"""
    if not chunks:
        return chunks

    merged: List[CodeChunk] = []
    buf = chunks[0]

    for i in range(1, len(chunks)):
        cur = chunks[i]
        if buf.line_count < MIN_CHUNK_LINES:
            # buf 太小，合并 cur 进来
            combined_name = f"{buf.symbol_name}+{cur.symbol_name}"
            combined_code = _get_lines(code, buf.start_line, cur.end_line)
            buf = CodeChunk(
                symbol_type=buf.symbol_type,
                symbol_name=combined_name,
                start_line=buf.start_line,
                end_line=cur.end_line,
                code=combined_code,
            )
        else:
            merged.append(buf)
            buf = cur

    # 最后一个 buf: 如果太小且有前一个 chunk，合并回去
    if buf.line_count < MIN_CHUNK_LINES and merged:
        prev = merged.pop()
        combined_name = f"{prev.symbol_name}+{buf.symbol_name}"
        combined_code = _get_lines(code, prev.start_line, buf.end_line)
        merged.append(CodeChunk(
            symbol_type=prev.symbol_type,
            symbol_name=combined_name,
            start_line=prev.start_line,
            end_line=buf.end_line,
            code=combined_code,
        ))
    else:
        merged.append(buf)

    return merged


def _chunk_fallback(code: str, file_path: str) -> List[CodeChunk]:
    """滑窗 fallback: 100 行窗口, 20 行重叠"""
    lines = code.split("\n")
    total = len(lines)
    if total == 0:
        return []

    chunks = []
    start = 0
    n = 0
    while start < total:
        end = min(start + FALLBACK_WINDOW, total)
        chunk_code = "\n".join(lines[start:end])
        n += 1
        chunks.append(CodeChunk(
            "file_chunk", f"__chunk_{n}",
            start + 1, end, chunk_code
        ))
        start += FALLBACK_WINDOW - FALLBACK_OVERLAP
        if end >= total:
            break

    return chunks


def chunk_code(code: str, file_path: str) -> List[CodeChunk]:
    """
    对代码文件进行语义分片。

    Args:
        code: 文件内容
        file_path: 文件路径 (用于检测语言和命名)

    Returns:
        CodeChunk 列表
    """
    if not code or not code.strip():
        return []

    total_lines = len(code.split("\n"))

    # 整个文件 < 50 行，作为一个 chunk
    if total_lines < MIN_CHUNK_LINES:
        return [CodeChunk("file_chunk", Path(file_path).stem, 1, total_lines, code)]

    lang = _detect_lang(file_path)
    chunks: List[CodeChunk] = []

    if lang == "lua":
        chunks = _chunk_lua(code, file_path)
    elif lang == "csharp":
        chunks = _chunk_csharp(code, file_path)

    # AST 解析失败或不支持的语言 → fallback
    if not chunks:
        chunks = _chunk_fallback(code, file_path)

    # 合并小 chunk
    chunks = _merge_small_chunks(chunks, code)

    return chunks
