import React from "react";

/**
 * 识别文本中的Wiki链接并渲染为可点击超链接。
 *
 * 支持的模式：
 * 1. Wiki token: (token: XxxYyyZzz) 格式 → https://lilithgames.wiki.cn/wiki/XxxYyyZzz
 * 2. wiki_story:数字 格式 → https://project.wiki.cn/my_project/story/detail/数字
 * 3. status_story:数字 格式 → https://project.wiki.cn/my_project/status_story/detail/数字
 */

// Wiki token: 括号内 "token: XXX" 或 "token:XXX"
const WIKI_TOKEN_RE = /\btoken:\s*([A-Za-z0-9_-]{20,})\b/g;

// wiki_story:数字（node ID 格式）
const WIKI_STORY_RE = /\bwiki_story:(\d{8,12})\b/g;

// status_story:数字
const STATUS_STORY_RE = /\bstatus_story:(\d{8,12})\b/g;

interface WikiLinkifyProps {
  text: string;
  style?: React.CSSProperties;
}

interface Segment {
  type: "text" | "wiki" | "story" | "status_story";
  value: string; // 原始匹配文本
  id?: string; // token 或 story ID
}

function parseSegments(text: string): Segment[] {
  // 收集所有匹配及其位置
  const matches: { start: number; end: number; segment: Segment }[] = [];

  for (const re of [WIKI_TOKEN_RE, WIKI_STORY_RE, STATUS_STORY_RE]) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      const type =
        re === WIKI_TOKEN_RE ? "wiki" : re === WIKI_STORY_RE ? "story" : "status_story";
      matches.push({
        start: m.index,
        end: m.index + m[0].length,
        segment: { type, value: m[0], id: m[1] },
      });
    }
  }

  if (matches.length === 0) return [{ type: "text", value: text }];

  // 按位置排序，去重重叠
  matches.sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;
  for (const m of matches) {
    if (m.start < cursor) continue; // 跳过重叠
    if (m.start > cursor) {
      segments.push({ type: "text", value: text.slice(cursor, m.start) });
    }
    segments.push(m.segment);
    cursor = m.end;
  }
  if (cursor < text.length) {
    segments.push({ type: "text", value: text.slice(cursor) });
  }
  return segments;
}

function getUrl(seg: Segment): string {
  switch (seg.type) {
    case "wiki":
      return `https://lilithgames.wiki.cn/wiki/${seg.id}`;
    case "story":
      return `https://project.wiki.cn/my_project/story/detail/${seg.id}`;
    case "status_story":
      return `https://project.wiki.cn/my_project/status_story/detail/${seg.id}`;
    default:
      return "";
  }
}

const WikiLinkify: React.FC<WikiLinkifyProps> = ({ text, style }) => {
  if (!text) return null;

  const segments = parseSegments(text);

  return (
    <span style={style}>
      {segments.map((seg, i) => {
        if (seg.type === "text") return <span key={i}>{seg.value}</span>;
        const url = getUrl(seg);
        return (
          <a
            key={i}
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            title={url}
            style={{ color: "#1890ff" }}
          >
            {seg.value}
          </a>
        );
      })}
    </span>
  );
};

export default WikiLinkify;
