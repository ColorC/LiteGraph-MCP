import {
  Select,
  Input,
  Button,
  Card,
  Table,
  Segmented,
  InputNumber,
  message,
  Spin,
  Tag,
  Switch,
} from "antd";
import { useState } from "react";
import { api } from "../api";

const { TextArea } = Input;

type QueryType = "default" | "code_semantic" | "custom" | "neighbors" | "path" | "sql";
type ViewMode = "table" | "json";

export default function QueryLab() {
  const [queryType, setQueryType] = useState<QueryType>("default");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [depth, setDepth] = useState(0);
  const [labelFilter, setLabelFilter] = useState("");
  const [hops, setHops] = useState(1);
  const [maxDepth, setMaxDepth] = useState(6);
  const [sourceId, setSourceId] = useState("");
  const [targetId, setTargetId] = useState("");
  const [sqlDb, setSqlDb] = useState("main");

  const [repoMapSemanticRerank, setRepoMapSemanticRerank] = useState(false);
  const [repoMapScoreThreshold, setRepoMapScoreThreshold] = useState(0);
  const [repoMapAlpha, setRepoMapAlpha] = useState(0.7);
  const [repoMapMaxFiles, setRepoMapMaxFiles] = useState(8);
  const [repoMapMaxSymbolsPerFile, setRepoMapMaxSymbolsPerFile] = useState(30);
  const [repoMapOrder, setRepoMapOrder] = useState<"top_down" | "score_only">("top_down");
  const [repoMapIncludeSignatureDetails, setRepoMapIncludeSignatureDetails] = useState(true);
  const [repoMapTimeoutMs, setRepoMapTimeoutMs] = useState(3500);

  const [expandNodeIds, setExpandNodeIds] = useState("");
  const [expandTopN, setExpandTopN] = useState(3);
  const [expandHops, setExpandHops] = useState(2);
  const [expandMaxSize, setExpandMaxSize] = useState(180);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [includeRawGraph, setIncludeRawGraph] = useState(false);

  const handleExecute = async () => {
    if (!query.trim() && queryType !== "path") {
      message.warning("请输入查询内容");
      return;
    }

    setLoading(true);
    try {
      let res;
      switch (queryType) {
        case "default":
          res = await api.queryLabSearch({
            query,
            mode: queryType,
            top_k: topK,
            label_filter: labelFilter,
            depth,
            include_raw_graph: includeRawGraph,
          });
          break;
        case "code_semantic":
          res = await api.queryLabSearch({
            query,
            mode: queryType,
            top_k: topK,
            label_filter: labelFilter,
            repo_map_semantic_rerank: repoMapSemanticRerank,
            repo_map_score_threshold: repoMapScoreThreshold,
            repo_map_alpha: repoMapAlpha,
            repo_map_max_files: repoMapMaxFiles,
            repo_map_max_symbols_per_file: repoMapMaxSymbolsPerFile,
            repo_map_order: repoMapOrder,
            repo_map_include_signature_details: repoMapIncludeSignatureDetails,
            repo_map_timeout_ms: repoMapTimeoutMs,
          });
          break;
        case "custom":
          res = await api.queryLabSearch({
            query,
            mode: queryType,
            top_k: topK,
            expand_node_ids: expandNodeIds
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean),
            expand_top_n: expandTopN,
            expand_hops: expandHops,
            expand_max_size: expandMaxSize,
          });
          break;
        case "neighbors":
          res = await api.queryLabNeighbors(query, hops);
          break;
        case "path":
          if (!sourceId.trim() || !targetId.trim()) {
            message.warning("请输入起点和终点节点 ID");
            setLoading(false);
            return;
          }
          res = await api.queryLabPath(sourceId, targetId, maxDepth);
          break;
        case "sql":
          res = await api.queryLabSql(query, sqlDb);
          break;
      }
      setResult(res);
      message.success(`查询完成，耗时 ${res.elapsed_ms || 0}ms`);
    } catch (e: any) {
      message.error(e.message || "查询失败");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const renderInputArea = () => {
    switch (queryType) {
      case "path":
        return (
          <div style={{ display: "flex", gap: 12 }}>
            <Input
              placeholder="起点节点 ID"
              value={sourceId}
              onChange={(e) => setSourceId(e.target.value)}
              style={{ flex: 1 }}
            />
            <Input
              placeholder="终点节点 ID"
              value={targetId}
              onChange={(e) => setTargetId(e.target.value)}
              style={{ flex: 1 }}
            />
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span>最大深度:</span>
              <InputNumber min={1} max={8} value={maxDepth} onChange={(v) => setMaxDepth(v || 6)} />
            </div>
          </div>
        );
      case "neighbors":
        return (
          <div style={{ display: "flex", gap: 12 }}>
            <Input
              placeholder="节点 ID"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{ flex: 1 }}
            />
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span>跳数:</span>
              <InputNumber min={1} max={3} value={hops} onChange={(v) => setHops(v || 1)} />
            </div>
          </div>
        );
      case "sql":
        return (
          <div>
            <div style={{ marginBottom: 8, display: "flex", gap: 12, alignItems: "center" }}>
              <span>数据库:</span>
              <Select value={sqlDb} onChange={setSqlDb} style={{ width: 150 }}>
                <Select.Option value="main">主库 (kg_graph)</Select.Option>
                <Select.Option value="snippets">片段库 (snippets)</Select.Option>
              </Select>
            </div>
            <TextArea
              placeholder="输入 SQL 查询 (仅支持 SELECT / PRAGMA / EXPLAIN)"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={6}
              style={{ fontFamily: "monospace" }}
            />
          </div>
        );
      default:
        return (
          <div>
            <Input
              placeholder="输入查询关键词或节点 ID"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              size="large"
            />
            <div style={{ marginTop: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
              <span>Top K:</span>
              <InputNumber min={1} max={100} value={topK} onChange={(v) => setTopK(v || 10)} />
              <span>深度:</span>
              <Select value={depth} onChange={setDepth} style={{ width: 150 }}>
                <Select.Option value={0}>0 (仅匹配节点)</Select.Option>
                <Select.Option value={1}>1 (+邻域)</Select.Option>
                <Select.Option value={2}>2 (+PPR+路径)</Select.Option>
              </Select>
              <span>标签过滤:</span>
              <Input
                placeholder="如 BusinessTerm"
                value={labelFilter}
                onChange={(e) => setLabelFilter(e.target.value)}
                style={{ width: 150 }}
              />
            </div>

            {queryType === "code_semantic" && (
              <Card size="small" title="代码搜索参数（Code Index + RepoMap）" style={{ marginTop: 12 }}>
                <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
                  <span>语义重排:</span>
                  <Switch checked={repoMapSemanticRerank} onChange={setRepoMapSemanticRerank} />

                  <span>阈值:</span>
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.01}
                    value={repoMapScoreThreshold}
                    onChange={(v) => setRepoMapScoreThreshold(v ?? 0)}
                  />

                  <span>Alpha:</span>
                  <InputNumber min={0} max={1} step={0.05} value={repoMapAlpha} onChange={(v) => setRepoMapAlpha(v ?? 0.7)} />

                  <span>最大文件数:</span>
                  <InputNumber min={1} max={20} value={repoMapMaxFiles} onChange={(v) => setRepoMapMaxFiles(v || 8)} />

                  <span>每文件最大符号:</span>
                  <InputNumber
                    min={1}
                    max={200}
                    value={repoMapMaxSymbolsPerFile}
                    onChange={(v) => setRepoMapMaxSymbolsPerFile(v || 30)}
                  />

                  <span>排序:</span>
                  <Select value={repoMapOrder} onChange={setRepoMapOrder} style={{ width: 140 }}>
                    <Select.Option value="top_down">top_down</Select.Option>
                    <Select.Option value="score_only">score_only</Select.Option>
                  </Select>

                  <span>签名详情:</span>
                  <Switch checked={repoMapIncludeSignatureDetails} onChange={setRepoMapIncludeSignatureDetails} />

                  <span>超时(ms):</span>
                  <InputNumber min={800} max={10000} step={100} value={repoMapTimeoutMs} onChange={(v) => setRepoMapTimeoutMs(v || 3500)} />
                </div>
              </Card>
            )}

            {queryType === "custom" && (
              <Card size="small" title="精确展开参数（Agent）" style={{ marginTop: 12 }}>
                <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
                  <span>中心节点 IDs:</span>
                  <Input
                    placeholder="逗号分隔，如 nodeA,nodeB"
                    value={expandNodeIds}
                    onChange={(e) => setExpandNodeIds(e.target.value)}
                    style={{ minWidth: 320 }}
                  />

                  <span>Top N:</span>
                  <InputNumber min={1} max={20} value={expandTopN} onChange={(v) => setExpandTopN(v || 3)} />

                  <span>Hops:</span>
                  <InputNumber min={1} max={4} value={expandHops} onChange={(v) => setExpandHops(v || 2)} />

                  <span>Max Size:</span>
                  <InputNumber min={40} max={800} value={expandMaxSize} onChange={(v) => setExpandMaxSize(v || 180)} />
                </div>
              </Card>
            )}

            <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8 }}>
              <span>返回原始图快照:</span>
              <Switch checked={includeRawGraph} onChange={setIncludeRawGraph} />
            </div>
          </div>
        );
    }
  };

  const renderResult = () => {
    if (!result) return null;

    const compactResult = {
      query: result.query,
      mode: result.mode,
      requested_mode: result.requested_mode,
      effective_mode: result.effective_mode,
      result_type: result.result_type,
      elapsed_ms: result.elapsed_ms,
      reorder_meta: result.reorder_meta,
      repo_map_meta: result.repo_map_meta,
      readable: result.readable,
      results_preview: Array.isArray(result.results)
        ? result.results.slice(0, 20).map((r: any) => ({
            id: r.id,
            label: r.label,
            name: r.name,
            score: r.score,
          }))
        : undefined,
    };

    // Index-Only 文本视图
    if (result.result_type === "text") {
      return (
        <pre style={{ whiteSpace: "pre-wrap", background: "#f6f8fa", padding: 16, borderRadius: 4 }}>
          {result.results}
        </pre>
      );
    }

    // 叙述视图 (depth >= 1)
    if (result.result_type === "narrative" && result.results?.narrative) {
      return (
        <pre style={{ whiteSpace: "pre-wrap", background: "#f6f8fa", padding: 16, borderRadius: 4 }}>
          {result.results.narrative}
        </pre>
      );
    }

    // 代码搜索结构化视图
    if (result.result_type === "code_search") {
      if (viewMode === "json") {
        return (
          <pre style={{ background: "#f6f8fa", padding: 16, borderRadius: 4, overflow: "auto" }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        );
      }

      const codeColumns = [
        { title: "#", dataIndex: "rank", key: "rank", width: 60 },
        { title: "文件", dataIndex: "path", key: "path", ellipsis: true },
        { title: "置信度", dataIndex: "confidence", key: "confidence", render: (v: string) => <Tag>{v}</Tag> },
        { title: "得分", dataIndex: "score", key: "score", render: (v: number) => v?.toFixed(4) },
      ];

      return (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card size="small" title="Code Index">
            <Table
              dataSource={result.code_index || result.results || []}
              columns={codeColumns}
              size="small"
              pagination={false}
              rowKey={(r: any) => `${r.id}-${r.rank}`}
            />
          </Card>

          <Card size="small" title="RepoMap 状态">
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
              <Tag color={result.repo_map_status === "ok" ? "green" : "orange"}>status: {result.repo_map_status || "unknown"}</Tag>
              <Tag>repo_maps: {Array.isArray(result.repo_map) ? result.repo_map.length : 0}</Tag>
              <Tag>related_terms: {Array.isArray(result.related_terms) ? result.related_terms.length : 0}</Tag>
            </div>
            {(result.repo_map || []).map((m: any, idx: number) => (
              <Card key={`${m.path || "repo-map"}-${idx}`} size="small" style={{ marginBottom: 8 }} title={m.path || "(unknown path)"}>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{m.content || ""}</pre>
              </Card>
            ))}
          </Card>
        </div>
      );
    }

    // custom 精确展开图
    if (result.result_type === "graph" && result.expanded_graph) {
      if (viewMode === "json") {
        return (
          <pre style={{ background: "#f6f8fa", padding: 16, borderRadius: 4, overflow: "auto" }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        );
      }

      const nodeColumns = [
        { title: "ID", dataIndex: "id", key: "id", ellipsis: true },
        { title: "Label", dataIndex: "label", key: "label", render: (v: string) => <Tag>{v}</Tag> },
        { title: "Name", dataIndex: "name", key: "name", ellipsis: true },
        { title: "中心", dataIndex: "is_center", key: "is_center", render: (v: boolean) => (v ? <Tag color="purple">center</Tag> : "") },
      ];

      return (
        <div>
          <div style={{ marginBottom: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Tag color="blue">nodes: {result.expanded_graph.nodes?.length || 0}</Tag>
            <Tag color="green">edges: {result.expanded_graph.edges?.length || 0}</Tag>
            <Tag>hops: {result.expanded_graph.hops || 0}</Tag>
            <Tag color="purple">centers: {result.expanded_graph.centers?.length || 0}</Tag>
          </div>
          <Table
            dataSource={result.expanded_graph.nodes || []}
            columns={nodeColumns}
            size="small"
            pagination={{ pageSize: 20 }}
            rowKey="id"
          />
        </div>
      );
    }

    // GraphRAG 可读卡片 / 聚合视图
    if (result.readable) {
      if (viewMode === "json") {
        const jsonPayload = result.raw_graph ? result : compactResult;
        return (
          <pre style={{ background: "#f6f8fa", padding: 16, borderRadius: 4, overflow: "auto" }}>
            {JSON.stringify(jsonPayload, null, 2)}
          </pre>
        );
      }

      const hasGraphCards = Array.isArray(result.readable.graph_cards) && result.readable.graph_cards.length > 0;
      const hasBusinessSnapshots = Array.isArray(result.readable.business_snapshots) && result.readable.business_snapshots.length > 0;
      const hasCodeSnapshots = Array.isArray(result.readable.code_snapshots) && result.readable.code_snapshots.length > 0;
      const groupedHighlights = (result.readable.highlights || []).reduce((acc: Record<string, any[]>, h: any) => {
        const k = h.label || "Unknown";
        if (!acc[k]) acc[k] = [];
        acc[k].push(h);
        return acc;
      }, {});

      return (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {result.readable.summary && (
            <Card size="small" title="摘要">
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <Tag color="blue">核心结果: {result.readable.summary.core_result_count ?? 0}</Tag>
                <Tag>尾部索引: {result.readable.summary.tail_index_count ?? 0}</Tag>
                <Tag color="green">业务快照: {result.readable.summary.business_snapshot_count ?? 0}</Tag>
                <Tag>代码快照: {result.readable.summary.code_snapshot_count ?? 0}</Tag>
              </div>
            </Card>
          )}

          {result.readable.highlights?.length > 0 && (
            <Card size="small" title="高优先级命中">
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {result.readable.highlights.slice(0, 8).map((h: any, idx: number) => (
                  <div key={idx} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <Tag color="blue">#{h.rank ?? idx + 1}</Tag>
                    <Tag>{h.label || "Unknown"}</Tag>
                    <span style={{ fontWeight: 500 }}>{h.name || "(no name)"}</span>
                    <span style={{ color: "#999" }}>{typeof h.score === "number" ? h.score.toFixed(4) : ""}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {hasGraphCards &&
            result.readable.graph_cards.map((card: any, idx: number) => (
              <Card
                key={`${card.center?.id || card.center?.name || "center"}-${idx}`}
                size="small"
                title={
                  <span>
                    <Tag color="purple">中心</Tag>
                    {card.center?.name || "(unknown)"}
                  </span>
                }
              >
                <div style={{ marginBottom: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <Tag>support: {typeof card.center?.support === "number" ? card.center.support.toFixed(3) : 0}</Tag>
                  <Tag color="blue">nodes: {card.counts?.nodes ?? 0}</Tag>
                  <Tag color="green">edges: {card.counts?.edges ?? 0}</Tag>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {(card.evidence_triplets || []).map((t: any, tIdx: number) => (
                    <div key={tIdx} style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                      <Tag>{t.source?.label || "Source"}</Tag>
                      <span style={{ fontWeight: 500 }}>{t.source?.name || "(unknown)"}</span>
                      <span style={{ color: "#999" }}>→</span>
                      <Tag color="processing">{t.relation || "related"}</Tag>
                      <span style={{ color: "#999" }}>→</span>
                      <Tag>{t.target?.label || "Target"}</Tag>
                      <span style={{ fontWeight: 500 }}>{t.target?.name || "(unknown)"}</span>
                    </div>
                  ))}
                </div>
              </Card>
            ))}

          {!hasGraphCards && hasBusinessSnapshots && (
            <Card size="small" title="业务聚合中心（回退）">
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {result.readable.business_snapshots.slice(0, 3).map((s: any, idx: number) => (
                  <div key={`${s.term || "center"}-${idx}`} style={{ padding: 10, border: "1px solid #f0f0f0", borderRadius: 6 }}>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
                      <Tag color="purple">中心</Tag>
                      <span style={{ fontWeight: 600 }}>{s.term || "(unknown)"}</span>
                      <Tag>support: {typeof s.support === "number" ? s.support.toFixed(3) : "0.000"}</Tag>
                      <Tag color="blue">nodes: {s.node_count ?? 0}</Tag>
                      <Tag color="green">edges: {s.edge_count ?? 0}</Tag>
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {(s.focus_nodes || []).slice(0, 8).map((n: any, nIdx: number) => (
                        <Tag key={nIdx}>{`${n.label || "Node"}: ${n.name || "(unknown)"}`}</Tag>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {!hasGraphCards && !hasBusinessSnapshots && Object.keys(groupedHighlights).length > 0 && (
            <Card size="small" title="命中分组（回退）">
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {Object.entries(groupedHighlights).map(([label, items]) => (
                  <div key={label} style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                    <Tag color="geekblue">{label}</Tag>
                    {(items as any[]).slice(0, 3).map((item, idx) => (
                      <Tag key={idx}>{item.name || "(no name)"}</Tag>
                    ))}
                  </div>
                ))}
              </div>
            </Card>
          )}

          {hasCodeSnapshots && (
            <Card size="small" title="代码聚合参考">
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {result.readable.code_snapshots.slice(0, 3).map((c: any, idx: number) => (
                  <div key={`${c.path || c.code || "code"}-${idx}`} style={{ padding: 10, border: "1px solid #f0f0f0", borderRadius: 6 }}>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
                      <Tag color="cyan">Code</Tag>
                      <span style={{ fontWeight: 600 }}>{c.code || "(unknown)"}</span>
                      <Tag>{c.path || ""}</Tag>
                      <Tag>score: {typeof c.score === "number" ? c.score.toFixed(3) : "0.000"}</Tag>
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {(c.outline_preview || []).slice(0, 6).map((o: any, oIdx: number) => (
                        <Tag key={oIdx}>{`${o.label || "Node"}: ${o.name || "(unknown)"}`}</Tag>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {result.readable.index_refs?.length > 0 && (
            <Card size="small" title="索引尾部参考">
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {result.readable.index_refs.map((r: any, idx: number) => (
                  <div key={idx} style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <Tag>#{r.rank ?? idx + 1}</Tag>
                    <Tag>{r.label || "Unknown"}</Tag>
                    <span>{r.name || "(no name)"}</span>
                    <span style={{ color: "#999" }}>{typeof r.score === "number" ? r.score.toFixed(4) : ""}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      );
    }

    // SQL 结果
    if (result.columns && result.rows) {
      const columns = result.columns.map((col: string) => ({
        title: col,
        dataIndex: col,
        key: col,
        ellipsis: true,
        render: (val: any) => (typeof val === "object" ? JSON.stringify(val) : String(val)),
      }));

      return viewMode === "table" ? (
        <Table
          dataSource={result.rows}
          columns={columns}
          size="small"
          pagination={{ pageSize: 20, showTotal: (t) => `共 ${t} 条` }}
          scroll={{ x: "max-content" }}
        />
      ) : (
        <pre style={{ background: "#f6f8fa", padding: 16, borderRadius: 4, overflow: "auto" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      );
    }

    // 路径结果
    if (result.found !== undefined) {
      if (!result.found) {
        return <div style={{ padding: 16, textAlign: "center", color: "#999" }}>未找到路径</div>;
      }
      return (
        <div style={{ padding: 16 }}>
          <div style={{ marginBottom: 12 }}>
            <Tag color="blue">找到路径</Tag>
            <span>跳数: {result.hops}</span>
          </div>
          <div style={{ marginBottom: 12 }}>
            {result.path?.map((node: any, i: number) => (
              <span key={i}>
                <Tag>{node.label}</Tag>
                <span style={{ fontWeight: "bold" }}>{node.name}</span>
                {i < result.path.length - 1 && (
                  <span style={{ margin: "0 8px", color: "#999" }}>
                    → [{result.edges?.[i]?.relationship}] →
                  </span>
                )}
              </span>
            ))}
          </div>
          {viewMode === "json" && (
            <pre style={{ background: "#f6f8fa", padding: 16, borderRadius: 4, overflow: "auto" }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      );
    }

    // 邻域结果
    if (result.nodes && result.edges) {
      const columns = [
        { title: "ID", dataIndex: "id", key: "id", ellipsis: true },
        { title: "Label", dataIndex: "label", key: "label", render: (v: string) => <Tag>{v}</Tag> },
        { title: "Name", dataIndex: "name", key: "name", ellipsis: true },
      ];

      return viewMode === "table" ? (
        <div>
          <div style={{ marginBottom: 12 }}>
            <Tag color="blue">{result.nodes.length} 节点</Tag>
            <Tag color="green">{result.edges.length} 边</Tag>
          </div>
          <Table
            dataSource={result.nodes}
            columns={columns}
            size="small"
            pagination={{ pageSize: 20 }}
            rowKey="id"
          />
        </div>
      ) : (
        <pre style={{ background: "#f6f8fa", padding: 16, borderRadius: 4, overflow: "auto" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      );
    }

    // 检索结果列表
    if (Array.isArray(result.results)) {
      const columns = [
        { title: "ID", dataIndex: "id", key: "id", ellipsis: true, width: "25%" },
        { title: "Label", dataIndex: "label", key: "label", render: (v: string) => <Tag>{v}</Tag> },
        { title: "Name", dataIndex: "name", key: "name", ellipsis: true },
        { title: "Score", dataIndex: "score", key: "score", render: (v: number) => v?.toFixed(4) },
      ];

      return viewMode === "table" ? (
        <Table
          dataSource={result.results}
          columns={columns}
          size="small"
          pagination={false}
          rowKey="id"
        />
      ) : (
        <pre style={{ background: "#f6f8fa", padding: 16, borderRadius: 4, overflow: "auto" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      );
    }

    // 默认 JSON 视图
    return (
      <pre style={{ background: "#f6f8fa", padding: 16, borderRadius: 4, overflow: "auto" }}>
        {JSON.stringify(result, null, 2)}
      </pre>
    );
  };

  const showViewToggle = result && result.result_type !== "narrative" && result.result_type !== "text";

  return (
    <div style={{ padding: 20, height: "100%", overflow: "auto" }}>
      <Card title="查询实验" size="small">
        <div style={{ marginBottom: 16 }}>
          <div style={{ marginBottom: 12 }}>
            <span style={{ marginRight: 8 }}>查询类型:</span>
            <Select value={queryType} onChange={setQueryType} style={{ width: 220 }}>
              <Select.Option value="default">默认搜索（主结果 + 快速展开）</Select.Option>
              <Select.Option value="code_semantic">代码搜索（Code Index + RepoMap）</Select.Option>
              <Select.Option value="custom">精确展开（Agent）</Select.Option>
              <Select.Option value="neighbors">邻域查询</Select.Option>
              <Select.Option value="path">路径查找</Select.Option>
              <Select.Option value="sql">SQL 查询</Select.Option>
            </Select>
          </div>

          {renderInputArea()}

          <div style={{ marginTop: 16 }}>
            <Button type="primary" onClick={handleExecute} loading={loading}>
              执行查询
            </Button>
          </div>
        </div>
      </Card>

      {result && (
        <Card
          title={
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>
                查询结果 {result.elapsed_ms && <span style={{ color: "#999" }}>({result.elapsed_ms}ms)</span>}
              </span>
              {showViewToggle && (
                <Segmented
                  options={[
                    { label: "表格", value: "table" },
                    { label: "JSON", value: "json" },
                  ]}
                  value={viewMode}
                  onChange={(v) => setViewMode(v as ViewMode)}
                />
              )}
            </div>
          }
          style={{ marginTop: 16 }}
          size="small"
        >
          <Card size="small" title="结果结构" style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Tag color="blue">mode: {result.effective_mode || result.mode || "unknown"}</Tag>
              <Tag>core_results: {Array.isArray(result.core_results) ? result.core_results.length : 0}</Tag>
              <Tag>quick_expansions: {Array.isArray(result.quick_expansions) ? result.quick_expansions.length : 0}</Tag>
              <Tag>index_refs: {Array.isArray(result.index_refs) ? result.index_refs.length : 0}</Tag>
              <Tag color={result.repo_map_status === "ok" ? "green" : "orange"}>repo_map_status: {result.repo_map_status || "n/a"}</Tag>
              <Tag color={result.raw_graph ? "purple" : "default"}>raw_graph: {result.raw_graph ? "on" : "off"}</Tag>
            </div>
          </Card>

          {result.raw_graph && (
            <Card size="small" title="原始快照已开启" style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <Tag>term_snapshots: {Array.isArray(result.raw_graph?.term_snapshots) ? result.raw_graph.term_snapshots.length : 0}</Tag>
                <Tag>code_repomap_snapshots: {Array.isArray(result.raw_graph?.code_repomap_snapshots) ? result.raw_graph.code_repomap_snapshots.length : 0}</Tag>
              </div>
            </Card>
          )}

          {result.repo_map_meta && (
            <Card size="small" title="RepoMap 元数据" style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
                <Tag color="blue">scoring: {result.repo_map_meta.scoring_mode || "unknown"}</Tag>
                <Tag>candidate: {result.repo_map_meta.candidate_files ?? 0}</Tag>
                <Tag color="green">selected: {result.repo_map_meta.selected_files ?? 0}</Tag>
                <Tag>threshold: {result.repo_map_meta.score_threshold ?? 0}</Tag>
                <Tag>alpha: {result.repo_map_meta.alpha ?? 0}</Tag>
                <Tag>order: {result.repo_map_meta.order || "top_down"}</Tag>
              </div>
              <pre style={{ background: "#f6f8fa", padding: 12, borderRadius: 4, overflow: "auto", margin: 0 }}>
                {JSON.stringify(result.repo_map_meta, null, 2)}
              </pre>
            </Card>
          )}
          <Spin spinning={loading}>{renderResult()}</Spin>
        </Card>
      )}
    </div>
  );
}
