import {
  SearchOutlined,
  ClearOutlined,
  ExpandAltOutlined,
  DeploymentUnitOutlined,
  DownOutlined,
  CheckOutlined,
} from "@ant-design/icons";
import {
  Input,
  Button,
  Space,
  Drawer,
  Descriptions,
  message,
  Checkbox,
  Tag,
  Typography,
  Modal,
  Switch,
  Dropdown,
  Menu,
  Badge,
} from "antd";
import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import type { GraphNode, GraphLink } from "../api";
import { api } from "../api";

const { Text } = Typography;

const FILTER_KEYS = [
  "embedding",
  "__indexColor",
  "fx",
  "fy",
  "fz",
  "vx",
  "vy",
  "vz",
  "index",
  "color",
  "val",
  "__bckgDimensions",
  "_edge",
  "x",
  "y",
  "z",
  "__threeObj",
];

const LABEL_MAP: Record<string, string> = {
  CodeFile: "代码文件",
  Folder: "文件夹",
  Story: "需求",
  Doc: "文档",
  ArchNode: "架构节点",
  BusinessTerm: "业务术语",
  WikiStory: "Wiki需求",
  Prefab: "预制体",
  Question: "问题",
  InBusinessEntity: "业务内实体",
  ConfigField: "配置字段",
  WikiDoc: "Wiki文档",
  NarrativeElement: "叙事元素",
  LogicNode: "逻辑节点",
};

const LABEL_COLORS: Record<string, string> = {
  CodeFile: "#61dafb",
  Folder: "#ffd866",
  Story: "#ff6b6b",
  Doc: "#a9dc76",
  ArchNode: "#ab9df2",
  BusinessTerm: "#78dce8",
  WikiStory: "#fc9867",
  Prefab: "#ff79c6",
  Question: "#f1fa8c",
  InBusinessEntity: "#bd93f9",
  ConfigField: "#50fa7b",
  WikiDoc: "#8be9fd",
  NarrativeElement: "#ffb86c",
  LogicNode: "#ff5555",
};
const DEFAULT_COLOR = "#888888";

function getNodeColor(node: GraphNode): string {
  return LABEL_COLORS[node.label || ""] || DEFAULT_COLOR;
}

// ── 3D sprite 缓存 ──
const spriteCache = new Map<string, THREE.Texture>();

function makeTextSprite(text: string, color: string): THREE.Sprite {
  const cacheKey = `${text}::${color}`;
  let texture = spriteCache.get(cacheKey);

  if (!texture) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    const fontSize = 48;
    ctx.font = `bold ${fontSize}px "Microsoft YaHei", "PingFang SC", sans-serif`;
    const textWidth = ctx.measureText(text).width;

    canvas.width = Math.ceil(textWidth + 24);
    canvas.height = fontSize + 16;

    ctx.font = `bold ${fontSize}px "Microsoft YaHei", "PingFang SC", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // 半透明暗色背景
    ctx.fillStyle = "rgba(20, 20, 30, 0.75)";
    const r = 8;
    const w = canvas.width,
      h = canvas.height;
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(w - r, 0);
    ctx.quadraticCurveTo(w, 0, w, r);
    ctx.lineTo(w, h - r);
    ctx.quadraticCurveTo(w, h, w - r, h);
    ctx.lineTo(r, h);
    ctx.quadraticCurveTo(0, h, 0, h - r);
    ctx.lineTo(0, r);
    ctx.quadraticCurveTo(0, 0, r, 0);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, w / 2, h / 2);

    texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    spriteCache.set(cacheKey, texture);
  }

  const spriteMaterial = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(spriteMaterial);
  const img = texture.image as HTMLCanvasElement;
  const aspect = img.width / img.height;
  sprite.scale.set(aspect * 6, 6, 1);
  return sprite;
}

// ────────────────────────────────────────────────────────────────────────────

const GraphExplorer: React.FC = () => {
  const fgRef = useRef<any>(null);
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({
    nodes: [],
    links: [],
  });
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [is3D, setIs3D] = useState(false); // 默认 2D，轻量

  const [availableLabels, setAvailableLabels] = useState<string[]>([]);
  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);
  const [nodeTypes, setNodeTypes] = useState<{ label: string; count: number }[]>([]);
  const [dropdownVisible, setDropdownVisible] = useState(false);

  const [proposalVisible, setProposalVisible] = useState(false);
  const [proposalTitle, setProposalTitle] = useState("");
  const [proposalDesc, setProposalDesc] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handlePropose = async () => {
    if (!selectedNode || !proposalTitle.trim()) {
      message.warning("请填写标题");
      return;
    }
    setSubmitting(true);
    try {
      await api.createQuestion({
        question: proposalTitle,
        category: "proposal",
        context: proposalDesc,
        related_node_id: selectedNode.id,
      });
      message.success("提议已提交");
      setProposalVisible(false);
      setProposalTitle("");
      setProposalDesc("");
    } catch {
      message.error("提交失败");
    } finally {
      setSubmitting(false);
    }
  };

  useEffect(() => {
    const init = async () => {
      // 自动侦测数据库中的节点类型
      const types = await api.getNodeTypes();
      setNodeTypes(types);
      const labels = types.map((t) => t.label);
      setAvailableLabels(labels);
      setSelectedLabels(labels); // 默认全选

      setLoading(true);
      try {
        const data = await api.getSampleGraph(50, "mesh");
        const nodes = data.nodes.map((n) => ({ ...n, val: 5 }));
        setGraphData({ nodes, links: data.links });
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  const loadMeshSample = async () => {
    setLoading(true);
    try {
      const labelsToUse =
        selectedLabels.length < availableLabels.length ? selectedLabels : undefined;
      const data = await api.getSampleGraph(50, "mesh", labelsToUse);
      const nodes = data.nodes.map((n) => ({ ...n, val: 5 }));
      setGraphData({ nodes, links: data.links });
      message.success("已加载随机关联网络");
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    try {
      const results = await api.searchNodes(searchQuery);
      const nodes = results.map((n) => ({ ...n, val: 5 }));
      setGraphData({ nodes, links: [] });
      setTimeout(() => fgRef.current?.zoomToFit(400, 50), 500);
      message.success(`找到 ${results.length} 个节点`);
    } catch (e) {
      console.error(e);
      message.error("搜索失败");
    } finally {
      setLoading(false);
    }
  };

  const expandNode = async (node: GraphNode) => {
    try {
      message.loading({ content: "正在展开邻居...", key: "expand" });
      const neighbors = await api.getNeighbors(node.id);

      setGraphData((prev) => {
        const newNodes = [...prev.nodes];
        const newLinks = [...prev.links];
        const existingNodeIds = new Set(newNodes.map((n) => n.id));
        const existingLinkIds = new Set(
          newLinks.map((l) => {
            const s = (l.source as any).id || l.source;
            const t = (l.target as any).id || l.target;
            return `${s}-${t}`;
          }),
        );

        let addedCount = 0;
        neighbors.forEach((n) => {
          if (!existingNodeIds.has(n.id)) {
            newNodes.push({ ...n, val: 3 });
            existingNodeIds.add(n.id);
            addedCount++;
          }
          const linkId = `${node.id}-${n.id}`;
          if (!existingLinkIds.has(linkId)) {
            newLinks.push({
              source: node.id,
              target: n.id,
              relationship: n._edge?.relationship,
            });
            existingLinkIds.add(linkId);
          }
        });

        message.success({ content: `已展开 ${addedCount} 个新节点`, key: "expand" });
        return { nodes: newNodes, links: newLinks };
      });
    } catch (e) {
      console.error(e);
      message.error({ content: "展开失败", key: "expand" });
    }
  };

  const getDisplayName = useCallback((node: GraphNode) => {
    if (node.name) return node.name;
    const idStr = String(node.id);
    if (idStr.includes("/") || idStr.includes("\\")) {
      const parts = idStr.split(/[/\\]/);
      return parts.slice(-2).join("/");
    }
    if (idStr.length > 25) return idStr.substring(0, 10) + "…" + idStr.substring(idStr.length - 10);
    return idStr;
  }, []);

  const filteredGraphData = useMemo(
    () => ({
      nodes: graphData.nodes.filter((n) => selectedLabels.includes(n.label || "") || !n.label),
      links: graphData.links.filter((l) => {
        const sId = (l.source as any).id || l.source;
        const tId = (l.target as any).id || l.target;
        const sourceVisible = graphData.nodes.find(
          (n) => n.id === sId && (selectedLabels.includes(n.label || "") || !n.label),
        );
        const targetVisible = graphData.nodes.find(
          (n) => n.id === tId && (selectedLabels.includes(n.label || "") || !n.label),
        );
        return sourceVisible && targetVisible;
      }),
    }),
    [graphData, selectedLabels],
  );

  // ── 3D 节点渲染 ──
  const nodeThreeObject = useCallback(
    (node: GraphNode) => {
      const label = getDisplayName(node);
      const color = getNodeColor(node);
      const group = new THREE.Group();

      const geometry = new THREE.SphereGeometry(1.5, 16, 16);
      const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 });
      group.add(new THREE.Mesh(geometry, material));

      const glowGeometry = new THREE.SphereGeometry(2.5, 16, 16);
      const glowMaterial = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.15 });
      group.add(new THREE.Mesh(glowGeometry, glowMaterial));

      const sprite = makeTextSprite(label, color);
      sprite.position.set(0, 4, 0);
      group.add(sprite);

      return group;
    },
    [getDisplayName],
  );

  // ── 2D 节点渲染 ──
  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const label = getDisplayName(node as GraphNode);
      const color = getNodeColor(node as GraphNode);
      const fontSize = 16 / globalScale;
      ctx.font = `bold ${fontSize}px "Microsoft YaHei", "PingFang SC", sans-serif`;

      const textWidth = ctx.measureText(label).width;
      const bckgDimensions = [textWidth + fontSize * 0.4, fontSize + fontSize * 0.4];

      // 暗色背景
      ctx.fillStyle = "rgba(20, 20, 30, 0.85)";
      ctx.fillRect(
        node.x! - bckgDimensions[0] / 2,
        node.y! - bckgDimensions[1] / 2,
        bckgDimensions[0],
        bckgDimensions[1],
      );

      // 发光边框
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5 / globalScale;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6 / globalScale;
      ctx.strokeRect(
        node.x! - bckgDimensions[0] / 2,
        node.y! - bckgDimensions[1] / 2,
        bckgDimensions[0],
        bckgDimensions[1],
      );
      ctx.shadowBlur = 0;

      // 文字
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, node.x!, node.y!);

      // 类型标签
      if (globalScale > 0.8) {
        const typeLabel = LABEL_MAP[(node as GraphNode).label as string] || "";
        if (typeLabel) {
          const subSize = 10 / globalScale;
          ctx.font = `${subSize}px sans-serif`;
          ctx.fillStyle = color;
          ctx.fillText(typeLabel, node.x!, node.y! + fontSize * 1.1);
        }
      }

      node.__bckgDimensions = bckgDimensions;
    },
    [getDisplayName],
  );

  const nodePointerAreaPaint = useCallback(
    (node: any, color: string, ctx: CanvasRenderingContext2D) => {
      const dims = node.__bckgDimensions as number[];
      if (dims) {
        ctx.fillStyle = color;
        ctx.fillRect(node.x! - dims[0] / 2, node.y! - dims[1] / 2, dims[0], dims[1] + 14);
      }
    },
    [],
  );

  // ── 共享事件处理 ──
  const handleNodeClick = useCallback(
    (node: any) => {
      setSelectedNode(node as GraphNode);
      setDrawerVisible(true);
      if (is3D) {
        const distance = 80;
        if (node.x !== undefined) {
          fgRef.current?.cameraPosition(
            { x: node.x, y: node.y, z: (node.z ?? 0) + distance },
            { x: node.x, y: node.y, z: node.z ?? 0 },
            1000,
          );
        }
      }
    },
    [is3D],
  );

  const handleModeSwitch = useCallback((checked: boolean) => {
    spriteCache.clear();
    setIs3D(checked);
  }, []);

  // ── 节点类型选择相关 ──
  const handleSelectAll = () => {
    setSelectedLabels(availableLabels);
    message.success("已全选所有类型");
  };

  const handleSelectNone = () => {
    setSelectedLabels([]);
    message.success("已取消全选");
  };

  const handleSelectInvert = () => {
    setSelectedLabels((prev) =>
      prev.length === availableLabels.length ? [] : availableLabels.filter((l) => !prev.includes(l))
    );
    message.success("已反选");
  };

  const getTypeMenu = () => {
    return (
      <Menu
        style={{
          maxHeight: "400px",
          overflow: "auto",
          minWidth: "280px",
          background: "#1a1a2e",
        }}
        selectedKeys={[]}
      >
        {/* 快捷操作 */}
        <Menu.Item
          key="all"
          onClick={handleSelectAll}
          style={{ background: "#1a1a2e", color: "#e0e0e0" }}
        >
          <CheckOutlined /> 全选
        </Menu.Item>
        <Menu.Item
          key="none"
          onClick={handleSelectNone}
          style={{ background: "#1a1a2e", color: "#e0e0e0" }}
        >
          <ClearOutlined /> 不选
        </Menu.Item>
        <Menu.Item
          key="invert"
          onClick={handleSelectInvert}
          style={{ background: "#1a1a2e", color: "#e0e0e0" }}
        >
          反选
        </Menu.Item>
        <Menu.Divider style={{ borderColor: "#333" }} />
        {/* 节点类型列表 */}
        {nodeTypes.map((type) => {
          const count = type.count;
          const color = LABEL_COLORS[type.label] || DEFAULT_COLOR;
          const name = LABEL_MAP[type.label] || type.label;
          return (
            <Menu.Item
              key={type.label}
              style={{
                padding: "8px 16px",
                background: "#1a1a2e",
                color: "#e0e0e0",
                marginBottom: 4,
              }}
            >
              <Checkbox
                checked={selectedLabels.includes(type.label)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedLabels((prev) => [...prev, type.label]);
                  } else {
                    setSelectedLabels((prev) => prev.filter((l) => l !== type.label));
                  }
                }}
                onClick={(e) => e.stopPropagation()}
                style={{ color: "#e0e0e0" }}
              >
                <Tag
                  color={color}
                  style={{
                    marginRight: 8,
                    border: "1px solid " + color,
                    background: "rgba(0,0,0,0.3)",
                    color: "#fff",
                    fontWeight: 500,
                  }}
                >
                  {name}
                </Tag>
                <span style={{ color: "#888", fontSize: 12 }}>
                  ({count})
                </span>
              </Checkbox>
            </Menu.Item>
          );
        })}
      </Menu>
    );
  };

  // ── 共享 link 属性 ──
  const sharedLinkProps = {
    linkDirectionalArrowLength: 3,
    linkDirectionalArrowRelPos: 1,
    linkDirectionalParticles: 1,
    linkDirectionalParticleWidth: 1.5,
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* 工具栏 */}
      <div
        style={{
          padding: "12px 16px",
          background: "rgba(15, 15, 25, 0.95)",
          borderBottom: "1px solid rgba(100, 100, 255, 0.15)",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          backdropFilter: "blur(10px)",
        }}
      >
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <Space.Compact style={{ flex: 1 }}>
            <Input
              placeholder="搜索节点 (ID, 标签, 名称)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onPressEnter={handleSearch}
              prefix={<SearchOutlined />}
            />
            <Button type="primary" onClick={handleSearch} loading={loading}>
              搜索
            </Button>
          </Space.Compact>

          <Button
            type="dashed"
            icon={<DeploymentUnitOutlined />}
            onClick={loadMeshSample}
            loading={loading}
          >
            随机网格
          </Button>

          <Button
            icon={<ClearOutlined />}
            onClick={() => {
              setGraphData({ nodes: [], links: [] });
              spriteCache.clear();
            }}
          >
            清空
          </Button>
          <Button onClick={() => fgRef.current?.zoomToFit(400, 50)}>适配</Button>

          <span style={{ color: "#aaa", fontSize: 12, marginLeft: 8 }}>2D</span>
          <Switch checked={is3D} onChange={handleModeSwitch} size="small" />
          <span style={{ color: "#aaa", fontSize: 12 }}>3D</span>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span style={{ fontWeight: "bold", fontSize: 12, color: "#aaa" }}>显示类型:</span>
          <Badge count={selectedLabels.length} size="small" style={{ marginRight: 8 }}>
            <Dropdown
              menu={{ items: [{ key: "types", label: getTypeMenu() }] }}
              trigger={["click"]}
              open={dropdownVisible}
              onOpenChange={setDropdownVisible}
            >
              <Button size="small" icon={<DownOutlined />}>
                选择节点类型
              </Button>
            </Dropdown>
          </Badge>
          {selectedLabels.length === 0 && (
            <Tag color="warning" style={{ fontSize: 11 }}>
              未选择任何类型
            </Tag>
          )}
        </div>
      </div>

      {/* 图渲染区域 */}
      <div style={{ flex: 1, position: "relative", background: "#0a0a14" }}>
        {is3D ? (
          <ForceGraph3D
            ref={fgRef}
            graphData={filteredGraphData}
            backgroundColor="#0a0a14"
            nodeAutoColorBy="label"
            nodeThreeObject={nodeThreeObject}
            nodeThreeObjectExtend={false}
            linkColor={() => "rgba(100, 120, 255, 0.3)"}
            linkWidth={0.5}
            linkOpacity={0.4}
            linkDirectionalArrowColor={() => "rgba(150, 170, 255, 0.5)"}
            linkDirectionalParticleColor={() => "rgba(120, 140, 255, 0.6)"}
            onNodeClick={handleNodeClick}
            nodeLabel={(node) =>
              `${(node as GraphNode).label || "Node"}: ${(node as GraphNode).id}`
            }
            {...sharedLinkProps}
          />
        ) : (
          <ForceGraph2D
            ref={fgRef}
            graphData={filteredGraphData}
            backgroundColor="#0a0a14"
            nodeAutoColorBy="label"
            nodeCanvasObject={nodeCanvasObject}
            nodePointerAreaPaint={nodePointerAreaPaint}
            linkColor={() => "rgba(100, 120, 255, 0.3)"}
            linkWidth={0.5}
            linkDirectionalArrowColor={() => "rgba(150, 170, 255, 0.5)"}
            linkDirectionalParticleColor={() => "rgba(120, 140, 255, 0.6)"}
            onNodeClick={handleNodeClick}
            nodeLabel={(node) =>
              `${(node as GraphNode).label || "Node"}: ${(node as GraphNode).id}`
            }
            d3VelocityDecay={0.1}
            d3AlphaDecay={0.02}
            {...sharedLinkProps}
          />
        )}
      </div>

      {/* 节点详情 Drawer */}
      <Drawer
        title={
          <Space>
            <DeploymentUnitOutlined />
            <span>{selectedNode ? selectedNode.name || selectedNode.id : "节点详情"}</span>
            {selectedNode && <Tag color="blue">{selectedNode.labels?.[0]}</Tag>}
          </Space>
        }
        placement="right"
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
        width={800}
        extra={
          <Button type="primary" size="small" onClick={() => setProposalVisible(true)}>
            提议变更
          </Button>
        }
      >
        {selectedNode && (
          <Descriptions column={1} bordered size="small">
            <Descriptions.Item label="ID">
              <Text copyable code>
                {selectedNode.id}
              </Text>
            </Descriptions.Item>
            {Object.entries(selectedNode)
              .filter(
                ([key]) =>
                  !FILTER_KEYS.includes(key) &&
                  key !== "id" &&
                  key !== "labels" &&
                  key !== "name" &&
                  key !== "label_display",
              )
              .map(([key, value]) => (
                <Descriptions.Item label={key} key={key}>
                  {typeof value === "object" ? JSON.stringify(value) : String(value)}
                </Descriptions.Item>
              ))}
          </Descriptions>
        )}
        {selectedNode && (
          <div style={{ marginTop: 20 }}>
            <Button block onClick={() => expandNode(selectedNode)} icon={<ExpandAltOutlined />}>
              展开邻居节点
            </Button>
          </div>
        )}
      </Drawer>

      {/* 提议变更 Modal */}
      <Modal
        title="提议变更"
        open={proposalVisible}
        onOk={handlePropose}
        onCancel={() => setProposalVisible(false)}
        confirmLoading={submitting}
        okText="提交"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <Text strong>变更标题/问题：</Text>
          <Input
            placeholder="例如：关联关系错误，应连接到..."
            value={proposalTitle}
            onChange={(e) => setProposalTitle(e.target.value)}
            style={{ marginTop: 8 }}
          />
        </div>
        <div>
          <Text strong>详细描述/上下文：</Text>
          <Input.TextArea
            rows={4}
            placeholder="请描述具体的变更建议..."
            value={proposalDesc}
            onChange={(e) => setProposalDesc(e.target.value)}
            style={{ marginTop: 8 }}
          />
        </div>
      </Modal>
    </div>
  );
};

export default GraphExplorer;
