import type { ColumnsType } from "antd/es/table";
import {
  EyeOutlined,
  ReloadOutlined,
  FileTextOutlined,
  CodeOutlined,
  ProjectOutlined,
  DeploymentUnitOutlined,
} from "@ant-design/icons";
import {
  Table,
  Tag,
  Button,
  Space,
  Drawer,
  Typography,
  Descriptions,
  message,
  Input,
  Select,
  Collapse,
  List,
  Card,
} from "antd";
import React, { useEffect, useState } from "react";
import type { Question, GraphNode } from "../api";
import { api } from "../api";
import WikiLinkify from "../components/WikiLinkify";

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;
const { Panel } = Collapse;

const FILTER_KEYS = [
  "embedding",
  "__indexColor",
  "fx",
  "fy",
  "vx",
  "vy",
  "index",
  "color",
  "val",
  "__bckgDimensions",
  "_edge",
  "x",
  "y",
];

// Common labels translation map
const LABEL_MAP: Record<string, string> = {
  CodeFile: "代码文件 (CodeFile)",
  Folder: "文件夹 (Folder)",
  Story: "需求 (Story)",
  Doc: "文档 (Doc)",
  ArchNode: "架构节点 (ArchNode)",
  BusinessTerm: "业务术语 (Term)",
  WikiStory: "Wiki需求 (Story)",
  Prefab: "预制体 (Prefab)",
  Question: "问题 (Question)",
};

const ApprovalQueue: React.FC = () => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loading, setLoading] = useState(false);

  // Selection & Detail State
  const [selectedQuestion, setSelectedQuestion] = useState<Question | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);

  // Related Data
  const [relatedNode, setRelatedNode] = useState<GraphNode | null>(null);
  const [contextNodes, setContextNodes] = useState<GraphNode[]>([]);
  const [dataLoading, setDataLoading] = useState(false);

  // Approval/Rejection Input State
  const [actionInput, setActionInput] = useState("");

  // Filters
  const [categoryFilter, setCategoryFilter] = useState<string | undefined>(undefined);
  const [statusFilter, setStatusFilter] = useState<string>("pending");
  const [keyword, setKeyword] = useState<string>("");

  const [pagination, setPagination] = useState({ current: 1, pageSize: 10, total: 0 });

  const fetchQuestions = async (page = 1, pageSize = 10) => {
    setLoading(true);
    try {
      const data = await api.getQuestions(categoryFilter, statusFilter, page, pageSize, keyword);
      setQuestions(data.items);
      setPagination({ current: page, pageSize, total: data.total ?? 0 });
    } catch (e) {
      console.error(e);
      message.error("加载问题列表失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQuestions();
  }, [categoryFilter, statusFilter, keyword]);

  const handleApprove = async () => {
    if (!selectedQuestion) return;
    try {
      await api.approveQuestion(selectedQuestion.id, actionInput);
      message.success("已同意建议");
      setDrawerVisible(false);
      fetchQuestions();
    } catch (e) {
      message.error("操作失败");
    }
  };

  const handleReject = async () => {
    if (!selectedQuestion) return;
    try {
      const reason = actionInput || "保持现状";
      await api.rejectQuestion(selectedQuestion.id, reason);
      message.success("已保持现状");
      setDrawerVisible(false);
      fetchQuestions();
    } catch (e) {
      message.error("操作失败");
    }
  };

  const showDetails = async (question: Question) => {
    setSelectedQuestion(question);
    setActionInput("");
    setDrawerVisible(true);

    // Reset data
    setRelatedNode(null);
    setContextNodes([]);

    if (question.related_node_id) {
      setDataLoading(true);
      try {
        // Parallel fetch: Node Details + Neighbors (Context)
        const [nodeData, neighbors] = await Promise.all([
          api.getNode(question.related_node_id).catch(() => null),
          api.getNeighbors(question.related_node_id).catch(() => [] as GraphNode[]),
        ]);

        setRelatedNode(nodeData);
        setContextNodes(neighbors);
      } catch (e) {
        console.warn("Could not fetch related info", e);
      } finally {
        setDataLoading(false);
      }
    }
  };

  const getIconForLabel = (label: string | undefined) => {
    if (label === "CodeFile") return <CodeOutlined />;
    if (label === "Doc") return <FileTextOutlined />;
    if (label === "Story") return <ProjectOutlined />;
    return <DeploymentUnitOutlined />;
  };

  // Helper to categorize context nodes
  const importantLabels = ["Story", "Doc", "CodeFile"];
  const primaryContext = contextNodes.filter((n) => importantLabels.includes(n.label || ""));
  const secondaryContext = contextNodes.filter((n) => !importantLabels.includes(n.label || ""));

  const renderNodeProperties = (node: GraphNode) => (
    <Descriptions
      column={1}
      size="small"
      bordered
      labelStyle={{ fontWeight: "bold", width: "120px" }}
      contentStyle={{ fontSize: "14px" }}
    >
      {Object.entries(node)
        .filter(([k]) => !FILTER_KEYS.includes(k) && k !== "id" && k !== "label")
        .map(([k, v]) => (
          <Descriptions.Item key={k} label={k}>
            <span
              style={{
                whiteSpace: "pre-wrap",
                display: "block",
                maxHeight: 150,
                overflowY: "auto",
                fontSize: 14,
              }}
            >
              <WikiLinkify
                text={typeof v === "object" ? JSON.stringify(v, null, 2) : String(v)}
              />
            </span>
          </Descriptions.Item>
        ))}
    </Descriptions>
  );

  const columns: ColumnsType<Question> = [
    {
      title: "问题内容",
      dataIndex: "question",
      key: "question",
      width: 400,
      render: (text) => (
        <Paragraph
          ellipsis={{ rows: 3, expandable: true, symbol: "展开" }}
          style={{ marginBottom: 0 }}
        >
          {text}
        </Paragraph>
      ),
    },
    {
      title: "关联节点",
      dataIndex: "related_node_id",
      key: "related",
      width: 150,
      ellipsis: true,
      render: (text) => <WikiLinkify text={text} style={{ fontSize: 14 }} />,
    },
    {
      title: "分类",
      dataIndex: "category",
      key: "category",
      width: 120,
      render: (cat) => <Tag color="blue">{cat}</Tag>,
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (status) => {
        let color = "default";
        if (status === "approved") color = "success";
        if (status === "rejected") color = "error";
        if (status === "pending") color = "processing";
        return <Tag color={color}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: "操作",
      key: "actions",
      width: 80,
      fixed: "right",
      render: (_, record) => (
        <Button type="link" icon={<EyeOutlined />} onClick={() => showDetails(record)}>
          详情
        </Button>
      ),
    },
  ];

  return (
    <div
      style={{
        padding: 24,
        background: "#fff",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        overflow: "auto",
      }}
    >
      <Space style={{ marginBottom: 16 }} wrap>
        <Select
          style={{ width: 150 }}
          onChange={setCategoryFilter}
          allowClear
          placeholder="全部分类"
          value={categoryFilter}
        >
          <Option value="proposal">人工提议</Option>
          <Option value="game_trivia">游戏问答</Option>
          <Option value="contradictory">矛盾</Option>
          <Option value="ambiguous">模糊</Option>
          <Option value="untraceable">无法追溯</Option>
          <Option value="wrong_association">错误关联</Option>
          <Option value="weak_association">弱关联</Option>
        </Select>

        <Select defaultValue="pending" style={{ width: 120 }} onChange={setStatusFilter}>
          <Option value="pending">待处理</Option>
          <Option value="approved">已批准</Option>
          <Option value="rejected">已拒绝</Option>
          <Option value="all">全部</Option>
        </Select>

        <Input.Search
          placeholder="搜索..."
          onSearch={(val) => setKeyword(val)}
          style={{ width: 200 }}
          allowClear
        />
        <Button icon={<ReloadOutlined />} onClick={() => fetchQuestions()}>
          刷新
        </Button>
      </Space>

      <div style={{ flex: 1, overflow: "auto" }}>
        <Table
          columns={columns}
          dataSource={questions}
          rowKey="id"
          loading={loading}
          pagination={{
            current: pagination.current,
            pageSize: pagination.pageSize,
            total: pagination.total,
            onChange: (page, pageSize) => fetchQuestions(page, pageSize),
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条`,
          }}
          scroll={{ x: 1000 }}
        />
      </div>

      <Drawer
        title="详情审批"
        placement="right"
        size="large"
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
        mask={false}
        footer={
          selectedQuestion?.status === "pending" && (
            <div style={{ padding: 16 }}>
              <div style={{ marginBottom: 16 }}>
                <div style={{ marginBottom: 8 }}>意见 / 修改建议:</div>
                <TextArea
                  rows={3}
                  value={actionInput}
                  onChange={(e) => setActionInput(e.target.value)}
                  placeholder="输入同意时的修改建议，或者保持现状的理由..."
                />
              </div>
              <Space>
                <Button type="primary" onClick={handleApprove}>
                  同意建议
                </Button>
                <Button danger onClick={handleReject}>
                  保持现状
                </Button>
              </Space>
            </div>
          )
        }
      >
        {selectedQuestion && (
          <Space direction="vertical" style={{ width: "100%" }} size="middle">
            {/* 0. Question Details (Evidence, etc) */}
            <Card title="问题详情" size="small" bordered={false} style={{ background: "#fffbe6" }}>
              <Descriptions column={1} size="small">
                <Descriptions.Item label="问题内容">
                  <Text strong>{selectedQuestion.question}</Text>
                </Descriptions.Item>
                <Descriptions.Item label="分类">
                  <Tag color="orange">{selectedQuestion.category}</Tag>
                </Descriptions.Item>
                {Object.entries(selectedQuestion)
                  .filter(
                    ([k]) =>
                      ![
                        "id",
                        "question",
                        "category",
                        "status",
                        "context",
                        "answer",
                        "related_node_id",
                        "created_at",
                      ].includes(k),
                  )
                  .map(([k, v]) => (
                    <Descriptions.Item key={k} label={k}>
                      <span
                        style={{
                          whiteSpace: "pre-wrap",
                          maxHeight: 100,
                          overflow: "auto",
                          display: "block",
                        }}
                      >
                        <WikiLinkify
                          text={typeof v === "object" ? JSON.stringify(v) : String(v)}
                        />
                      </span>
                    </Descriptions.Item>
                  ))}
              </Descriptions>
            </Card>

            {/* 1. Question Info */}
            <div
              style={{
                background: "#f6ffed",
                padding: 16,
                borderRadius: 6,
                border: "1px solid #b7eb8f",
              }}
            >
              <Title level={5} style={{ marginTop: 0 }}>
                当前问题
              </Title>
              <Paragraph style={{ fontSize: 16 }}>{selectedQuestion.question}</Paragraph>
              <Space wrap>
                <Tag>ID: {selectedQuestion.id}</Tag>
                <Tag color="blue">{selectedQuestion.category}</Tag>
                <Text type="secondary">{selectedQuestion.created_at}</Text>
              </Space>
            </div>

            {/* 2. Related Node (Main Focus) */}
            <Card
              title={<Space>{getIconForLabel(relatedNode?.label)} 核心关联节点</Space>}
              loading={dataLoading}
              size="small"
            >
              {relatedNode ? (
                <div key={relatedNode.id}>
                  <div style={{ marginBottom: 10 }}>
                    <Text strong style={{ fontSize: 16, marginRight: 10 }}>
                      {relatedNode.name || relatedNode.id}
                    </Text>
                    <Tag color="geekblue">
                      {LABEL_MAP[relatedNode.label || ""] || relatedNode.label}
                    </Tag>
                  </div>
                  {renderNodeProperties(relatedNode)}
                </div>
              ) : (
                <Text type="secondary">未找到关联节点数据</Text>
              )}
            </Card>

            {/* 3. Priority Context (Story, Doc, Code) */}
            {primaryContext.length > 0 && (
              <List
                header={<div style={{ fontWeight: "bold" }}>直接相关信息 (Story/Doc/Code)</div>}
                bordered
                dataSource={primaryContext}
                renderItem={(item) => (
                  <List.Item>
                    <div style={{ width: "100%" }}>
                      <Space style={{ marginBottom: 8 }}>
                        {getIconForLabel(item.label)}
                        <Text strong>{item.name || item.id}</Text>
                        <Tag>{LABEL_MAP[item.label || ""] || item.label}</Tag>
                        {item._edge && <Tag color="cyan">关系: {item._edge.relationship}</Tag>}
                      </Space>
                      <Collapse ghost size="small">
                        <Panel header="查看详情" key="1">
                          {renderNodeProperties(item)}
                        </Panel>
                      </Collapse>
                    </div>
                  </List.Item>
                )}
              />
            )}

            {/* 4. Other Context (Collapsible) */}
            {secondaryContext.length > 0 && (
              <Collapse>
                <Panel header={`其他相关节点 (${secondaryContext.length})`} key="others">
                  <List
                    size="small"
                    dataSource={secondaryContext}
                    renderItem={(item) => (
                      <List.Item>
                        <Space>
                          <Tag>{LABEL_MAP[item.label || ""] || item.label}</Tag>
                          <Text>{item.name || item.id}</Text>
                          {item._edge && <Text type="secondary">({item._edge.relationship})</Text>}
                        </Space>
                      </List.Item>
                    )}
                  />
                </Panel>
              </Collapse>
            )}

            {/* 5. Original Context Text */}
            <Collapse>
              <Panel header="原始上下文 (Context)" key="ctx">
                <Paragraph style={{ whiteSpace: "pre-wrap" }}>
                  <WikiLinkify text={selectedQuestion.context} />
                </Paragraph>
              </Panel>
            </Collapse>
          </Space>
        )}
      </Drawer>
    </div>
  );
};

export default ApprovalQueue;
