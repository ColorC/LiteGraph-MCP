import {
  Table,
  Tag,
  Input,
  Select,
  Card,
  Statistic,
  Row,
  Col,
  Drawer,
  Descriptions,
  Spin,
} from "antd";
import { useState, useEffect, useCallback } from "react";
import type { Question } from "../api";
import { api } from "../api";
import WikiLinkify from "../components/WikiLinkify";

const STATUS_COLORS: Record<string, string> = {
  pending: "default",
  approved: "blue",
  in_progress: "processing",
  completed: "success",
  completed_no_action: "warning",
  failed: "error",
  rejected: "volcano",
};

const STATUS_LABELS: Record<string, string> = {
  pending: "待审批",
  approved: "已审批",
  in_progress: "执行中",
  completed: "已完成",
  completed_no_action: "无需执行",
  failed: "执行失败",
  rejected: "已拒绝",
};

export default function ProposalMonitor() {
  const [stats, setStats] = useState<Record<string, any>>({});
  const [items, setItems] = useState<Question[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [status, setStatus] = useState<string>("all");
  const [keyword, setKeyword] = useState("");
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Question | null>(null);

  const fetchStats = useCallback(async () => {
    const data = await api.getProposalStats();
    setStats(data);
  }, []);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    const data = await api.getProposalHistory(status, page, pageSize, keyword || undefined);
    setItems(data.items || []);
    setTotal(data.total || 0);
    setLoading(false);
  }, [status, page, pageSize, keyword]);

  useEffect(() => {
    fetchStats();
    fetchHistory();
    const timer = setInterval(() => {
      fetchStats();
      fetchHistory();
    }, 10000);
    return () => clearInterval(timer);
  }, [fetchStats, fetchHistory]);

  const columns = [
    {
      title: "标题",
      dataIndex: "question",
      key: "question",
      ellipsis: true,
      width: "30%",
      render: (text: string) => <span title={text}>{text}</span>,
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (s: string) => (
        <Tag color={STATUS_COLORS[s] || "default"}>{STATUS_LABELS[s] || s}</Tag>
      ),
    },
    {
      title: "关联节点",
      dataIndex: "related_node_id",
      key: "related_node_id",
      ellipsis: true,
      width: "20%",
    },
    {
      title: "创建时间",
      dataIndex: "created_at",
      key: "created_at",
      width: 160,
    },
    {
      title: "完成时间",
      dataIndex: "completed_at",
      key: "completed_at",
      width: 160,
      render: (t: string) => t || "-",
    },
    {
      title: "执行结果",
      dataIndex: "execution_result",
      key: "execution_result",
      ellipsis: true,
      render: (t: string) => t || "-",
    },
  ];

  return (
    <div style={{ padding: 20, height: "100%", overflow: "auto" }}>
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={4}>
          <Card size="small">
            <Statistic title="总计" value={stats.total || 0} />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="待审批"
              value={stats.pending_proposal || 0}
              valueStyle={{ color: "#999" }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="已审批"
              value={stats.approved || 0}
              valueStyle={{ color: "#1890ff" }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="执行中"
              value={stats.in_progress || 0}
              valueStyle={{ color: "#1890ff" }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small">
            <Statistic
              title="已完成"
              value={(stats.completed || 0) + (stats.completed_no_action || 0)}
              valueStyle={{ color: "#52c41a" }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card size="small">
            <Statistic title="失败" value={stats.failed || 0} valueStyle={{ color: "#ff4d4f" }} />
          </Card>
        </Col>
      </Row>

      <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        <Select
          value={status}
          onChange={(v) => {
            setStatus(v);
            setPage(1);
          }}
          style={{ width: 150 }}
        >
          <Select.Option value="all">全部状态</Select.Option>
          {Object.entries(STATUS_LABELS).map(([k, v]) => (
            <Select.Option key={k} value={k}>
              {v}
            </Select.Option>
          ))}
        </Select>
        <Input.Search
          placeholder="搜索标题/上下文"
          allowClear
          onSearch={(v) => {
            setKeyword(v);
            setPage(1);
          }}
          style={{ width: 300 }}
        />
      </div>

      <Spin spinning={loading}>
        <Table
          dataSource={items}
          columns={columns}
          rowKey="id"
          size="small"
          pagination={{
            current: page,
            pageSize,
            total,
            onChange: (p) => setPage(p),
            showTotal: (t) => `共 ${t} 条`,
          }}
          onRow={(record) => ({ onClick: () => setSelected(record), style: { cursor: "pointer" } })}
        />
      </Spin>

      <Drawer title="Proposal 详情" open={!!selected} onClose={() => setSelected(null)} width={800}>
        {selected && (
          <Descriptions column={1} bordered size="small">
            <Descriptions.Item label="ID">{selected.id}</Descriptions.Item>
            <Descriptions.Item label="标题">{selected.question}</Descriptions.Item>
            <Descriptions.Item label="状态">
              <Tag color={STATUS_COLORS[selected.status]}>
                {STATUS_LABELS[selected.status] || selected.status}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="分类">{selected.category}</Descriptions.Item>
            {selected.answer && (
              <Descriptions.Item label="人类审批意见">
                <div
                  style={{
                    whiteSpace: "pre-wrap",
                    background: "#fffbe6",
                    padding: 8,
                    borderRadius: 4,
                  }}
                >
                  {selected.answer}
                </div>
              </Descriptions.Item>
            )}
            <Descriptions.Item label="关联节点">
              <WikiLinkify text={selected.related_node_id} />
            </Descriptions.Item>
            <Descriptions.Item label="上下文">
              <div style={{ whiteSpace: "pre-wrap", maxHeight: 200, overflow: "auto" }}>
                <WikiLinkify text={selected.context} />
              </div>
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">{selected.created_at}</Descriptions.Item>
            {selected.approved_at && (
              <Descriptions.Item label="审批时间">{selected.approved_at}</Descriptions.Item>
            )}
            {selected.started_at && (
              <Descriptions.Item label="开始执行">{selected.started_at}</Descriptions.Item>
            )}
            {selected.completed_at && (
              <Descriptions.Item label="完成时间">{selected.completed_at}</Descriptions.Item>
            )}
            {selected.execution_result && (
              <Descriptions.Item label="执行结果">
                <div style={{ whiteSpace: "pre-wrap" }}>
                  <WikiLinkify text={selected.execution_result} />
                </div>
              </Descriptions.Item>
            )}
            {selected.execution_log && (
              <Descriptions.Item label="执行日志">
                <pre
                  style={{
                    fontSize: 12,
                    maxHeight: 400,
                    overflow: "auto",
                    background: "#f6f8fa",
                    padding: 8,
                    borderRadius: 4,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-all",
                  }}
                >
                  {selected.execution_log}
                </pre>
              </Descriptions.Item>
            )}
            {selected.execution_error && (
              <Descriptions.Item label="错误详情">
                <pre
                  style={{
                    fontSize: 12,
                    maxHeight: 300,
                    overflow: "auto",
                    background: "#f5f5f5",
                    padding: 8,
                  }}
                >
                  {selected.execution_error}
                </pre>
              </Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Drawer>
    </div>
  );
}
