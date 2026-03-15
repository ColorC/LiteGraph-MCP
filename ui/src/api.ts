const API_BASE = "/api";

export interface Question {
  id: string;
  question: string;
  category: string;
  status: string;
  context: string;
  related_node_id: string;
  answer?: string;
  created_at: string;
  // Dynamic props
  [key: string]: any;
}

export interface GraphNode {
  id: string;
  label?: string;
  [key: string]: any;
}

export interface GraphLink {
  source: string;
  target: string;
  relationship?: string;
  [key: string]: any;
}

export const api = {
  async getQuestions(
    category?: string,
    status: string = "pending",
    page: number = 1,
    pageSize: number = 20,
    keyword?: string,
  ): Promise<{ items: Question[]; total: number }> {
    const params = new URLSearchParams();
    if (category && category !== "all") params.append("category", category);
    if (status && status !== "all") params.append("status", status);
    params.append("page", page.toString());
    params.append("page_size", pageSize.toString());
    if (keyword) params.append("keyword", keyword);

    try {
      const res = await fetch(`${API_BASE}/questions?${params.toString()}`);
      if (!res.ok) throw new Error("Failed to fetch questions");
      return res.json();
    } catch (e) {
      console.error("API Error:", e);
      return { items: [], total: 0 };
    }
  },

  async approveQuestion(id: string, refined_answer?: string): Promise<void> {
    const res = await fetch(`${API_BASE}/questions/${id}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refined_answer }),
    });
    if (!res.ok) throw new Error("Failed to approve");
  },

  async rejectQuestion(id: string, reason: string): Promise<void> {
    const res = await fetch(`${API_BASE}/questions/${id}/reject`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason }),
    });
    if (!res.ok) throw new Error("Failed to reject");
  },

  async createQuestion(data: Partial<Question>): Promise<Question> {
    const res = await fetch(`${API_BASE}/questions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to create question");
    return res.json();
  },

  async searchNodes(query: string): Promise<GraphNode[]> {
    try {
      const res = await fetch(`${API_BASE}/graph/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error("Failed to search nodes");
      return res.json();
    } catch (e) {
      console.error("Search Error:", e);
      return [];
    }
  },

  async getNeighbors(nodeId: string): Promise<GraphNode[]> {
    try {
      const res = await fetch(`${API_BASE}/graph/neighbors?node_id=${encodeURIComponent(nodeId)}`);
      if (!res.ok) throw new Error("Failed to get neighbors");
      return res.json();
    } catch (e) {
      console.error("Neighbor Error:", e);
      return [];
    }
  },

  async getSampleGraph(
    limit: number = 50,
    strategy: "mesh" | "random" = "mesh",
    labels?: string[],
  ): Promise<{ nodes: GraphNode[]; links: GraphLink[] }> {
    try {
      const params = new URLSearchParams({ limit: limit.toString(), strategy });
      if (labels && labels.length > 0) params.append("labels", labels.join(","));
      const res = await fetch(`${API_BASE}/graph/sample?${params.toString()}`);
      if (!res.ok) throw new Error("Failed to get sample");
      return res.json();
    } catch (e) {
      console.error("Sample Error:", e);
      return { nodes: [], links: [] };
    }
  },

  async getLabels(): Promise<string[]> {
    try {
      const res = await fetch(`${API_BASE}/graph/labels`);
      if (!res.ok) throw new Error("Failed to fetch labels");
      return res.json();
    } catch (e) {
      console.error("Labels Error:", e);
      return [];
    }
  },

  async getNodeTypes(): Promise<{ label: string; count: number }[]> {
    try {
      const res = await fetch(`${API_BASE}/graph/node-types`);
      if (!res.ok) throw new Error("Failed to fetch node types");
      return res.json();
    } catch (e) {
      console.error("NodeTypes Error:", e);
      return [];
    }
  },

  async getNode(nodeId: string): Promise<GraphNode> {
    try {
      const res = await fetch(`${API_BASE}/graph/nodes/${encodeURIComponent(nodeId)}`);
      if (!res.ok) throw new Error("Failed to get node");
      return res.json();
    } catch (e) {
      console.error("Get Node Error:", e);
      throw e;
    }
  },

  async getProposalStats(): Promise<Record<string, any>> {
    try {
      const res = await fetch(`${API_BASE}/proposals/stats`);
      if (!res.ok) throw new Error("Failed to fetch proposal stats");
      return res.json();
    } catch (e) {
      console.error("Proposal Stats Error:", e);
      return {};
    }
  },

  async getProposalHistory(
    status?: string,
    page: number = 1,
    pageSize: number = 20,
    keyword?: string,
  ): Promise<{ items: Question[]; total: number }> {
    const params = new URLSearchParams();
    if (status && status !== "all") params.append("status", status);
    params.append("page", page.toString());
    params.append("page_size", pageSize.toString());
    if (keyword) params.append("keyword", keyword);
    try {
      const res = await fetch(`${API_BASE}/proposals/history?${params.toString()}`);
      if (!res.ok) throw new Error("Failed to fetch proposal history");
      return res.json();
    } catch (e) {
      console.error("Proposal History Error:", e);
      return { items: [], total: 0 };
    }
  },

  // Query Lab APIs
  async queryLabSearch(params: {
    query: string;
    mode?: "default" | "code_semantic" | "custom" | "hybrid" | "dense" | "index_only" | "bm25" | "exact";
    top_k?: number;
    label_filter?: string;
    depth?: number;
    include_raw_graph?: boolean;
    repo_map_semantic_rerank?: boolean;
    repo_map_score_threshold?: number;
    repo_map_alpha?: number;
    repo_map_max_files?: number;
    repo_map_max_symbols_per_file?: number;
    repo_map_order?: "top_down" | "score_only";
    repo_map_include_signature_details?: boolean;
    repo_map_timeout_ms?: number;
    expand_node_ids?: string[];
    expand_top_n?: number;
    expand_hops?: number;
    expand_max_size?: number;
  }): Promise<any> {
    const mode = params.mode || "default";
    const payload: Record<string, any> = {
      query: params.query,
      mode,
    };

    if (mode === "default") {
      payload.top_k = params.top_k || 10;
      payload.label_filter = params.label_filter || "";
      payload.depth = params.depth || 0;
      if (params.include_raw_graph) {
        payload.include_raw_graph = true;
      }
    }

    if (mode === "code_semantic") {
      payload.top_k = params.top_k || 10;
      payload.label_filter = params.label_filter || "";
      payload.repo_map_semantic_rerank = params.repo_map_semantic_rerank ?? false;
      payload.repo_map_score_threshold = params.repo_map_score_threshold ?? 0;
      payload.repo_map_alpha = params.repo_map_alpha ?? 0.7;
      payload.repo_map_max_files = params.repo_map_max_files ?? 8;
      payload.repo_map_max_symbols_per_file = params.repo_map_max_symbols_per_file ?? 30;
      payload.repo_map_order = params.repo_map_order || "top_down";
      payload.repo_map_include_signature_details = params.repo_map_include_signature_details ?? true;
      payload.repo_map_timeout_ms = params.repo_map_timeout_ms ?? 3500;
    }

    if (mode === "custom") {
      payload.top_k = params.top_k || 10;
      payload.expand_node_ids = params.expand_node_ids || [];
      payload.expand_top_n = params.expand_top_n ?? 3;
      payload.expand_hops = params.expand_hops ?? 2;
      payload.expand_max_size = params.expand_max_size ?? 180;
      if (params.label_filter) payload.label_filter = params.label_filter;
      if (params.depth !== undefined) payload.depth = params.depth;
    }

    const res = await fetch(`${API_BASE}/query-lab/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Query failed");
    return res.json();
  },

  async queryLabSql(sql: string, db: string = "main"): Promise<any> {
    const res = await fetch(`${API_BASE}/query-lab/sql`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql, db }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "SQL query failed");
    }
    return res.json();
  },

  async queryLabNeighbors(node_id: string, hops: number = 1): Promise<any> {
    const res = await fetch(`${API_BASE}/query-lab/neighbors`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id, hops }),
    });
    if (!res.ok) throw new Error("Neighbors query failed");
    return res.json();
  },

  async queryLabPath(source_id: string, target_id: string, max_depth: number = 6): Promise<any> {
    const res = await fetch(`${API_BASE}/query-lab/path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id, target_id, max_depth }),
    });
    if (!res.ok) throw new Error("Path query failed");
    return res.json();
  },
};
