
<p align="center">
  <a href="https://gemini.google.com/">
    <img src="https://img.shields.io/badge/Made%20with-Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white" alt="Made with Gemini">
  </a>
</p>

# LiteGraph-MCP: An Industrial-Grade Knowledge Graph MCP Server

LiteGraph-MCP is a zero-infrastructure, high-performance Knowledge Graph Retrieval-Augmented Generation (GraphRAG) server designed specifically for AI Agents (via the Model Context Protocol - MCP). 

Unlike standard GraphRAG implementations that focus on extracting summaries from text to feed back to LLMs, LiteGraph-MCP is built from the ground up to solve a critical problem in enterprise software engineering: **"How do we map business requirements directly to physical code assets reliably, without suffering from context window pollution or hallucination?"**

## 🌟 Why We Built This

Currently, the open-source ecosystem is polarized:
- **Too Heavy:** Microsoft's GraphRAG or Neo4j-based solutions require heavy infrastructure (JVMs, vector databases, distributed clusters) which is overkill for a local AI Agent or an embedded IDE copilot.
- **Too In-Memory:** Implementations like NanoGraphRAG rely on `NetworkX`, which breaks down or OOMs when your enterprise graph scales past 100,000 nodes and edges.
- **Naïve RAG Assumptions:** Traditional RAG assumes the ingested document is the "Ground Truth". In a real software pipeline, the design doc (e.g., Jira/Confluence/Wiki) is often outdated. The *only* ground truth is the current version of the code in the version control system (Git). 

## 💡 Core Design Philosophy

### 1. Zero-Infrastructure (Pure SQLite + NumPy)
We eliminate the need for Neo4j, Milvus, or Redis. 
- **Storage:** All nodes and edges are flattened into a single SQLite database (`kg_graph.db`). We use **SQL Recursive Common Table Expressions (CTEs)** to perform multi-hop traversals and approximate PageRank directly at the database level, meaning *zero Python memory overhead*.
- **Retrieval:** We extract the dense embeddings and sparse BM25 tokens into highly optimized, cached NumPy matrices upon startup. This allows for millisecond-level hybrid retrieval across 100k+ nodes without needing an external vector database.

### 2. The "Index-Only" Agentic Pattern 
**This is the most critical innovation of this project.**

Traditional RAG retrieves paragraphs of text and stuffs them into the LLM's context window. For an AI Agent operating on a massive codebase, this leads to **Context Pollution** and hallucination. 

We introduced the **`Index-Only` retrieval mode**. When an Agent queries the graph, the system *intentionally hides* the descriptive text. It only returns:
1. The Term/Entity Name
2. The Node Type (e.g., `CodeFile`, `BusinessTerm`)
3. The **Physical Asset Link** (e.g., `src/auth/manager.ts`, or a wiki URL).

**Why?** Because in industrial environments, production-grade content extremely dislikes errors. The Knowledge Graph is treated as a **Discovery Layer (Radar/Map)**. It gives the Agent the "pointer" to the file. The Agent must then use its own tools to read the *actual, current* file from the file system (the **Ground Truth Layer**). This strictly boundaries the trust model: the graph helps you *find* it, but you must read the actual code to *modify* it.

### 3. Enterprise Ingestion Principles (Code <-> Requirements)
Though the ingestion scripts are highly customized per company, the principles embedded in this architecture are universal:
- **Business Terms as Hubs:** We extract "Business Entities" from spec documents (e.g., "Authentication", "Matchmaking").
- **Asset Linkage:** We map these entities to actual code files via Version Control System (VCS) changelists. If a developer submits a commit fixing "Auth" in `auth_mgr.cpp`, an edge is created: `(Auth) -[IMPLEMENTED_IN]-> (auth_mgr.cpp)`.
- **Infrastructure Filtering:** Deep technical infra files (e.g., `string_utils.cpp`) are deliberately downgraded in the graph to prevent the LLM from getting distracted from the business logic.

## 🚀 Features

- **11 Native MCP Tools:** Exposes tools for Claude/Cursor including `kg_query`, `graph_index_only`, `kg_neighbors` (N-hop exploration), `kg_merge_nodes`, and `kg_tree_op`.
- **Hybrid Search:** Dense Vector + Sparse BM25 + Reciprocal Rank Fusion (RRF).
- **Repo Map Integration:** When using `Index-Only` mode, it can automatically attach a structural summary (AST Repo Map) of the located code files.
- **FastAPI Ready:** Runs as an HTTP or Stdio MCP server out of the box.
- **Web UI Included:** A React-based lab environment for visualizing the graph and managing the knowledge approval queue.

## 📦 Getting Started

### 1. Environment Requirements
- Python 3.10+
- Node.js 18+ (for Web UI build)

### 2. Installation
```bash
# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Build Frontend (Optional)
cd ui && npm install && npm run build && cd ..

# Configure Environment Variables
cp .env.example .env
```

### 3. Run the Server
```bash
python server.py
```
- **Web UI**: http://localhost:8000
- **MCP HTTP**: http://localhost:8001/mcp