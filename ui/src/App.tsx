import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import ApprovalQueue from "./pages/ApprovalQueue";
import GraphExplorer from "./pages/GraphExplorer";
import ProposalMonitor from "./pages/ProposalMonitor";
import QueryLab from "./pages/QueryLab";
import "./App.css";

function App() {
  return (
    <Router>
      <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <nav
          style={{
            padding: "15px 20px",
            backgroundColor: "#333",
            color: "white",
            display: "flex",
            gap: "20px",
            alignItems: "center",
          }}
        >
          <h1 style={{ margin: 0, marginRight: "20px" }}>OpenGraph Agent Admin</h1>
          <Link to="/" style={{ color: "white", textDecoration: "none" }}>
            图谱探索
          </Link>
          <Link to="/approval" style={{ color: "white", textDecoration: "none" }}>
            审批队列
          </Link>
          <Link to="/proposals" style={{ color: "white", textDecoration: "none" }}>
            执行监控
          </Link>
          <Link to="/query-lab" style={{ color: "white", textDecoration: "none" }}>
            查询实验
          </Link>
        </nav>

        <div style={{ flex: 1, overflow: "hidden" }}>
          <Routes>
            <Route path="/" element={<GraphExplorer />} />
            <Route path="/approval" element={<ApprovalQueue />} />
            <Route path="/proposals" element={<ProposalMonitor />} />
            <Route path="/query-lab" element={<QueryLab />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
