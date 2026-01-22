import { Routes, Route, Link } from "react-router-dom";
import Yusa from "./pages/yusa.jsx";
import Law from "./pages/law.jsx";
import Jogi from "./pages/jogi.jsx";
import Boonjang from "./pages/boonjang.jsx";

export default function App() {
  return (
    <div style={{ padding: 24 }}>
      <nav style={{ display: "flex", gap: 12 }}>
        <Link to="/law">법적위험페이지</Link>
        <Link to="/boonjang">분쟁유형페이지</Link>
        <Link to="/yusa">유사판례페이지</Link>
        <Link to="/jogi">조기위험페이지</Link>
      </nav>

      <Routes>
        <Route path="/law" element={<Law />} />
        <Route path="/boonjang" element={<Boonjang />} />
        <Route path="/yusa" element={<Yusa />} />
        <Route path="/jogi" element={<Jogi />} />

      </Routes>
    </div>
  );
}
