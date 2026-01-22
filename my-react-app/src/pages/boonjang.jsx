import { Link } from "react-router-dom";
import { useEffect, useState } from "react";


export default function Boonjang() {
  const [msg, setMsg] = useState("");

  useEffect(() => {
    fetch("/api/boonjang")
      .then((r) => r.text())
      .then(setMsg)
      .catch(() => setMsg("호출 실패"));
  }, []);

  return (
    <div>
      <h2>boonjang</h2>
      <p>분쟁유형페이지</p>
      <a href="/">홈으로</a>
    </div>
  );
}
