import { Link } from "react-router-dom";
import { useEffect, useState } from "react";


export default function Jogi() {
  const [msg, setMsg] = useState("");

  useEffect(() => {
    fetch("/api/jogi")
      .then((r) => r.text())
      .then(setMsg)
      .catch(() => setMsg("호출 실패"));
  }, []);

  return (
    <div>
      <h2>law</h2>
      <p>조기위험페이지</p>
      <a href="/">홈으로</a>
    </div>
  );
}
