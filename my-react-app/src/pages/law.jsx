import { Link } from "react-router-dom";
import { useEffect, useState } from "react";


export default function Law() {
  const [msg, setMsg] = useState("");

  useEffect(() => {
    fetch("/api/law")
      .then((r) => r.text())
      .then(setMsg)
      .catch(() => setMsg("호출 실패"));
  }, []);

  return (
    <div>
      <h2>law</h2>
      <p>법적위험페이지</p>
      <a href="/">홈으로</a>
    </div>
  );
}
