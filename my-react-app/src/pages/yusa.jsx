import { Link } from "react-router-dom";
import { useEffect, useState } from "react";


export default function Yusa() {
  const [msg, setMsg] = useState("");

  useEffect(() => {
    fetch("/api/yusa")
      .then((r) => r.text())
      .then(setMsg)
      .catch(() => setMsg("호출 실패"));
  }, []);

  return (
    <div>
      <h2>yusa</h2>
      <p>유사판례페이지</p>
      <a href="/">홈으로</a>
    </div>
  );
}
