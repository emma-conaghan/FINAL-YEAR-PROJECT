// Simple, readable vanilla JS (realistic for a final-year project PoC)

const $ = (id) => document.getElementById(id);

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function escapeHTML(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  }[m]));
}

function normalizeSeverity(sev) {
  if (!sev) return "UNKNOWN";
  const u = String(sev).toUpperCase();
  if (u.includes("CRIT") || u.includes("HIGH") || u.includes("ERROR")) return "HIGH";
  if (u.includes("MED") || u.includes("WARN")) return "MEDIUM";
  if (u.includes("LOW")) return "LOW";
  if (u.includes("INFO")) return "INFO";
  return "UNKNOWN";
}

function setRing(el, score, colorVar) {
  el.style.setProperty("--pct", clamp(score, 0, 100));
  el.style.setProperty("--ringColor", colorVar);
}

function gradeFromScore(s) {
  if (s >= 90) return ["Strong", "var(--good)"];
  if (s >= 75) return ["Good", "var(--warn)"];
  return ["Needs work", "var(--bad)"];
}

// PoC scoring:
// Start at 100 and penalise based on Semgrep severity counts.
function scoreFromCounts(counts) {
  const high = counts.HIGH || 0;
  const med  = counts.MEDIUM || 0;
  const low  = counts.LOW || 0;
  const info = counts.INFO || 0;

  const security = clamp(100 - (high * 20 + med * 10 + low * 3 + info * 1), 0, 100);

  // Placeholder category scores (so UI looks Lighthouse-y, but honest as PoC)
  const maintain = clamp(90 - (high + med + low + info) * 2, 40, 100);
  const reliability = clamp(92 - (high * 6 + med * 3 + low * 1), 40, 100);
  const style = clamp(95 - (low * 2 + info * 1), 50, 100);

  const overall = Math.round(security * 0.55 + maintain * 0.20 + reliability * 0.15 + style * 0.10);

  return { overall, security, maintain, reliability, style };
}

function parseSemgrepJSON(obj) {
  const results = Array.isArray(obj?.results) ? obj.results : [];

  const findings = results.map(r => ({
    severity: normalizeSeverity(r?.extra?.severity),
    rule: r?.check_id || r?.rule_id || "unknown-rule",
    message: r?.extra?.message || r?.message || "(no message)",
    path: r?.path || r?.extra?.path || "(unknown)",
    start: r?.start ? `${r.start.line}:${r.start.col}` : ""
  }));

  const counts = findings.reduce((acc, f) => {
    acc.total++;
    acc[f.severity] = (acc[f.severity] || 0) + 1;
    return acc;
  }, { total: 0 });

  return { findings, counts };
}

let state = {
  findings: [],
  counts: { total: 0 },
  target: "—"
};

function updateHeader(scores) {
  $("overallScore").textContent = scores.overall;
  const [overallGrade, overallColor] = gradeFromScore(scores.overall);
  setRing($("ringOverall"), scores.overall, overallColor);

  $("secScore").textContent = scores.security;
  const [secGrade, secColor] = gradeFromScore(scores.security);
  $("secGrade").textContent = secGrade;
  $("secGrade").style.borderColor = secColor;

  // Small rings
  $("ringSecNum").textContent = scores.security;
  $("ringMaintNum").textContent = scores.maintain;
  $("ringRelNum").textContent = scores.reliability;
  $("ringStyleNum").textContent = scores.style;

  setRing($("ringSec"), scores.security, gradeFromScore(scores.security)[1]);
  setRing($("ringMaint"), scores.maintain, gradeFromScore(scores.maintain)[1]);
  setRing($("ringRel"), scores.reliability, gradeFromScore(scores.reliability)[1]);
  setRing($("ringStyle"), scores.style, gradeFromScore(scores.style)[1]);

  // Meta counts line
  const c = state.counts;
  const parts = ["HIGH", "MEDIUM", "LOW", "INFO", "UNKNOWN"]
    .filter(k => c[k])
    .map(k => `${k}:${c[k]}`);

  $("counts").textContent = `${c.total || 0} total` + (parts.length ? ` (${parts.join(", ")})` : "");
}

function renderTable() {
  const q = $("search").value.trim().toLowerCase();
  const sev = $("sevFilter").value;

  let rows = state.findings.slice();

  if (sev !== "all") rows = rows.filter(f => f.severity === sev);

  if (q) {
    rows = rows.filter(f =>
      (f.rule || "").toLowerCase().includes(q) ||
      (f.message || "").toLowerCase().includes(q) ||
      (f.path || "").toLowerCase().includes(q)
    );
  }

  $("tbody").innerHTML = rows.map(f => {
    const cls = f.severity === "HIGH" ? "high" : f.severity === "MEDIUM" ? "med" : f.severity === "LOW" ? "low" : "";
    const loc = `${f.path}${f.start ? ` @ ${f.start}` : ""}`;
    return `
      <tr>
        <td>
          <span class="sev ${cls}">
            <span class="b"></span>${escapeHTML(f.severity)}
          </span>
        </td>
        <td class="code">${escapeHTML(f.rule)}</td>
        <td>${escapeHTML(f.message)}</td>
        <td class="code">${escapeHTML(loc)}</td>
      </tr>
    `;
  }).join("");

  $("empty").style.display = rows.length ? "none" : "block";
}

function applyReport(targetLabel) {
  $("stamp").textContent = new Date().toLocaleString();
  $("runId").textContent = "poc-" + Math.random().toString(16).slice(2, 6);
  $("target").textContent = targetLabel || state.target;

  const scores = scoreFromCounts(state.counts);
  updateHeader(scores);
  renderTable();
}

function loadDemo() {
  state.findings = [
    { severity: "HIGH", rule: "python.lang.security.audit.eval-used", message: "Use of eval() can lead to arbitrary code execution.", path: "sample/workflow_engine_vuln.py", start: "311:12" },
    { severity: "HIGH", rule: "python.lang.security.audit.subprocess-shell-true", message: "subprocess with shell=True can enable command injection.", path: "sample/workflow_engine_vuln.py", start: "316:12" },
    { severity: "MEDIUM", rule: "python.lang.security.audit.pickle-loads", message: "pickle.loads on untrusted data can lead to code execution.", path: "sample/workflow_engine_vuln.py", start: "320:12" }
  ];

  state.counts = state.findings.reduce((acc, f) => {
    acc.total++;
    acc[f.severity] = (acc[f.severity] || 0) + 1;
    return acc;
  }, { total: 0 });

  state.target = "demo_semgrep.json";
  applyReport(state.target);
}

function exportReport() {
  const scores = scoreFromCounts(state.counts);
  const payload = {
    generated_at: new Date().toISOString(),
    target: state.target,
    tool: "Semgrep (auto rules)",
    counts: state.counts,
    scores,
    findings: state.findings
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "vibecode_report.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// Wire up events
$("search").addEventListener("input", renderTable);
$("sevFilter").addEventListener("change", renderTable);

$("btnDemo").addEventListener("click", loadDemo);
$("btnExport").addEventListener("click", exportReport);

$("fileInput").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  $("fileName").textContent = file.name;

  try {
    const obj = JSON.parse(await file.text());
    const parsed = parseSemgrepJSON(obj);

    state.findings = parsed.findings;
    state.counts = parsed.counts;
    state.target = file.name;

    applyReport(file.name);
  } catch (err) {
    console.error(err);
    alert("Could not parse JSON. Make sure you selected Semgrep JSON output.");
  }
});

// Initial view
$("stamp").textContent = new Date().toLocaleString();
loadDemo();
