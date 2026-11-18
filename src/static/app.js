"use strict";

const TOKEN = new URLSearchParams(location.search).get("token") || "";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;       // textContent only: no HTML injection
  return e;
};

let state = { services: [], groups: [], robots: {}, topics: {}, ros: false };
let selected = null;
const logs = {};                                       // name -> [lines]
const MAX_LINES = 1000;
let lastSvcSig = "";

// ------------------------------------------------------------------ api
async function api(path, method = "POST", body) {
  const sep = path.includes("?") ? "&" : "?";
  const url = TOKEN ? `${path}${sep}token=${encodeURIComponent(TOKEN)}` : path;
  const res = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const t = await res.text();
    alert(`${method} ${path}: ${res.status} ${t}`);
    return null;
  }
  return res.json();
}

// ------------------------------------------------------------------ render
function renderServices() {
  const sig = JSON.stringify(state.services.map(s => [s.name, s.status, s.pid, Math.floor(s.uptime / 5)])) + selected;
  if (sig === lastSvcSig) return;
  lastSvcSig = sig;
  const root = $("groups");
  root.replaceChildren();
  for (const g of state.groups) {
    const box = el("div", "group");
    const head = el("div", "group-head");
    head.append(el("strong", null, g));
    const goAll = el("button", "small go", "start all");
    goAll.onclick = () => api(`/api/groups/${encodeURIComponent(g)}/start`);
    const stopAll = el("button", "small stop", "stop all");
    stopAll.onclick = () => api(`/api/groups/${encodeURIComponent(g)}/stop`);
    head.append(goAll, stopAll);
    box.append(head);

    for (const s of state.services.filter(x => x.group === g)) {
      const row = el("div", "svc" + (s.name === selected ? " selected" : ""));
      row.append(el("span", "dot " + s.status));
      const info = el("div", "info");
      info.append(el("div", "name", s.name));
      info.append(el("div", "desc", s.description || s.cmd));
      row.append(info);
      row.append(el("span", "meta", s.status === "running" ? `${s.uptime}s` : s.status));
      const btns = el("div", "btns");
      const b = (label, cls, act) => {
        const x = el("button", "small " + cls, label);
        x.onclick = (ev) => { ev.stopPropagation(); api(`/api/services/${encodeURIComponent(s.name)}/${act}`); };
        return x;
      };
      btns.append(b("start", "go", "start"), b("stop", "stop", "stop"), b("restart", "restart", "restart"));
      row.append(btns);
      row.onclick = () => selectService(s.name);
      box.append(row);
    }
    root.append(box);
  }
}

function renderTopics() {
  const root = $("topics");
  root.replaceChildren();
  for (const [t, n] of Object.entries(state.topics || {})) {
    const c = el("span", "chip");
    c.append(el("span", "dot " + (n > 0 ? "ok" : "bad")), el("span", null, t));
    root.append(c);
  }
}

function renderFleet() {
  const tb = document.querySelector("#fleet tbody");
  tb.replaceChildren();
  for (const [r, d] of Object.entries(state.robots || {})) {
    const tr = el("tr");
    const dotCell = (ok, label) => {
      const td = el("td");
      td.append(el("span", "dot " + (ok ? "ok" : "bad")), document.createTextNode(" " + label));
      return td;
    };
    tr.append(el("td", null, r));
    tr.append(dotCell(d.skill_alive, d.skill_alive ? (d.skill_state || "alive") : "down"));
    tr.append(dotCell(d.agent_alive, d.agent_alive ? "alive" : "down"));
    tr.append(el("td", null, d.planner_status + (d.holding ? " (holding)" : "")));
    tr.append(el("td", null, d.goal || ""));
    const td = el("td");
    const stop = el("button", "small stop", "stop");
    stop.onclick = () => api("/api/robots/stop", "POST", { robot: r });
    td.append(stop);
    tr.append(td);
    tb.append(tr);
  }
}

function renderHeader() {
  $("title").textContent = state.title || "Robot Dashboard";
  const ros = $("rosflag");
  ros.textContent = state.ros ? "ROS on" : "ROS off";
  ros.className = "pill " + (state.ros ? "on" : "off");
}

function renderAll() { renderHeader(); renderServices(); renderTopics(); renderFleet(); }

// ------------------------------------------------------------------ logs
function paintLog() {
  const pre = $("log");
  pre.textContent = (logs[selected] || []).join("\n");
  if ($("autoscroll").checked) pre.scrollTop = pre.scrollHeight;
}

async function selectService(name) {
  selected = name;
  lastSvcSig = "";
  $("logName").textContent = name;
  const r = await api(`/api/logs/${encodeURIComponent(name)}?n=300`, "GET");
  if (r) logs[name] = r.lines;
  paintLog();
  renderServices();
}

function pushLog(name, line) {
  const a = (logs[name] = logs[name] || []);
  a.push(line);
  if (a.length > MAX_LINES) a.splice(0, a.length - MAX_LINES);
  if (name === selected) {
    const pre = $("log");
    pre.append(document.createTextNode((pre.textContent ? "\n" : "") + line));
    if ($("autoscroll").checked) pre.scrollTop = pre.scrollHeight;
  }
}

// ------------------------------------------------------------------ chat
function addChat(m) {
  const box = $("chat");
  const row = el("div", "msg " + m.sender);
  row.append(el("span", "t", m.t || ""), el("b", null, m.sender + ": "), document.createTextNode(m.text));
  box.append(row);
  while (box.children.length > 200) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}

// ------------------------------------------------------------------ websocket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws${TOKEN ? "?token=" + encodeURIComponent(TOKEN) : ""}`);
  ws.onopen = () => { $("conn").textContent = "online"; $("conn").className = "pill on"; };
  ws.onclose = () => { $("conn").textContent = "offline"; $("conn").className = "pill off"; setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "state") { state = m; renderAll(); }
    else if (m.type === "log") pushLog(m.name, m.line);
    else if (m.type === "chat") addChat(m);
    else if (m.type === "svc") {
      const i = state.services.findIndex(s => s.name === m.name);
      if (i >= 0) { state.services[i] = m; lastSvcSig = ""; renderServices(); }
    }
  };
}

// ------------------------------------------------------------------ buttons
$("btnStopRobots").onclick = () => api("/api/robots/stop", "POST", {});
$("btnKillAll").onclick = () => { if (confirm("Stop every service on this machine?")) api("/api/stop_all_services"); };
$("btnStartAll").onclick = () => { if (confirm("Start every service on this machine, in order?")) api("/api/start_all_services"); };
$("btnClearLog").onclick = () => { if (selected) { logs[selected] = []; paintLog(); } };
$("chatForm").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("chatInput").value.trim();
  if (!text) return;
  $("chatInput").value = "";
  await api("/api/chat", "POST", { text });
};

connect();
