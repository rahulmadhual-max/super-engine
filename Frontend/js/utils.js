/* ═══════════════════════════════════════════════════════════
   utils.js — Shared utilities: toast, API, auth
═══════════════════════════════════════════════════════════ */

// ── API base URL ─────────────────────────────────────────────
const API_URL =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:5000"
    : "https://your-backend-url.onrender.com"; // ← replace with your Render URL

// ── Toast notifications ──────────────────────────────────────
(function initToasts() {
  const container = document.createElement("div");
  container.id = "toast-container";
  document.body.appendChild(container);
})();

function showToast(message, type = "info", duration = 4000) {
  const icons = { success: "✓", error: "✕", info: "◆" };
  const container = document.getElementById("toast-container");

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || icons.info}</span>
    <span class="toast-msg">${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("hiding");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  }, duration);
}

// ── Token helpers ────────────────────────────────────────────
const Auth = {
  setToken(token)  { localStorage.setItem("ra_token", token); },
  getToken()       { return localStorage.getItem("ra_token"); },
  setName(name)    { localStorage.setItem("ra_name", name); },
  getName()        { return localStorage.getItem("ra_name") || ""; },
  setEmail(email)  { localStorage.setItem("ra_email", email); },
  getEmail()       { return localStorage.getItem("ra_email") || ""; },
  clear() {
    localStorage.removeItem("ra_token");
    localStorage.removeItem("ra_name");
    localStorage.removeItem("ra_email");
  },
  isLoggedIn() { return !!this.getToken(); },
  requireAuth() {
    if (!this.isLoggedIn()) {
      window.location.href = "login.html";
      return false;
    }
    return true;
  },
  logout() {
    this.clear();
    window.location.href = "login.html";
  },
};

// ── API fetch wrapper ────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const token = Auth.getToken();
  const headers = { ...(options.headers || {}) };

  // ✅ Always set JSON for non-FormData — not just when token exists
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  const data = await res.json();
  return { ok: res.ok, status: res.status, data };
}
