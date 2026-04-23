/* ═══════════════════════════════════════════════════════════
   utils.js — Shared utilities: toast, API, auth
═══════════════════════════════════════════════════════════ */

// ── API base URL ─────────────────────────────────────────────
const API_URL = "https://ai-resume-analyzer-9tnm.onrender.com";

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

  isLoggedIn() {
    return !!this.getToken();
  },

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

  // Set JSON header only if not FormData
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  // Add auth token if exists
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    const res = await fetch(`${API_URL}${path}`, {
      ...options,
      headers,
    });

    let data;
    try {
      data = await res.json();
    } catch {
      data = { error: "Invalid JSON response" };
    }

    return {
      ok: res.ok,
      status: res.status,
      data,
    };

  } catch (err) {
    return {
      ok: false,
      status: 0,
      data: {
        error: "Network error. Please check your internet connection.",
      },
    };
  }
}