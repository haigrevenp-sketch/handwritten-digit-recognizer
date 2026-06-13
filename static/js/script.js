/**
 * ================================================================
 *  Digit Recognizer — script.js
 *  Handles: canvas drawing, file upload, API calls,
 *           result rendering, history, charts, neural canvas,
 *           dark/light toggle, real-time prediction.
 * ================================================================
 */

"use strict";

/* ─────────────────────────────────────────────────────────────
   Utilities
───────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

function showToast(msg, type = "info") {
  const toast = $("toastEl");
  const body  = $("toastBody");
  if (!toast || !body) return;
  body.textContent = msg;
  toast.className = "toast align-items-center border-0 glass-toast";
  if (type === "success") toast.style.borderLeftColor = "#06d6a0";
  else if (type === "error") toast.style.borderLeftColor = "#ff4757";
  new bootstrap.Toast(toast, { delay: 3500 }).show();
}

function showSpinner(show) {
  const el = $("spinnerOverlay");
  if (el) el.classList.toggle("d-none", !show);
}

/* ─────────────────────────────────────────────────────────────
   Theme Toggle
───────────────────────────────────────────────────────────── */
(function initTheme() {
  const btn   = $("themeToggle");
  const saved = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  updateThemeBtn(saved);

  if (btn) btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next    = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    updateThemeBtn(next);
  });

  function updateThemeBtn(theme) {
    if (!btn) return;
    btn.innerHTML = theme === "dark"
      ? '<i class="bi bi-sun-fill"></i>'
      : '<i class="bi bi-moon-stars-fill"></i>';
  }
})();

/* ─────────────────────────────────────────────────────────────
   Neural Network Canvas (hero background)
───────────────────────────────────────────────────────────── */
(function initNeuralCanvas() {
  const canvas = $("neuralCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let W, H, nodes, edges;
  const N = 60;

  function resize() {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  function makeNodes() {
    nodes = Array.from({ length: N }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: 2 + Math.random() * 2,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const theme = document.documentElement.getAttribute("data-theme");
    const nodeColor = theme === "light" ? "rgba(0,120,200,0.6)" : "rgba(0,212,255,0.7)";
    const lineColor = theme === "light" ? "rgba(0,120,200,0.12)" : "rgba(0,212,255,0.12)";

    // move
    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });

    // edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < 140) {
          ctx.beginPath();
          ctx.strokeStyle = lineColor;
          ctx.globalAlpha = 1 - d / 140;
          ctx.lineWidth = 0.8;
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;

    // nodes
    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = nodeColor;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  resize();
  makeNodes();
  draw();
  window.addEventListener("resize", () => { resize(); makeNodes(); });
})();

/* ─────────────────────────────────────────────────────────────
   Showcase digit cycler
───────────────────────────────────────────────────────────── */
(function initShowcase() {
  const el = $("showcaseNum");
  if (!el) return;
  const digits = [0,1,2,3,4,5,6,7,8,9];
  let i = 7;
  setInterval(() => {
    el.style.opacity = "0";
    setTimeout(() => {
      i = (i + 1) % 10;
      el.textContent = digits[i];
      el.style.opacity = "1";
    }, 300);
  }, 1800);
})();

/* ─────────────────────────────────────────────────────────────
   Drawing Canvas
───────────────────────────────────────────────────────────── */
(function initCanvas() {
  const canvas = $("drawCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let painting  = false;
  let history   = [];          // undo stack (array of ImageData)
  let rtTimer   = null;

  // white pen on black bg
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#ffffff";
  ctx.lineJoin    = "round";
  ctx.lineCap     = "round";
  ctx.lineWidth   = 18;

  // brush size
  const brushSlider = $("brushSize");
  const brushVal    = $("brushVal");
  if (brushSlider) {
    brushSlider.addEventListener("input", () => {
      ctx.lineWidth = brushSlider.value;
      if (brushVal) brushVal.textContent = brushSlider.value;
    });
  }

  function getPos(e) {
    const r = canvas.getBoundingClientRect();
    const src = e.touches ? e.touches[0] : e;
    return {
      x: (src.clientX - r.left) * (canvas.width  / r.width),
      y: (src.clientY - r.top)  * (canvas.height / r.height),
    };
  }

  function saveHistory() {
    history.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (history.length > 20) history.shift();
  }

  function startPaint(e) {
    e.preventDefault();
    saveHistory();
    painting = true;
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  }
  function paint(e) {
    e.preventDefault();
    if (!painting) return;
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    // trigger real-time prediction
    if ($("realtimeToggle")?.checked) {
      clearTimeout(rtTimer);
      rtTimer = setTimeout(doRealtimePrediction, 400);
    }
  }
  function stopPaint() { painting = false; ctx.beginPath(); }

  canvas.addEventListener("mousedown",  startPaint);
  canvas.addEventListener("mousemove",  paint);
  canvas.addEventListener("mouseup",    stopPaint);
  canvas.addEventListener("mouseleave", stopPaint);
  canvas.addEventListener("touchstart", startPaint, { passive: false });
  canvas.addEventListener("touchmove",  paint,      { passive: false });
  canvas.addEventListener("touchend",   stopPaint);

  // Clear
  $("clearBtn")?.addEventListener("click", () => {
    saveHistory();
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hideResultPanel("drawResults");
    clearRealtime();
  });

  // Undo
  $("undoBtn")?.addEventListener("click", () => {
    if (history.length) ctx.putImageData(history.pop(), 0, 0);
  });

  // Predict button
  $("predictDrawBtn")?.addEventListener("click", () => {
    const b64 = canvas.toDataURL("image/png");
    runPrediction("/api/predict/draw", JSON.stringify({ image: b64 }),
                  "drawResults", "resultPanel", "placeholderCard");
  });

  // Real-time prediction
  async function doRealtimePrediction() {
    try {
      const b64  = canvas.toDataURL("image/png");
      const resp = await fetch("/api/predict/realtime", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: b64 }),
      });
      const data = await resp.json();
      renderRealtime(data);
    } catch (_) {}
  }

  function renderRealtime(data) {
    const rtDigit = $("rtDigit");
    const rtBars  = $("rtBars");
    if (rtDigit) rtDigit.textContent = data.predicted ?? "—";
    if (!rtBars || !data.top3) return;
    rtBars.innerHTML = "";
    const max = data.top3[0].probability;
    data.top3.forEach(t => {
      const b = document.createElement("div");
      b.className = "rt-bar";
      b.style.height = Math.max(2, (t.probability / max) * 18) + "px";
      b.title = `${t.digit}: ${t.probability}%`;
      rtBars.appendChild(b);
    });
  }

  function clearRealtime() {
    const rd = $("rtDigit"); if (rd) rd.textContent = "—";
    const rb = $("rtBars");  if (rb) rb.innerHTML = "";
  }
})();

/* ─────────────────────────────────────────────────────────────
   File Upload
───────────────────────────────────────────────────────────── */
(function initUpload() {
  const dropzone = $("dropzone");
  const fileInput = $("fileInput");
  const predictBtn = $("predictUploadBtn");
  const dzIdle    = $("dzIdle");
  const dzPreview = $("dzPreview");
  const previewImg = $("previewImg");
  const dzFilename = $("dzFilename");

  if (!dropzone) return;

  // drag-over styling
  ["dragenter","dragover"].forEach(ev =>
    dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add("drag-over"); })
  );
  ["dragleave","drop"].forEach(ev =>
    dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove("drag-over"); })
  );

  dropzone.addEventListener("drop", e => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.type.match(/image\/(png|jpeg)/)) {
      showToast("Please upload a PNG or JPG file.", "error"); return;
    }
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      dzFilename.textContent = file.name;
      dzIdle.classList.add("d-none");
      dzPreview.classList.remove("d-none");
      predictBtn.disabled = false;
      // store file ref
      predictBtn._file = file;
    };
    reader.readAsDataURL(file);
  }

  predictBtn.addEventListener("click", () => {
    const file = predictBtn._file;
    if (!file) return;
    const fd = new FormData();
    fd.append("image", file);
    runPredictionForm("/api/predict/upload", fd,
                      "uploadResults", "uploadResultPanel", "uploadPlaceholderCard");
  });
})();

/* ─────────────────────────────────────────────────────────────
   Prediction runner (JSON body)
───────────────────────────────────────────────────────────── */
async function runPrediction(url, body, resultContainerId, panelId, placeholderId) {
  showSpinner(true);
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const data = await resp.json();
    if (data.error) { showToast(data.error, "error"); return; }
    renderResult(data, panelId, placeholderId);
    showToast(`Predicted: ${data.predicted}  (${data.confidence}% confidence)`, "success");
    loadHistory();
  } catch (err) {
    showToast("Prediction failed. Is Flask running?", "error");
    console.error(err);
  } finally {
    showSpinner(false);
  }
}

/* ─────────────────────────────────────────────────────────────
   Prediction runner (FormData)
───────────────────────────────────────────────────────────── */
async function runPredictionForm(url, formData, resultContainerId, panelId, placeholderId) {
  showSpinner(true);
  try {
    const resp = await fetch(url, { method: "POST", body: formData });
    const data = await resp.json();
    if (data.error) { showToast(data.error, "error"); return; }
    renderResult(data, panelId, placeholderId);
    showToast(`Predicted: ${data.predicted}  (${data.confidence}% confidence)`, "success");
    loadHistory();
  } catch (err) {
    showToast("Prediction failed. Is Flask running?", "error");
    console.error(err);
  } finally {
    showSpinner(false);
  }
}

/* ─────────────────────────────────────────────────────────────
   Render Result Panel
───────────────────────────────────────────────────────────── */
function renderResult(data, panelId, placeholderId) {
  const panel       = $(panelId);
  const placeholder = $(placeholderId);
  if (!panel) return;

  if (placeholder) placeholder.classList.add("d-none");
  panel.classList.remove("d-none");

  const top3HTML = (data.top3 || []).map((t, i) => `
    <div class="top3-row">
      <span class="top3-rank">#${i+1}</span>
      <div class="top3-digit-badge ${i===0 ? 'first' : ''}">${t.digit}</div>
      <div class="top3-bar-wrap">
        <div class="top3-bar-fill" style="width:${t.probability}%"></div>
      </div>
      <span class="top3-pct">${t.probability.toFixed(1)}%</span>
    </div>
  `).join("");

  const gradcamSection = data.gradcam ? `
    <div class="mt-3">
      <div class="card-label mb-2"><i class="bi bi-eye me-2 accent"></i>Grad-CAM Heatmap</div>
      <img src="data:image/png;base64,${data.gradcam}" class="gradcam-img w-100" alt="Grad-CAM">
    </div>
  ` : "";

  const chartSection = data.chart ? `
    <div class="mt-3">
      <div class="card-label mb-2"><i class="bi bi-bar-chart me-2 accent"></i>Confidence per Class</div>
      <img src="data:image/png;base64,${data.chart}" class="chart-img" alt="Confidence Chart">
    </div>
  ` : "";

  // grab the last prediction id from history for PDF link (will be updated after loadHistory)
  panel.innerHTML = `
    <div class="glass-card result-card">
      <div class="text-center mb-3">
        <div class="big-digit-badge">${data.predicted}</div>
        <h4 class="mb-1">Predicted: <span class="gradient-text mono">${data.predicted}</span></h4>
        <p class="section-sub mb-2">Confidence: <strong>${data.confidence.toFixed(1)}%</strong></p>
        <div class="confidence-bar-wrap mb-3 mx-auto" style="max-width:300px">
          <div class="confidence-bar-fill" style="width:0%" data-target="${data.confidence}"></div>
        </div>
      </div>

      <div class="card-label mb-2"><i class="bi bi-list-ol me-2 accent"></i>Top 3 Predictions</div>
      ${top3HTML}

      ${gradcamSection}
      ${chartSection}

      <div class="d-flex gap-2 mt-3">
        <button class="btn btn-outline-glow flex-grow-1" id="dlReportBtn_${panelId}" onclick="downloadLatestReport()">
          <i class="bi bi-file-pdf me-2"></i>Download Report
        </button>
      </div>
    </div>
  `;

  // animate confidence bar
  requestAnimationFrame(() => {
    const bar = panel.querySelector(".confidence-bar-fill");
    if (bar) {
      setTimeout(() => bar.style.width = bar.dataset.target + "%", 50);
    }
  });
}

/* ─────────────────────────────────────────────────────────────
   Hide result panel
───────────────────────────────────────────────────────────── */
function hideResultPanel(containerId) {
  const container = $(containerId);
  if (!container) return;
  const panel = container.querySelector("[id$='ResultPanel'], #resultPanel");
  if (panel) panel.classList.add("d-none");
  const ph = container.querySelector("[id$='PlaceholderCard'], #placeholderCard");
  if (ph) ph.classList.remove("d-none");
}

/* ─────────────────────────────────────────────────────────────
   Prediction History
───────────────────────────────────────────────────────────── */
let _latestPredId = null;

async function loadHistory() {
  const tbody = $("historyTbody");
  if (!tbody) return;
  try {
    const resp = await fetch("/api/history");
    const rows = await resp.json();
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-center opacity-50 py-4">No predictions yet.</td></tr>`;
      return;
    }
    _latestPredId = rows[0].id;
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td class="mono opacity-60">${r.id}</td>
        <td class="mono small">${r.timestamp}</td>
        <td><span class="badge bg-secondary rounded-pill text-capitalize">${r.source}</span></td>
        <td><div class="digit-pill">${r.predicted}</div></td>
        <td>
          <div class="d-flex align-items-center gap-2">
            <div style="width:60px;height:6px;background:rgba(255,255,255,0.1);border-radius:6px;overflow:hidden">
              <div style="width:${r.confidence}%;height:100%;background:var(--accent);border-radius:6px"></div>
            </div>
            <span class="mono small">${r.confidence.toFixed(1)}%</span>
          </div>
        </td>
        <td class="small mono">${(r.top3 || []).map(t => `${t.digit}(${t.probability}%)`).join(', ')}</td>
        <td>
          <a href="/api/report/${r.id}" class="btn-icon" title="Download PDF" target="_blank">
            <i class="bi bi-file-pdf"></i>
          </a>
        </td>
      </tr>
    `).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center opacity-50 py-4">Could not load history.</td></tr>`;
  }
}

function downloadLatestReport() {
  if (_latestPredId) window.open(`/api/report/${_latestPredId}`, "_blank");
  else showToast("No prediction saved yet.", "error");
}

$("refreshHistory")?.addEventListener("click", loadHistory);

/* ─────────────────────────────────────────────────────────────
   Performance Charts (Chart.js)
───────────────────────────────────────────────────────────── */
(function initCharts() {
  const stats = window.MODEL_STATS || {};
  const epochs = stats.acc_history?.length || 12;
  const labels = Array.from({ length: epochs }, (_, i) => i + 1);

  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { labels: { color: "#aaa", font: { size: 11 } } },
      tooltip: {
        backgroundColor: "rgba(13,13,31,0.92)",
        titleColor: "#fff", bodyColor: "#aaa",
        borderColor: "rgba(255,255,255,0.1)", borderWidth: 1,
      },
    },
    scales: {
      x: { ticks: { color: "#aaa" }, grid: { color: "rgba(255,255,255,0.05)" } },
      y: { ticks: { color: "#aaa" }, grid: { color: "rgba(255,255,255,0.05)" } },
    },
  };

  // Accuracy
  const accCtx = $("accChart");
  if (accCtx) {
    new Chart(accCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Train Acc (%)",
            data: stats.acc_history || [],
            borderColor: "#00d4ff", backgroundColor: "rgba(0,212,255,0.08)",
            tension: 0.4, fill: true, pointRadius: 3,
          },
          {
            label: "Val Acc (%)",
            data: stats.val_acc_history || [],
            borderColor: "#f72585", backgroundColor: "rgba(247,37,133,0.06)",
            tension: 0.4, fill: true, pointRadius: 3, borderDash: [5,5],
          },
        ],
      },
      options: { ...chartDefaults },
    });
  }

  // Loss
  const lossCtx = $("lossChart");
  if (lossCtx) {
    new Chart(lossCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Train Loss",
            data: stats.loss_history || [],
            borderColor: "#00d4ff", backgroundColor: "rgba(0,212,255,0.08)",
            tension: 0.4, fill: true, pointRadius: 3,
          },
          {
            label: "Val Loss",
            data: stats.val_loss_history || [],
            borderColor: "#f72585", backgroundColor: "rgba(247,37,133,0.06)",
            tension: 0.4, fill: true, pointRadius: 3, borderDash: [5,5],
          },
        ],
      },
      options: { ...chartDefaults },
    });
  }
})();

/* ─────────────────────────────────────────────────────────────
   Navbar scroll effect
───────────────────────────────────────────────────────────── */
(function initNavScroll() {
  const nav = $("mainNav");
  if (!nav) return;
  const onScroll = () => {
    nav.style.background = window.scrollY > 60
      ? "rgba(5,5,16,0.95)"
      : "rgba(5,5,16,0.75)";
  };
  window.addEventListener("scroll", onScroll, { passive: true });
})();

/* ─────────────────────────────────────────────────────────────
   Smooth scroll for anchor links
───────────────────────────────────────────────────────────── */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", e => {
    const target = document.querySelector(a.getAttribute("href"));
    if (target) {
      e.preventDefault();
      const navH = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--nav-h")) || 68;
      window.scrollTo({ top: target.offsetTop - navH - 16, behavior: "smooth" });
    }
  });
});

/* ─────────────────────────────────────────────────────────────
   Bootstrap tab sync for custom tabs
───────────────────────────────────────────────────────────── */
(function initTabs() {
  const ctabs = document.querySelectorAll(".ctab");
  ctabs.forEach(btn => {
    btn.addEventListener("shown.bs.tab", () => {
      ctabs.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
    btn.addEventListener("click", () => {
      ctabs.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
})();

/* ─────────────────────────────────────────────────────────────
   On page load
───────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  loadHistory();
});
