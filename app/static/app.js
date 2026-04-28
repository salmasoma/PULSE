const form = document.getElementById("analysis-form");
const submitButton = document.getElementById("submit-button");
const submitStatus = document.getElementById("submit-status");
const resultSummary = document.getElementById("result-summary");
const reasonedReport = document.getElementById("reasoned-report");
const resultDetails = document.getElementById("result-details");
const taskTableBody = document.getElementById("task-table-body");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function statusClassName(status) {
  return `status-${status || "unknown"}`;
}

function formatValue(value) {
  if (typeof value === "number") {
    return value.toFixed(4);
  }
  if (Array.isArray(value)) {
    if (value.length > 6) {
      return `[${value.slice(0, 4).map(formatValue).join(", ")}, ...]`;
    }
    return `[${value.map(formatValue).join(", ")}]`;
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function formatPlan(plan) {
  if (!plan || plan.length === 0) {
    return '<p class="empty-state">No specialist task was scheduled.</p>';
  }
  return `<div class="plan-chips">${plan.map((step) => `<span class="chip">${escapeHtml(step)}</span>`).join("")}</div>`;
}

function setReasoningPlaceholder(message) {
  reasonedReport.className = "reasoned-report empty-state";
  reasonedReport.textContent = message;
}

function renderSummary(payload) {
  const quality = payload.quality || {};
  resultSummary.classList.remove("empty-state");
  resultSummary.innerHTML = `
    <div class="summary-grid">
      <div class="summary-tile">
        <span>Detected domain</span>
        <strong>${escapeHtml(payload.detected_domain || "unknown")}</strong>
      </div>
      <div class="summary-tile">
        <span>Routed anatomy</span>
        <strong>${escapeHtml(payload.detected_anatomy || payload.routing_label || "unknown")}</strong>
      </div>
      <div class="summary-tile">
        <span>Request ID</span>
        <strong>${escapeHtml(payload.request_id)}</strong>
      </div>
      <div class="summary-tile">
        <span>Image quality</span>
        <strong>${escapeHtml(quality.quality_label || "unknown")}</strong>
      </div>
      <div class="summary-tile">
        <span>Routing model</span>
        <strong>${escapeHtml(payload.routing_task_id || "none")}</strong>
      </div>
      <div class="summary-tile">
        <span>Focus</span>
        <strong>${payload.intents && payload.intents.length ? escapeHtml(payload.intents.join(", ")) : "all domain models"}</strong>
      </div>
    </div>
    ${formatPlan(payload.plan)}
  `;
}

function renderArtifactCard(name, url) {
  const lower = url.toLowerCase();
  const imageMarkup = lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".webp")
    ? `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" loading="lazy" />`
    : "";
  return `
    <article class="artifact-card">
      ${imageMarkup}
      <a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(name)}</a>
    </article>
  `;
}

function renderResults(payload) {
  const clinicalReportBlock = payload.report_markdown ? `
    <section class="report-card">
      <h3>Structured clinical report</h3>
      <pre>${escapeHtml(payload.report_markdown)}</pre>
    </section>
  ` : "";

  const cards = [];
  for (const result of payload.results || []) {
    const kvPairs = Object.entries(result.outputs || {});
    const artifacts = Object.entries(result.artifacts || {});
    cards.push(`
      <article class="result-card">
        <header>
          <h3>${escapeHtml(result.tool_name)}</h3>
          <span class="result-status ${statusClassName(result.status)}">${escapeHtml(result.status)}</span>
        </header>
        <p>${escapeHtml(result.summary)}</p>
        ${kvPairs.length ? `
          <dl class="kv-list">
            ${kvPairs.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(formatValue(value))}</dd></div>`).join("")}
          </dl>
        ` : ""}
        ${artifacts.length ? `
          <div class="artifact-grid">
            ${artifacts.map(([name, url]) => renderArtifactCard(name, url)).join("")}
          </div>
        ` : ""}
        ${result.errors && result.errors.length ? `<p><strong>Errors:</strong> ${escapeHtml(result.errors.join("; "))}</p>` : ""}
      </article>
    `);
  }

  const technicalAppendixBlock = payload.technical_appendix_markdown ? `
    <section class="report-card">
      <h3>Technical appendix</h3>
      <pre>${escapeHtml(payload.technical_appendix_markdown)}</pre>
    </section>
  ` : "";

  resultDetails.innerHTML = clinicalReportBlock + cards.join("") + technicalAppendixBlock;
}

function renderReasonedReport(payload) {
  reasonedReport.className = "reasoned-report";
  reasonedReport.innerHTML = `
    <section class="report-card reasoned-card">
      <div class="reasoned-header">
        <div>
          <p class="section-kicker">Reasoning</p>
          <h3>Reasoned interpretation</h3>
        </div>
        <span class="reasoned-model">${escapeHtml(payload.model || "OpenAI")}</span>
      </div>
      <pre>${escapeHtml(payload.output || "")}</pre>
    </section>
  `;
}

function renderHealth(payload) {
  document.getElementById("health-status").textContent = payload.status;
  document.getElementById("health-tasks").textContent = payload.tasks_total;
  document.getElementById("health-ready").textContent = payload.tasks_ready;
  document.getElementById("health-checkpoints").textContent = payload.checkpoints_available;
  document.getElementById("health-device").textContent = payload.device;
}

function renderTasks(payload) {
  const rows = payload.tasks.map((task) => `
    <tr>
      <td>${escapeHtml(task.domain)}</td>
      <td><strong>${escapeHtml(task.task_id)}</strong><br /><small>${escapeHtml(task.title)}</small></td>
      <td>${escapeHtml(task.task_type)}</td>
      <td><span class="status-pill ${task.status}">${escapeHtml(task.status)}</span></td>
      <td>
        <span class="checkpoint-pill ${task.checkpoint_available ? "available" : "missing"}">
          ${escapeHtml(task.checkpoint_available ? "available" : "missing")}
        </span>
      </td>
    </tr>
  `).join("");

  taskTableBody.innerHTML = rows || '<tr><td colspan="5">No tasks discovered.</td></tr>';
}

async function loadBootstrapData() {
  const [healthResponse, tasksResponse] = await Promise.all([
    fetch("/api/health"),
    fetch("/api/tasks"),
  ]);

  if (!healthResponse.ok || !tasksResponse.ok) {
    throw new Error("Failed to load PULSE bootstrap data.");
  }

  renderHealth(await healthResponse.json());
  renderTasks(await tasksResponse.json());
}

function collectArtifactUrls(payload) {
  const urls = [];
  for (const result of payload.results || []) {
    for (const url of Object.values(result.artifacts || {})) {
      if (typeof url === "string" && url.toLowerCase().endsWith(".png") && !urls.includes(url)) {
        urls.push(url);
      }
      if (urls.length >= 2) {
        return urls;
      }
    }
  }
  return urls;
}

async function fetchArtifactFile(url, fallbackName) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch artifact: ${url}`);
  }
  const blob = await response.blob();
  if (blob.type !== "image/png") {
    throw new Error(`Artifact is not a PNG image: ${url}`);
  }
  const filename = url.split("/").pop() || fallbackName;
  return new File([blob], filename, { type: "image/png" });
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : null);
    reader.onerror = () => reject(new Error(`Failed to read ${file.name || "image file"}.`));
    reader.readAsDataURL(file);
  });
}

function buildReasoningPayload(payload) {
  const findings = (payload.results || []).map((result) => ({
    title: result.display_name || result.tool_name || "Finding",
    summary: result.summary || "",
    label: result.outputs?.label || null,
    confidence: typeof result.confidence === "number"
      ? result.confidence
      : (typeof result.outputs?.confidence === "number" ? result.outputs.confidence : null),
    details: Object.entries(result.outputs || {}).map(([key, value]) => `${key}: ${formatValue(value)}`),
    artifacts_available: Object.keys(result.artifacts || {}),
  }));

  return {
    exam: payload.detected_domain || "unknown",
    detected_anatomy: payload.detected_anatomy || payload.routing_label || payload.detected_domain || "unknown",
    subview: payload.subview || null,
    quality: {
      brightness: payload.quality?.brightness ?? null,
      contrast: payload.quality?.contrast ?? null,
      quality_label: payload.quality?.quality_label || "unknown",
    },
    reasoning_evidence: {
      organ_identification: {
        primary_hypothesis: payload.detected_domain || "unknown",
        anatomy_label: payload.detected_anatomy || payload.routing_label || payload.detected_domain || "unknown",
      },
      downstream_support: findings,
      quality_constraints: {
        quality_label: payload.quality?.quality_label || "unknown",
        single_image_review: true,
      },
    },
    base_impression: payload.report_markdown || "",
    base_note: payload.technical_appendix_markdown || "",
    user_prompt: form.elements.namedItem("prompt")?.value?.trim() || null,
    findings,
  };
}

async function generateReasonedInterpretation(payload) {
  const enableReasoning = form.elements.namedItem("enable_reasoning")?.checked;
  const primaryImage = form.elements.namedItem("image")?.files?.[0];

  if (!enableReasoning) {
    setReasoningPlaceholder("Reasoned interpretation is disabled for this run.");
    return;
  }

  if (!primaryImage) {
    setReasoningPlaceholder("Reasoned interpretation could not start because the original image is no longer available in the form.");
    return;
  }

  setReasoningPlaceholder("Generating clinician-facing reasoning...");

  const artifactUrls = collectArtifactUrls(payload);
  const artifactDataUrls = [];
  for (const [index, url] of artifactUrls.entries()) {
    try {
      const file = await fetchArtifactFile(url, `artifact_${index + 1}.png`);
      const dataURL = await fileToDataURL(file);
      if (dataURL) {
        artifactDataUrls.push(dataURL);
      }
    } catch (_error) {
      // Artifact images are optional support; skip them if the fetch fails.
    }
  }

  const requestBody = {
    model: "gpt-5.4",
    report_mode: "reasoning",
    structured_payload: buildReasoningPayload(payload),
    primary_image_url: await fileToDataURL(primaryImage),
    artifact_image_urls: artifactDataUrls,
  };

  const response = await fetch("/api/reason", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  const bodyText = await response.text();
  let parsed;
  try {
    parsed = bodyText ? JSON.parse(bodyText) : {};
  } catch {
    parsed = { detail: bodyText || "Reasoning service returned an invalid response." };
  }

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error("Reasoning endpoint is unavailable on this server. Start the LX server wrapper to enable it.");
    }
    throw new Error(parsed.detail || "Reasoned interpretation failed.");
  }

  renderReasonedReport(parsed);
}

async function submitAnalysis(event) {
  event.preventDefault();
  submitButton.disabled = true;
  submitStatus.textContent = "Uploading image and running the agent plan...";
  resultDetails.innerHTML = "";
  resultSummary.classList.add("empty-state");
  resultSummary.textContent = "Analysis in progress.";
  setReasoningPlaceholder("Reasoned interpretation will appear here after the structured PULSE analysis completes.");

  try {
    const data = new FormData(form);
    const checkedHints = Array.from(document.querySelectorAll('input[name="task_hint"]:checked'))
      .map((element) => element.value)
      .join(",");
    data.set("task_hints", checkedHints);

    const response = await fetch("/api/analyze", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "PULSE analysis failed.");
    }

    renderSummary(payload);
    renderResults(payload);
    submitStatus.textContent = "Structured analysis complete. Generating reasoning...";

    try {
      await generateReasonedInterpretation(payload);
      submitStatus.textContent = "Analysis and reasoning complete.";
    } catch (reasoningError) {
      setReasoningPlaceholder(String(reasoningError.message || reasoningError));
      submitStatus.textContent = "Structured analysis complete. Reasoning report unavailable.";
    }
  } catch (error) {
    resultSummary.classList.add("empty-state");
    resultSummary.textContent = String(error.message || error);
    resultDetails.innerHTML = "";
    setReasoningPlaceholder("Reasoned interpretation is unavailable because the main analysis did not complete.");
    submitStatus.textContent = "Analysis failed.";
  } finally {
    submitButton.disabled = false;
  }
}

form.addEventListener("submit", submitAnalysis);

loadBootstrapData().catch((error) => {
  document.getElementById("health-status").textContent = "offline";
  taskTableBody.innerHTML = `<tr><td colspan="5">${escapeHtml(error.message)}</td></tr>`;
  submitStatus.textContent = "Server bootstrap failed.";
});
