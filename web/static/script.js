const elements = {
  input: document.getElementById("queryInput"),
  button: document.getElementById("searchButton"),
  loading: document.getElementById("loadingIndicator"),
  answerPanel: document.getElementById("answerPanel"),
  answerText: document.getElementById("answerText"),
  sourcesPanel: document.getElementById("sourcesPanel"),
  sourcesGrid: document.getElementById("sourcesGrid"),
  errorPanel: document.getElementById("errorPanel"),
};

const severityColors = {
  CRITICAL: "#ff3b3b",
  HIGH: "#ff8c00",
  MEDIUM: "#ffd700",
  LOW: "#00ff88",
  UNKNOWN: "#00b4d8",
};

function setLoadingState(isLoading) {
  elements.loading.classList.toggle("active", isLoading);
  elements.button.disabled = isLoading;
}

function resetPanels() {
  elements.answerPanel.classList.remove("active");
  elements.sourcesPanel.classList.remove("active");
  elements.errorPanel.classList.remove("active");
  elements.errorPanel.textContent = "";
  elements.answerText.textContent = "";
  elements.sourcesGrid.innerHTML = "";
}

function normalizeSeverity(severity) {
  return String(severity || "UNKNOWN").toUpperCase();
}

function severityColor(severity) {
  return severityColors[normalizeSeverity(severity)] || severityColors.UNKNOWN;
}

function truncateDescription(text, maxLength = 200) {
  const value = String(text || "Not available");
  return value.length > maxLength ? `${value.slice(0, maxLength).trim()}...` : value;
}

function formatProducts(products) {
  if (Array.isArray(products)) {
    return products.length ? products.join(", ") : "Not available";
  }

  return products ? String(products) : "Not available";
}

function createCveCard(result) {
  const severity = normalizeSeverity(result.severity);
  const scorePercent = (Number(result.score || 0) * 100).toFixed(1);
  const cvss = result.cvss ?? "N/A";
  const nvdLink = result.nvd_url
    ? `<a class="nvd-link" href="${result.nvd_url}" target="_blank" rel="noopener noreferrer">Open in NVD</a>`
    : "";

  const card = document.createElement("article");
  card.className = "cve-card";
  card.style.borderLeftColor = severityColor(severity);

  card.innerHTML = `
    <div class="cve-top">
      <span class="cve-id">${result.cve_id || "Unknown CVE"}</span>
      <span class="severity-badge" style="background:${severityColor(severity)}">${severity}</span>
      <span class="cvss-pill">CVSS ${cvss}</span>
    </div>
    <p class="products">AFFECTED: ${formatProducts(result.affected_products)}</p>
    <p class="description">${truncateDescription(result.description)}</p>
    <div class="card-bottom">
      <span>MATCH: ${scorePercent}%</span>
      ${nvdLink}
    </div>
  `;

  return card;
}

function renderResults(data) {
  elements.answerText.textContent = data.answer || "No answer returned.";
  elements.answerPanel.classList.add("active");

  if (Array.isArray(data.cves) && data.cves.length > 0) {
    data.cves.forEach((result) => {
      elements.sourcesGrid.appendChild(createCveCard(result));
    });
    elements.sourcesPanel.classList.add("active");
  }
}

function showError(message) {
  elements.errorPanel.textContent = message;
  elements.errorPanel.classList.add("active");
}

async function runSearch() {
  const query = elements.input.value.trim();
  if (!query) {
    showError("Enter a query before searching the CVE database.");
    return;
  }

  resetPanels();
  setLoadingState(true);

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The search request failed.");
    }

    renderResults(data);
  } catch (error) {
    showError(error.message || "An unexpected error occurred.");
  } finally {
    setLoadingState(false);
  }
}

elements.button.addEventListener("click", runSearch);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runSearch();
  }
});
