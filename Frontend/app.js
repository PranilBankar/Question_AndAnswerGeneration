/**
 * app.js — Frontend logic for NEET Biology Q&A
 *
 * Handles:
 *   - API calls to FastAPI backend (classify, answer, full analysis)
 *   - Dynamic result rendering
 *   - Loading states and error handling
 *   - Health check polling
 */

const API_BASE = "http://localhost:8000";

// ==============================
// DOM REFERENCES
// ==============================
const questionInput = document.getElementById("questionInput");
const charCount = document.getElementById("charCount");
const btnClassify = document.getElementById("btnClassify");
const btnAnswer = document.getElementById("btnAnswer");
const btnFull = document.getElementById("btnFull");

const loadingSection = document.getElementById("loadingSection");
const loaderText = document.getElementById("loaderText");
const loaderSteps = document.getElementById("loaderSteps");

const resultsSection = document.getElementById("resultsSection");
const classificationCard = document.getElementById("classificationCard");
const answerCard = document.getElementById("answerCard");
const justificationCard = document.getElementById("justificationCard");
const sourcesCard = document.getElementById("sourcesCard");

const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");


// ==============================
// CHAR COUNT
// ==============================
questionInput.addEventListener("input", () => {
    const len = questionInput.value.length;
    charCount.textContent = `${len} / 500`;
});

// Enter key to submit (Shift+Enter for newline)
questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleFullAnalysis();
    }
});


// ==============================
// HEALTH CHECK
// ==============================
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            const data = await res.json();
            statusDot.className = "status-dot online";
            statusText.textContent = "Server Online";
            return true;
        }
    } catch (e) {
        // Server offline
    }
    statusDot.className = "status-dot offline";
    statusText.textContent = "Server Offline";
    return false;
}

// Check on load, then every 30s
checkHealth();
setInterval(checkHealth, 30000);


// ==============================
// BUTTON HANDLERS
// ==============================

/** 🏷️ Classify Only */
async function handleClassify() {
    const question = questionInput.value.trim();
    if (!validate(question)) return;

    setLoading(true, "Classifying your question...", [
        "BERT analyzing question...",
        "Extracting chapter & topics...",
    ]);

    try {
        const res = await fetch(`${API_BASE}/api/classify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        setLoading(false);
        renderClassificationOnly(data);
    } catch (err) {
        setLoading(false);
        showError(err.message);
    }
}


/** 📝 Generate Answer (no classifier) */
async function handleAnswer() {
    const question = questionInput.value.trim();
    if (!validate(question)) return;

    setLoading(true, "Generating answer...", [
        "Retrieving NCERT context...",
        "Generating answer via AI...",
        "Verifying accuracy...",
    ]);

    try {
        const res = await fetch(`${API_BASE}/api/qa`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, use_classifier: false }),
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        setLoading(false);
        renderQAResult(data, false);
    } catch (err) {
        setLoading(false);
        showError(err.message);
    }
}


/** 🔬 Full Analysis (classify + answer) */
async function handleFullAnalysis() {
    const question = questionInput.value.trim();
    if (!validate(question)) return;

    setLoading(true, "Running full analysis...", [
        "BERT classifying chapter...",
        "Extracting topics...",
        "Retrieving NCERT context...",
        "Generating answer via AI...",
        "Verifying & scoring...",
    ]);

    try {
        const res = await fetch(`${API_BASE}/api/qa`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, use_classifier: true }),
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        setLoading(false);
        renderQAResult(data, true);
    } catch (err) {
        setLoading(false);
        showError(err.message);
    }
}


// ==============================
// VALIDATION
// ==============================
function validate(question) {
    if (!question || question.length < 5) {
        showError("Please enter a question (at least 5 characters).");
        return false;
    }
    return true;
}


// ==============================
// LOADING STATE
// ==============================
function setLoading(show, text = "", steps = []) {
    const allBtns = [btnClassify, btnAnswer, btnFull];

    if (show) {
        hideResults();
        hideError();
        loadingSection.style.display = "block";
        loaderText.textContent = text;
        loaderSteps.innerHTML = "";

        // Animate steps appearing one by one
        steps.forEach((step, i) => {
            setTimeout(() => {
                const el = document.createElement("div");
                el.className = "loader-step";
                el.innerHTML = `<span>⏳</span> ${step}`;
                el.style.animationDelay = `${i * 0.15}s`;
                loaderSteps.appendChild(el);
            }, i * 800);
        });

        allBtns.forEach((b) => (b.disabled = true));
    } else {
        loadingSection.style.display = "none";
        allBtns.forEach((b) => (b.disabled = false));
    }
}


// ==============================
// RENDER: Classification Only
// ==============================
function renderClassificationOnly(data) {
    hideResults();
    resultsSection.style.display = "flex";

    const cls = data.classification;
    if (cls) {
        renderClassificationCard(cls);
        classificationCard.style.display = "block";
    }

    // Hide QA cards
    answerCard.style.display = "none";
    justificationCard.style.display = "none";
    sourcesCard.style.display = "none";
}


// ==============================
// RENDER: Full QA Result
// ==============================
function renderQAResult(data, showClassification) {
    hideResults();
    resultsSection.style.display = "flex";

    // Classification
    if (showClassification && data.classification) {
        renderClassificationCard(data.classification);
        classificationCard.style.display = "block";
    } else {
        classificationCard.style.display = "none";
    }

    // Answer
    if (data.answer) {
        renderAnswerCard(data);
        answerCard.style.display = "block";
    }

    // Justification
    if (data.justification && data.justification.length > 0) {
        renderJustificationCard(data.justification);
        justificationCard.style.display = "block";
    }

    // Sources
    if (data.sources && data.sources.length > 0) {
        renderSourcesCard(data.sources, data.confidence);
        sourcesCard.style.display = "block";
    }
}


// ==============================
// RENDER HELPERS
// ==============================

function renderClassificationCard(cls) {
    // Chapter confidence badge
    const confEl = document.getElementById("chapterConfidence");
    const confPct = (cls.chapter_confidence * 100).toFixed(1);
    const confClass = cls.chapter_confidence >= 0.8 ? "confidence-high" :
                      cls.chapter_confidence >= 0.5 ? "confidence-medium" : "confidence-low";
    confEl.className = `confidence-badge ${confClass}`;
    confEl.textContent = `${confPct}% confidence`;

    // Chapter name
    document.getElementById("chapterName").textContent = cls.chapter;

    // Topics
    const topicsList = document.getElementById("topicsList");
    topicsList.innerHTML = "";

    if (cls.topics && cls.topics.length > 0) {
        cls.topics.forEach((t) => {
            const item = document.createElement("div");
            item.className = "topic-item";
            item.innerHTML = `
                <div class="topic-name">
                    <span class="topic-code">${escapeHtml(t.section_code)}</span>
                    <span class="topic-title">${escapeHtml(t.section_title)}</span>
                </div>
                <span class="topic-similarity">${(t.similarity * 100).toFixed(1)}% match</span>
            `;
            topicsList.appendChild(item);
        });
    } else {
        topicsList.innerHTML = `<p style="color: var(--text-muted); font-size: 0.85rem;">No specific topics identified</p>`;
    }
}


function renderAnswerCard(data) {
    // Verification badge
    const badge = document.getElementById("verificationBadge");
    if (data.verified) {
        badge.className = "verification-badge verified-true";
        badge.textContent = "✓ Verified";
    } else {
        badge.className = "verification-badge verified-false";
        badge.textContent = "✗ Unverified";
    }

    // Answer text
    document.getElementById("answerContent").textContent = data.answer;
}


function renderJustificationCard(justification) {
    const list = document.getElementById("justificationList");
    list.innerHTML = "";
    justification.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        list.appendChild(li);
    });
}


function renderSourcesCard(sources, confidence) {
    // Overall confidence
    const confEl = document.getElementById("overallConfidence");
    const confPct = (confidence * 100).toFixed(1);
    const confClass = confidence >= 0.7 ? "confidence-high" :
                      confidence >= 0.4 ? "confidence-medium" : "confidence-low";
    confEl.className = `confidence-badge ${confClass}`;
    confEl.textContent = `${confPct}% confidence`;

    // Source list
    const list = document.getElementById("sourcesList");
    list.innerHTML = "";
    sources.forEach((s) => {
        const item = document.createElement("div");
        item.className = "source-item";
        item.innerHTML = `
            <div class="source-meta">
                <span class="source-chapter">${escapeHtml(s.chapter || "")}</span>
                <span class="source-section">${escapeHtml(s.section || "")} — ${escapeHtml(s.section_title || "")}</span>
                <span class="source-similarity">${(s.similarity * 100).toFixed(1)}%</span>
            </div>
            <div class="source-text">${escapeHtml(s.text_content || "")}</div>
        `;
        list.appendChild(item);
    });
}


// ==============================
// UTILITY
// ==============================

function hideResults() {
    resultsSection.style.display = "none";
    classificationCard.style.display = "none";
    answerCard.style.display = "none";
    justificationCard.style.display = "none";
    sourcesCard.style.display = "none";
}

function showError(msg) {
    hideResults();
    errorSection.style.display = "block";
    errorMessage.textContent = msg;
}

function hideError() {
    errorSection.style.display = "none";
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
