/**
 * NanoAmharic-MorphBPE Web UI Client Logic
 * Handles interactive tabs, generation endpoints, chat history, live tokenizer lab, and telemetry.
 */

// Global State
let activeTab = "story";
let chatHistory = [];

document.addEventListener("DOMContentLoaded", () => {
    fetchHealthStatus();
    setupEventListeners();
});

function setupEventListeners() {
    // Slider value synchronizers
    const bindSlider = (sliderId, labelId, suffix = "") => {
        const slider = document.getElementById(sliderId);
        const label = document.getElementById(labelId);
        if (slider && label) {
            slider.addEventListener("input", (e) => {
                label.innerText = e.target.value + suffix;
            });
        }
    };

    bindSlider("temp-slider", "temp-val");
    bindSlider("topk-slider", "topk-val");
    bindSlider("topp-slider", "topp-val");
    bindSlider("tokens-slider", "tokens-val");

    // Chat enter key
    const chatInput = document.getElementById("chat-input");
    if (chatInput) {
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                sendChatMessage();
            }
        });
    }
}

// -----------------------------------------------------------------------------
// Tab Switching
// -----------------------------------------------------------------------------
function switchTab(tabName) {
    activeTab = tabName;
    document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.style.display = "none";
    });
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.remove("active");
    });

    const targetPane = document.getElementById(`tab-${tabName}`);
    if (targetPane) targetPane.style.display = "block";

    const targetBtn = document.getElementById(`btn-tab-${tabName}`);
    if (targetBtn) targetBtn.classList.add("active");
}

// -----------------------------------------------------------------------------
// Health & Telemetry
// -----------------------------------------------------------------------------
async function fetchHealthStatus() {
    try {
        const res = await fetch("/health");
        if (res.ok) {
            const data = await res.json();
            const ramBadge = document.getElementById("status-ram");
            const modelBadge = document.getElementById("status-model");
            if (ramBadge) ramBadge.innerText = `RAM: ${data.memory_rss_mb} MB`;
            if (modelBadge) modelBadge.innerText = `${(data.parameters / 1e6).toFixed(2)}M Params`;
        }
    } catch (e) {
        console.warn("Health check error:", e);
    }
}

function getGenerationParams() {
    return {
        temperature: parseFloat(document.getElementById("temp-slider")?.value || 0.7),
        top_k: parseInt(document.getElementById("topk-slider")?.value || 40),
        top_p: parseFloat(document.getElementById("topp-slider")?.value || 0.9),
        max_tokens: parseInt(document.getElementById("tokens-slider")?.value || 50),
        precision: document.getElementById("precision-select")?.value || "fp32"
    };
}

// -----------------------------------------------------------------------------
// Tab 1: Storyteller
// -----------------------------------------------------------------------------
async function generateStory() {
    const topicInput = document.getElementById("story-topic");
    const genreInput = document.getElementById("story-genre");
    const outBox = document.getElementById("story-output");
    const telemetry = document.getElementById("story-telemetry");

    const topic = topicInput.value.trim() || "አንድ ጥበበኛ ገበሬ";
    const genre = genreInput ? genreInput.value : "ተረት";
    const params = getGenerationParams();

    outBox.innerText = "";
    outBox.classList.add("loading");
    telemetry.innerHTML = "";

    try {
        const res = await fetch("/api/story", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                topic: topic,
                genre: genre,
                max_tokens: params.max_tokens,
                temperature: params.temperature,
                top_k: params.top_k,
                top_p: params.top_p,
                precision: params.precision
            })
        });

        const data = await res.json();
        outBox.classList.remove("loading");
        outBox.innerText = data.full_text || "ምንም ጽሑፍ አልተገኘም።";

        telemetry.innerHTML = `
            <div class="telemetry-item">⚡ ፍጥነት፡ <span class="telemetry-val">${data.tokens_per_second} tok/s</span></div>
            <div class="telemetry-item">⏱ የፈጀው ጊዜ፡ <span class="telemetry-val">${data.elapsed_seconds}s</span></div>
            <div class="telemetry-item">📝 የተፈጠሩ ቃላት፡ <span class="telemetry-val">${data.tokens_generated} tokens</span></div>
        `;
    } catch (err) {
        outBox.classList.remove("loading");
        outBox.innerText = "ስህተት ተከስቷል፡ " + err.message;
    }
}

function setStoryPreset(text) {
    const input = document.getElementById("story-topic");
    if (input) {
        input.value = text;
        generateStory();
    }
}

// -----------------------------------------------------------------------------
// Tab 2: Conversational Chat
// -----------------------------------------------------------------------------
async function sendChatMessage() {
    const input = document.getElementById("chat-input");
    const chatWindow = document.getElementById("chat-window");
    const telemetry = document.getElementById("chat-telemetry");
    const message = input.value.trim();
    if (!message) return;

    // Render user message
    const userBubble = document.createElement("div");
    userBubble.className = "chat-bubble user";
    userBubble.innerText = message;
    chatWindow.appendChild(userBubble);
    input.value = "";
    chatWindow.scrollTop = chatWindow.scrollHeight;

    // Loading bot bubble
    const botBubble = document.createElement("div");
    botBubble.className = "chat-bubble bot";
    botBubble.innerText = "...";
    chatWindow.appendChild(botBubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    const params = getGenerationParams();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
                max_tokens: params.max_tokens,
                temperature: params.temperature,
                precision: params.precision
            })
        });

        const data = await res.json();
        botBubble.innerText = data.assistant_response || "መልስ ማግኘት አልተቻለም።";
        chatHistory.push({ user: message, assistant: data.assistant_response });
        chatWindow.scrollTop = chatWindow.scrollHeight;

        telemetry.innerHTML = `
            <div class="telemetry-item">⚡ <span class="telemetry-val">${data.tokens_per_second} tok/s</span></div>
            <div class="telemetry-item">⏱ <span class="telemetry-val">${data.elapsed_seconds}s</span></div>
        `;
    } catch (err) {
        botBubble.innerText = "ስህተት፡ " + err.message;
    }
}

function clearChat() {
    chatHistory = [];
    const chatWindow = document.getElementById("chat-window");
    if (chatWindow) {
        chatWindow.innerHTML = `
            <div class="chat-bubble bot">
                ሰላም! እኔ ናኖ-አማርኛ ነኝ። ዛሬ በምን ልርዳዎ?
            </div>
        `;
    }
}

function setChatPreset(text) {
    const input = document.getElementById("chat-input");
    if (input) {
        input.value = text;
        sendChatMessage();
    }
}

// -----------------------------------------------------------------------------
// Tab 3: Text Completion
// -----------------------------------------------------------------------------
async function generateCompletion() {
    const input = document.getElementById("completion-prompt");
    const outBox = document.getElementById("completion-output");
    const telemetry = document.getElementById("completion-telemetry");

    const prompt = input.value.trim() || "የኢትዮጵያ ታሪካዊ ከተሞች";
    const params = getGenerationParams();

    outBox.innerText = "";
    outBox.classList.add("loading");
    telemetry.innerHTML = "";

    try {
        const res = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: params.max_tokens,
                temperature: params.temperature,
                top_k: params.top_k,
                top_p: params.top_p,
                precision: params.precision
            })
        });

        const data = await res.json();
        outBox.classList.remove("loading");
        outBox.innerText = data.full_text || data.continuation || "";

        telemetry.innerHTML = `
            <div class="telemetry-item">⚡ ፍጥነት፡ <span class="telemetry-val">${data.tokens_per_second} tok/s</span></div>
            <div class="telemetry-item">⏱ ጊዜ፡ <span class="telemetry-val">${data.elapsed_seconds}s</span></div>
            <div class="telemetry-item">📝 ብዛት፡ <span class="telemetry-val">${data.tokens_generated} tokens</span></div>
        `;
    } catch (err) {
        outBox.classList.remove("loading");
        outBox.innerText = "ስህተት፡ " + err.message;
    }
}

function setCompletionPreset(text) {
    const input = document.getElementById("completion-prompt");
    if (input) {
        input.value = text;
        generateCompletion();
    }
}

// -----------------------------------------------------------------------------
// Tab 4: MorphBPE Tokenizer & Morpheme Lab
// -----------------------------------------------------------------------------
async function inspectTokens() {
    const input = document.getElementById("tokenizer-input");
    const container = document.getElementById("tokenizer-results");
    const text = input.value.trim() || "በመንግሥታችን አልተረዱልንም";

    container.innerHTML = "<p style='color: var(--text-secondary);'>ቃላቱ እየተተነተኑ ነው...</p>";

    try {
        const res = await fetch("/api/tokenize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        const data = await res.json();
        renderTokenizerComparison(data);
    } catch (err) {
        container.innerHTML = "<p style='color: var(--danger);'>ስህተት፡ " + err.message + "</p>";
    }
}

function renderTokenizerComparison(data) {
    const container = document.getElementById("tokenizer-results");
    
    let morphChipsHtml = data.morphbpe_tokens.map(tok => 
        `<span class="token-chip morph-chip">${tok}</span>`
    ).join(" ");

    let stdChipsHtml = data.standard_bpe_tokens.map(tok => 
        `<span class="token-chip std-chip">${tok}</span>`
    ).join(" ");

    let analysisHtml = data.words_analysis.map(w => {
        let bStatus = w.crosses_standard ? 
            `<span style="color:var(--danger); font-size:0.85rem;">⚠️ Standard BPE ድንበር አቋርጧል</span>` : 
            `<span style="color:var(--success); font-size:0.85rem;">✅ ድንበር ተጠብቋል</span>`;

        return `
            <div style="margin-bottom: 12px; padding: 10px; background: rgba(255,255,255,0.02); border-radius:6px;">
                <div style="font-weight:bold; color:var(--accent);">${w.word} ${bStatus}</div>
                <div style="font-size:0.88rem; color:var(--text-secondary); margin-top:4px;">
                    የተለዩ ድንበሮች፡ [${w.boundaries.join(", ") || "ምንም"}] | 
                    MorphBPE: <strong>[${w.morphbpe_subwords.join(", ")}]</strong> | 
                    Standard BPE: <strong>[${w.standard_bpe_subwords.join(", ")}]</strong>
                </div>
            </div>
        `;
    }).join("");

    container.innerHTML = `
        <div class="token-grid">
            <div class="token-row">
                <div class="token-row-title">
                    <span>MorphBPE (λ = 0.50) — ሥነ-ምዕላዳዊ ድንበር ጠባቂ</span>
                    <span style="color: var(--accent);">${data.morphbpe_tokens.length} subwords</span>
                </div>
                <div class="tokens-chips">${morphChipsHtml}</div>
            </div>

            <div class="token-row">
                <div class="token-row-title">
                    <span>Standard BPE (λ = 0.0) — ድግግሞሽ-ተኮር (መደበኛ BPE)</span>
                    <span style="color: #f87171;">${data.standard_bpe_tokens.length} subwords</span>
                </div>
                <div class="tokens-chips">${stdChipsHtml}</div>
            </div>

            <div class="token-row">
                <div class="token-row-title">የቃላት ዝርዝር እና የሞርፎሎጂ ድንበር ትንተና</div>
                <div>${analysisHtml}</div>
            </div>
        </div>
    `;
}

function setTokenizerPreset(text) {
    const input = document.getElementById("tokenizer-input");
    if (input) {
        input.value = text;
        inspectTokens();
    }
}
