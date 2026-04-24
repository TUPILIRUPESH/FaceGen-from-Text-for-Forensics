/**
 * ForensicAI — Main JavaScript
 * Handles form submission, face generation API calls, image display,
 * download/print, history management, and toast notifications.
 */

// ─── State ──────────────────────────────────────────────────────────
const MAX_HISTORY = 5;
let generationHistory = [];
let isGenerating = false;

// ─── DOM References ─────────────────────────────────────────────────
const form = document.getElementById('descriptionForm');
const generateBtn = document.getElementById('generateBtn');
const regenerateBtn = document.getElementById('regenerateBtn');
const outputPlaceholder = document.getElementById('outputPlaceholder');
const outputResults = document.getElementById('outputResults');
const variationGrid = document.getElementById('variationGrid');
const processedDescription = document.getElementById('processedDescription');
const genTime = document.getElementById('genTime');
const genCount = document.getElementById('genCount');
const historyList = document.getElementById('historyList');
const historyEmpty = document.getElementById('historyEmpty');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const charCount = document.getElementById('charCount');
const toastContainer = document.getElementById('toastContainer');
const deviceInfo = document.getElementById('deviceInfo');
const statusBadge = document.getElementById('statusBadge');

// ─── Init ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
});

function setupEventListeners() {
    form.addEventListener('submit', handleGenerate);
    regenerateBtn.addEventListener('click', handleRegenerate);
    clearHistoryBtn.addEventListener('click', clearHistory);

    // Character count for textarea
    const textarea = document.getElementById('additionalDesc');
    textarea.addEventListener('input', () => {
        charCount.textContent = textarea.value.length;
        if (textarea.value.length > 900) {
            charCount.style.color = 'var(--warning)';
        } else if (textarea.value.length > 1000) {
            charCount.style.color = 'var(--error)';
        } else {
            charCount.style.color = 'var(--text-muted)';
        }
    });
}

// ─── Health Check ───────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch('/health');
        const data = await res.json();
        deviceInfo.textContent = data.device === 'cuda' ? 'GPU (CUDA)' : 'CPU';

        const models = data.models || {};
        const mode = models.generator === 'StyleGAN2' ? 'Full Model' : 'Demo Mode';
        statusBadge.querySelector('.status-text').textContent = `Ready — ${mode}`;
    } catch (e) {
        deviceInfo.textContent = 'Unknown';
        statusBadge.querySelector('.status-text').textContent = 'Disconnected';
        statusBadge.querySelector('.status-dot').style.background = 'var(--error)';
    }
}

// ─── Form Submission ────────────────────────────────────────────────
async function handleGenerate(e) {
    e.preventDefault();
    if (isGenerating) return;
    await generateFace();
}

async function handleRegenerate() {
    if (isGenerating) return;
    await generateFace();
}

async function generateFace() {
    const data = collectFormData();

    // Validate — at least one field filled
    const hasContent = Object.values(data).some(v => v && v.trim());
    if (!hasContent) {
        showToast('Please fill in at least one description field.', 'error');
        return;
    }

    setLoading(true);

    try {
        const res = await fetch('/generate-face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await res.json();

        if (!res.ok) {
            throw new Error(result.error || 'Generation failed');
        }

        displayResults(result);
        addToHistory(result);
        showToast(`Generated ${result.images.length} variations successfully!`, 'success');

    } catch (err) {
        console.error('Generation error:', err);
        showToast(err.message || 'Failed to generate face. Check server.', 'error');
    } finally {
        setLoading(false);
    }
}

function collectFormData() {
    return {
        gender: document.getElementById('gender').value,
        age_group: document.getElementById('ageGroup').value,
        skin_tone: document.getElementById('skinTone').value,
        face_shape: document.getElementById('faceShape').value,
        eye_color: document.getElementById('eyeColor').value,
        hair_style: document.getElementById('hairStyle').value,
        facial_hair: document.getElementById('facialHair').value,
        accessories: document.getElementById('accessories').value,
        description: document.getElementById('additionalDesc').value,
    };
}

// ─── Display Results ────────────────────────────────────────────────
function displayResults(result) {
    outputPlaceholder.style.display = 'none';
    outputResults.style.display = 'flex';

    // Description
    processedDescription.textContent = result.description;

    // Generation info
    genTime.textContent = `${result.generation_time_ms.toFixed(0)}ms`;
    genCount.textContent = result.images.length;

    // Build variation cards
    variationGrid.innerHTML = '';
    result.images.forEach((imgBase64, idx) => {
        const score = result.similarity_scores[idx];
        const card = createVariationCard(imgBase64, idx, score);
        variationGrid.appendChild(card);
    });
}

function createVariationCard(imgBase64, index, score) {
    const card = document.createElement('div');
    card.className = 'variation-card';
    card.style.animation = `fadeInUp 0.4s ease-out ${index * 0.1}s both`;

    const scoreClass = score >= 80 ? 'score-high' : score >= 60 ? 'score-medium' : 'score-low';

    card.innerHTML = `
        <div class="card-header">
            <span class="variation-label">Variation ${index + 1}</span>
            <span class="score-badge ${scoreClass}">
                <i class="fas fa-chart-line"></i> ${score}%
            </span>
        </div>
        <div class="variation-image-wrap">
            <img src="data:image/png;base64,${imgBase64}" alt="Generated face variation ${index + 1}" />
        </div>
        <div class="variation-actions">
            <button class="btn-action" onclick="downloadImage('${imgBase64}', ${index})" title="Download PNG">
                <i class="fas fa-download"></i> Download
            </button>
            <button class="btn-action" onclick="printImage('${imgBase64}')" title="Print">
                <i class="fas fa-print"></i> Print
            </button>
        </div>
    `;

    return card;
}

// Add fadeInUp animation
if (!document.getElementById('dynamicAnimations')) {
    const style = document.createElement('style');
    style.id = 'dynamicAnimations';
    style.textContent = `
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
    `;
    document.head.appendChild(style);
}

// ─── Download ───────────────────────────────────────────────────────
function downloadImage(base64, index) {
    try {
        const byteString = atob(base64);
        const arrayBuffer = new ArrayBuffer(byteString.length);
        const uint8Array = new Uint8Array(arrayBuffer);
        for (let i = 0; i < byteString.length; i++) {
            uint8Array[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([uint8Array], { type: 'image/png' });
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `suspect_composite_v${index + 1}_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        showToast('Image downloaded successfully!', 'success');
    } catch (err) {
        showToast('Download failed.', 'error');
    }
}

// ─── Print ──────────────────────────────────────────────────────────
function printImage(base64) {
    const printWin = window.open('', '_blank');
    printWin.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>ForensicAI — Suspect Composite</title>
            <style>
                body { display: flex; flex-direction: column; align-items: center; 
                       justify-content: center; min-height: 100vh; margin: 0; 
                       font-family: Arial, sans-serif; background: white; }
                h2 { margin-bottom: 10px; color: #333; }
                p { color: #666; margin-bottom: 20px; font-size: 14px; }
                img { max-width: 400px; border: 2px solid #333; border-radius: 8px; }
                .footer { margin-top: 20px; font-size: 11px; color: #999; }
            </style>
        </head>
        <body>
            <h2>Suspect Facial Composite</h2>
            <p>Generated by ForensicAI — ${new Date().toLocaleString()}</p>
            <img src="data:image/png;base64,${base64}" />
            <p class="footer">This is an AI-generated composite for investigative purposes only.</p>
        </body>
        </html>
    `);
    printWin.document.close();
    printWin.onload = () => {
        printWin.print();
    };
}

// ─── History ────────────────────────────────────────────────────────
function addToHistory(result) {
    const entry = {
        id: Date.now(),
        time: new Date().toLocaleTimeString(),
        description: result.description,
        images: result.images,
        scores: result.similarity_scores,
        attributes: result.attributes,
        generation_time_ms: result.generation_time_ms,
    };

    generationHistory.unshift(entry);
    if (generationHistory.length > MAX_HISTORY) {
        generationHistory.pop();
    }

    renderHistory();
}

function renderHistory() {
    if (generationHistory.length === 0) {
        historyEmpty.style.display = 'flex';
        historyList.innerHTML = '';
        return;
    }

    historyEmpty.style.display = 'none';
    historyList.innerHTML = '';

    generationHistory.forEach((entry, idx) => {
        const item = document.createElement('div');
        item.className = 'history-item' + (idx === 0 ? ' active' : '');
        item.onclick = () => loadHistoryEntry(entry);

        let thumbsHtml = '';
        entry.images.slice(0, 3).forEach((img) => {
            thumbsHtml += `<img class="history-thumb" src="data:image/png;base64,${img}" alt="thumbnail" />`;
        });

        item.innerHTML = `
            <div class="history-time"><i class="fas fa-clock"></i> ${entry.time}</div>
            <div class="history-desc">${entry.description}</div>
            <div class="history-thumbnails">${thumbsHtml}</div>
        `;

        historyList.appendChild(item);
    });
}

function loadHistoryEntry(entry) {
    const result = {
        images: entry.images,
        description: entry.description,
        similarity_scores: entry.scores,
        generation_time_ms: entry.generation_time_ms,
    };
    displayResults(result);

    // Update active state
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    event.currentTarget.classList.add('active');
}

function clearHistory() {
    generationHistory = [];
    renderHistory();
    showToast('History cleared.', 'success');
}

// ─── Loading State ──────────────────────────────────────────────────
function setLoading(loading) {
    isGenerating = loading;
    generateBtn.disabled = loading;

    const btnContent = generateBtn.querySelector('.btn-content');
    const btnLoading = generateBtn.querySelector('.btn-loading');

    if (loading) {
        btnContent.style.display = 'none';
        btnLoading.style.display = 'flex';
        statusBadge.querySelector('.status-text').textContent = 'Generating...';
        statusBadge.querySelector('.status-dot').style.background = 'var(--warning)';
    } else {
        btnContent.style.display = 'flex';
        btnLoading.style.display = 'none';
        statusBadge.querySelector('.status-text').textContent = 'System Ready';
        statusBadge.querySelector('.status-dot').style.background = 'var(--success)';
    }
}

// ─── Toast Notifications ────────────────────────────────────────────
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icon = type === 'success' ? 'check-circle' : 'exclamation-circle';
    toast.innerHTML = `<i class="fas fa-${icon}"></i> <span>${message}</span>`;

    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.35s ease-in forwards';
        setTimeout(() => toast.remove(), 350);
    }, 3500);
}
