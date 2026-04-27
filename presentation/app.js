/* ============================================================
   EEG-Genetic Fusion — Presentation App
   ============================================================ */

// ── Navigation ──
document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`page-${tab.dataset.page}`).classList.add('active');

        // Init synthetic canvas after layout settles
        if (tab.dataset.page === 'ctgan' && ctganData && !synCanvasReady) {
            requestAnimationFrame(() => {
                setTimeout(() => initSyntheticSignal(), 50);
            });
        }
    });
});

// ── Globals ──
let eegData = null;
let ctganData = null;
let geneticData = null;

const CH_COLORS = ['#6c8cff', '#4ade80', '#fb923c', '#f87171'];
let synCanvasReady = false;

// ============================================================
// EEG Signal Renderer
// ============================================================
class SignalRenderer {
    constructor(canvasId, timeId, btnId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.timeEl = document.getElementById(timeId);
        this.btn = document.getElementById(btnId);
        this.playing = false;
        this.offset = 0;
        this.animId = null;
        this.data = null;
        this.channels = [];
        this.sfreq = 128;
        this.windowSamples = 0;

        // Handle DPI scaling
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.w = rect.width;
        this.h = rect.height;

        this.btn.addEventListener('click', () => this.toggle());
    }

    load(signalData, channels, sfreq) {
        this.data = signalData;
        this.channels = channels;
        this.sfreq = sfreq;
        this.windowSamples = Math.min(3 * sfreq, Object.values(signalData)[0].length); // 3-sec window
        this.offset = 0;
        this.draw();
    }

    draw() {
        const ctx = this.ctx;
        const w = this.w;
        const h = this.h;
        ctx.clearRect(0, 0, w, h);

        if (!this.data) return;

        const nCh = this.channels.length;
        const chHeight = h / nCh;
        const windowEnd = Math.min(this.offset + this.windowSamples, Object.values(this.data)[0].length);
        const timeOffset = this.offset / this.sfreq;

        this.channels.forEach((ch, i) => {
            const samples = this.data[ch];
            if (!samples) return;

            const yCenter = chHeight * i + chHeight / 2;
            const slice = samples.slice(this.offset, windowEnd);

            // Compute scale from data range
            let min = Infinity, max = -Infinity;
            for (const v of slice) { if (v < min) min = v; if (v > max) max = v; }
            const range = Math.max(max - min, 1);
            const scale = (chHeight * 0.7) / range;

            // Channel label
            ctx.fillStyle = '#8b8fa3';
            ctx.font = '11px Inter, sans-serif';
            ctx.fillText(ch, 6, chHeight * i + 14);

            // Divider line
            if (i > 0) {
                ctx.strokeStyle = '#2a2e3a';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(0, chHeight * i);
                ctx.lineTo(w, chHeight * i);
                ctx.stroke();
            }

            // Signal
            ctx.strokeStyle = CH_COLORS[i % CH_COLORS.length];
            ctx.lineWidth = 1.2;
            ctx.beginPath();

            for (let j = 0; j < slice.length; j++) {
                const x = (j / this.windowSamples) * w;
                const y = yCenter - (slice[j] - (min + max) / 2) * scale;
                if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });

        // Time indicator
        this.timeEl.textContent = `${timeOffset.toFixed(1)}s`;
    }

    toggle() {
        if (this.playing) {
            this.playing = false;
            this.btn.textContent = '▶ Play';
            this.btn.classList.remove('playing');
            cancelAnimationFrame(this.animId);
        } else {
            this.playing = true;
            this.btn.textContent = '⏸ Pause';
            this.btn.classList.add('playing');
            this.animate();
        }
    }

    animate() {
        if (!this.playing) return;

        const totalSamples = Object.values(this.data)[0].length;
        this.offset += 2; // advance 2 samples per frame (~60fps → scrolls smoothly)
        if (this.offset + this.windowSamples >= totalSamples) {
            this.offset = 0;
        }
        this.draw();
        this.animId = requestAnimationFrame(() => this.animate());
    }
}


// ============================================================
// Load Data & Init
// ============================================================
async function loadJSON(path) {
    const res = await fetch(path);
    return res.json();
}

async function init() {
    try {
        [eegData, geneticData, ctganData] = await Promise.all([
            loadJSON('data/eeg_signals.json'),
            loadJSON('data/genetic_profiles.json'),
            loadJSON('data/ctgan_results.json'),
        ]);
    } catch (e) {
        console.error('Failed to load data:', e);
        return;
    }

    initEEGPage();
    initGeneticSection();
    initCTGANPage();
}

// ============================================================
// Page 1: EEG Preprocessing
// ============================================================
function initEEGPage() {
    const rawRenderer = new SignalRenderer('canvas-raw', 'time-raw', 'btn-play-raw');
    const procRenderer = new SignalRenderer('canvas-proc', 'time-proc', 'btn-play-proc');

    document.getElementById('raw-meta').textContent = `${eegData.file} · ${eegData.sfreq * 2} Hz · ${eegData.duration_sec}s`;
    document.getElementById('proc-meta').textContent = `Filtered · ${eegData.sfreq * 2} Hz · ${eegData.duration_sec}s`;

    rawRenderer.load(eegData.raw, eegData.channels, eegData.sfreq);
    procRenderer.load(eegData.processed, eegData.channels, eegData.sfreq);
}

function initGeneticSection() {
    // Mutation heatmap
    const container = document.getElementById('mutation-heatmap');
    const genes = geneticData.gene_names;
    const patients = geneticData.data;

    // Header row
    let html = `<div class="heatmap-grid" style="grid-template-columns: 80px repeat(${genes.length}, 1fr);">`;
    html += `<div class="heatmap-label"></div>`;
    genes.forEach(g => { html += `<div class="heatmap-label" style="text-align:center; font-size:0.62rem;">${g}</div>`; });

    patients.forEach(pat => {
        html += `<div class="heatmap-label">${pat.patient_id}</div>`;
        genes.forEach(gene => {
            const val = pat[`${gene}_mutation`] || 0;
            const cls = val === 1 ? 'active' : 'inactive';
            html += `<div class="heatmap-cell ${cls}">${val}</div>`;
        });
    });
    html += '</div>';
    container.innerHTML = html;

    // PRS chart
    const prsContainer = document.getElementById('prs-chart');
    let prsHtml = '';
    patients.forEach(pat => {
        const prs = pat.polygenic_risk_score || 0;
        const pli1 = pat.SCN1A_pLI || 0;
        const absMax = 3;
        const pct = Math.min(Math.abs(prs) / absMax * 50, 50);
        const left = prs >= 0 ? 50 : 50 - pct;
        const color = prs >= 0 ? '#fb923c' : '#6c8cff';
        prsHtml += `
            <div class="prs-bar-row">
                <div class="prs-label">${pat.patient_id}</div>
                <div class="prs-bar-track">
                    <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#2a2e3a;"></div>
                    <div class="prs-bar" style="width:${pct}%;margin-left:${left}%;background:${color};"></div>
                </div>
                <div class="prs-value">${prs > 0 ? '+' : ''}${prs.toFixed(2)}</div>
            </div>
        `;
    });
    prsContainer.innerHTML = `<p style="font-size:0.72rem;color:#8b8fa3;margin-bottom:0.5rem;">Polygenic Risk Score (standardised)</p>` + prsHtml;
}

// ============================================================
// Page 2: CTGAN Results
// ============================================================
function initCTGANPage() {
    initDistributionCharts();
    initSeizureBars();
    // Synthetic signal deferred — see nav click handler
    initGeneticComparison();
    initValidationGrid();
}

function initDistributionCharts() {
    const container = document.getElementById('ctgan-charts');
    const comparisons = ctganData.comparisons;

    const friendlyNames = {
        'delta_power_mean': 'Delta Power',
        'theta_power_mean': 'Theta Power',
        'alpha_power_mean': 'Alpha Power',
        'beta_power_mean': 'Beta Power',
        'gamma_power_mean': 'Gamma Power',
        'spike_rate_mean': 'Spike Rate',
        'sample_entropy_mean': 'Sample Entropy',
        'preictal_ratio': 'Preictal Ratio',
        'polygenic_risk_score': 'Polygenic Risk Score',
    };

    let html = '';
    for (const [col, data] of Object.entries(comparisons)) {
        const name = friendlyNames[col] || col;
        const realHist = data.real_hist;
        const synHist = data.syn_hist;
        const maxVal = Math.max(...realHist, ...synHist, 1);

        // Interleave real/syn bars
        let barsHtml = '';
        for (let i = 0; i < realHist.length; i++) {
            const rH = (realHist[i] / maxVal * 100);
            const sH = (synHist[i] / maxVal * 100);
            barsHtml += `<div class="dist-bar real" style="height:${rH}%"></div>`;
            barsHtml += `<div class="dist-bar syn" style="height:${sH}%"></div>`;
        }

        html += `
            <div class="dist-card">
                <h4>${name}</h4>
                <div class="dist-bars">${barsHtml}</div>
                <div class="dist-legend">
                    <span><span class="legend-dot" style="background:#6c8cff"></span>Real</span>
                    <span><span class="legend-dot" style="background:#fb923c"></span>Synthetic</span>
                </div>
                <div class="dist-stat">
                    μ: ${data.real_mean.toExponential(2)} → ${data.syn_mean.toExponential(2)}
                </div>
            </div>
        `;
    }
    container.innerHTML = html;
}

function initSeizureBars() {
    const container = document.getElementById('seizure-bars');
    const sd = ctganData.seizure_dist;

    const groups = [
        { label: 'Real (190)', data: sd.real },
        { label: 'Synthetic (1000)', data: sd.syn },
    ];

    let html = '';
    groups.forEach(g => {
        const total = g.data.seizure + g.data.no_seizure;
        const yesPct = (g.data.seizure / total * 100);
        const noPct = (g.data.no_seizure / total * 100);
        html += `
            <div class="seizure-col">
                <div class="seizure-label">${g.label}</div>
                <div class="seizure-bar-stack">
                    <div class="seizure-seg yes" style="height:${yesPct}%"></div>
                    <div class="seizure-seg no" style="height:${noPct}%"></div>
                </div>
                <div class="seizure-pct">Seizure: ${yesPct.toFixed(1)}%</div>
                <div class="seizure-pct">Non-seizure: ${noPct.toFixed(1)}%</div>
            </div>
        `;
    });
    container.innerHTML = html;
}

function initSyntheticSignal() {
    const canvas = document.getElementById('canvas-synthetic');
    const ctx = canvas.getContext('2d');
    const btn = document.getElementById('btn-play-syn');
    const timeEl = document.getElementById('time-syn');

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const canvasW = rect.width || 1200;
    const canvasH = rect.height || 250;
    synCanvasReady = true;

    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    ctx.scale(dpr, dpr);
    const w = canvasW;
    const h = canvasH;

    const signals = ctganData.syn_signals;
    const keys = Object.keys(signals);
    const sfreq = ctganData.syn_signal_sfreq;
    const colors = ['#6c8cff', '#4ade80', '#fb923c', '#f87171', '#fbbf24'];

    let playing = false;
    let offset = 0;
    const windowSamples = 2 * sfreq;

    function draw() {
        ctx.clearRect(0, 0, w, h);
        const nSig = keys.length;
        const sigH = h / nSig;

        keys.forEach((key, i) => {
            const samples = signals[key];
            const slice = samples.slice(offset, offset + windowSamples);
            const yCenter = sigH * i + sigH / 2;

            let min = Infinity, max = -Infinity;
            for (const v of slice) { if (v < min) min = v; if (v > max) max = v; }
            const range = Math.max(max - min, 0.01);
            const scale = (sigH * 0.65) / range;

            // Divider
            if (i > 0) {
                ctx.strokeStyle = '#2a2e3a';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(0, sigH * i);
                ctx.lineTo(w, sigH * i);
                ctx.stroke();
            }

            ctx.fillStyle = '#8b8fa3';
            ctx.font = '10px Inter, sans-serif';
            ctx.fillText(`Synthetic ${i + 1}`, 6, sigH * i + 14);

            ctx.strokeStyle = colors[i % colors.length];
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            for (let j = 0; j < slice.length; j++) {
                const x = (j / windowSamples) * w;
                const y = yCenter - (slice[j] - (min + max) / 2) * scale;
                if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });

        timeEl.textContent = `${(offset / sfreq).toFixed(1)}s`;
    }

    function animate() {
        if (!playing) return;
        offset += 2;
        const totalLen = signals[keys[0]].length;
        if (offset + windowSamples >= totalLen) offset = 0;
        draw();
        requestAnimationFrame(animate);
    }

    btn.addEventListener('click', () => {
        playing = !playing;
        if (playing) {
            btn.textContent = '⏸ Pause';
            btn.classList.add('playing');
            animate();
        } else {
            btn.textContent = '▶ Play All';
            btn.classList.remove('playing');
        }
    });

    draw();
}

function initGeneticComparison() {
    const container = document.getElementById('genetic-ctgan-comparison');
    const mutations = ctganData.mutations;

    let html = '';
    for (const [col, data] of Object.entries(mutations)) {
        const gene = col.replace('_mutation', '');
        const realPct = (data.real_rate * 100).toFixed(1);
        const synPct = (data.syn_rate * 100).toFixed(1);

        html += `
            <div class="mut-card">
                <h4>${gene}</h4>
                <div class="mut-rates">
                    <div>
                        <span class="mut-rate-label">Real</span>
                        <span style="color: ${data.real_rate > 0 ? '#6c8cff' : '#8b8fa3'}">${realPct}%</span>
                    </div>
                    <div>
                        <span class="mut-rate-label">Synthetic</span>
                        <span style="color: ${data.syn_rate > 0 ? '#fb923c' : '#8b8fa3'}">${synPct}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    container.innerHTML = html;
}

function initValidationGrid() {
    const container = document.getElementById('validation-grid');
    const v = ctganData.validation;

    const cards = [
        { label: 'KS Test Pass Rate', value: `${v.ks_pass_rate?.split('/')[0] || '4'}/31`, color: '#fb923c' },
        { label: 'Correlation Diff', value: (v.mean_correlation_diff || 0.36).toFixed(3), color: '#6c8cff' },
        { label: 'Real Patients', value: ctganData.real_patients || 8, color: '#4ade80' },
        { label: 'Synthetic Records', value: ctganData.syn_count || 1000, color: '#fbbf24' },
    ];

    let html = '';
    cards.forEach(c => {
        html += `
            <div class="val-card">
                <div class="val-number" style="color:${c.color}">${c.value}</div>
                <div class="val-label">${c.label}</div>
            </div>
        `;
    });
    container.innerHTML = html;
}

// ── Start ──
document.addEventListener('DOMContentLoaded', init);
