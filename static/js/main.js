/* ═══════════════════════════════════════════════════════════
   AI Video Analytics Dashboard — Main JavaScript
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initAnimatedCounters();
    initScrollAnimations();
    initMobileMenu();

    // Dynamic Dashboard Updates
    if (document.getElementById('syncStatus')) {
        syncSystemStatus();
        setInterval(syncSystemStatus, 15000); // Pulse every 15s
    }

    // Initialize Module Live Demos
    initUploadZone('vehicleUpload', '/api/detect-vehicles', renderVehicleResults);
    initUploadZone('anprUpload', '/api/anpr', renderANPRResults);
    initUploadZone('faceUpload', '/api/recognize-face', renderFaceResults);
    initUploadZone('peopleUpload', '/api/count-people', renderPeopleResults);
    initUploadZone('weaponUpload', '/api/detect-weapons', renderWeaponResults);
    initUploadZone('behaviorUpload', '/api/analyze-behavior', renderBehaviorResults);
    initUploadZone('helmetUploadZone', '/api/detect-helmets', renderHelmetResults);
});

/* ── Animated Counters ────────────────────────────────────── */
function initAnimatedCounters() {
    const counters = document.querySelectorAll('.count-up');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(el => observer.observe(el));
}

function animateCounter(el) {
    const target = parseInt(el.dataset.target) || 0;
    const suffix = el.dataset.suffix || '';
    const prefix = el.dataset.prefix || '';
    const duration = 1500;
    const start = performance.now();

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 4);
        const current = Math.floor(eased * target);
        el.textContent = prefix + current.toLocaleString() + suffix;
        if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
}

/* ── Scroll Animations ────────────────────────────────────── */
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}

/* ── Mobile Menu ──────────────────────────────────────────── */
function initMobileMenu() {
    const toggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');

    if (toggle && sidebar) {
        toggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });

        document.addEventListener('click', (e) => {
            if (sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                !toggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }
}

/* ── Tab Switching ────────────────────────────────────────── */
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    const btn = document.querySelector(`[data-tab="${tabId}"]`);
    const pane = document.getElementById(tabId);

    if (btn) btn.classList.add('active');
    if (pane) pane.classList.add('active');
}

/* ── File Upload & Drag-Drop ──────────────────────────────── */
function initUploadZone(zoneId, endpoint, renderCallback) {
    const zone = document.getElementById(zoneId);
    if (!zone) return;

    const input = zone.querySelector('input[type="file"]');

    // Drag events
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            // Check for a corresponding Night Vision toggle
            let nvFlag = false;
            const nvToggle = document.getElementById(zoneId.replace('Upload', 'NV'));
            if (nvToggle) nvFlag = nvToggle.checked;

            processUpload(e.dataTransfer.files[0], endpoint, renderCallback, nvFlag);
        }
    });

    // Click upload
    if (!input) input = document.querySelector(`#${zoneId} input[type="file"]`);

    if (input) {
        input.addEventListener('change', (e) => {
            if (e.target.files.length) {
                let nvFlag = false;
                const nvToggle = document.getElementById(zoneId.replace('Upload', 'NV'));
                if (nvToggle) nvFlag = nvToggle.checked;

                processUpload(e.target.files[0], endpoint, renderCallback, nvFlag);
            }
        });
    }
}

function processUpload(file, endpoint, renderCallback, nightVision = false) {
    const validTypes = [
        'image/jpeg', 'image/png', 'image/bmp', 'image/webp',
        'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/webm'
    ];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a valid image or video file', 'error');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showNotification('File too large. Max 50MB.', 'error');
        return;
    }

    const isVideo = file.type.startsWith('video/');

    // For individual modules, if it's a video, we do LIVE analysis side-by-side
    // instead of showing a loading overlay and waiting for a full process.
    if (isVideo && !endpoint.includes('analyze-video')) {
        startLiveVideoAnalysis(file, endpoint, renderCallback, nightVision);
        return;
    }

    const loadingText = isVideo ? 'Analyzing video frames with AI models (this may take a moment)...' : 'Analyzing image with AI models...';
    const loadingEl = document.querySelector('#loadingOverlay .loading-text');
    if (loadingEl) loadingEl.textContent = loadingText;

    showLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('night_vision', nightVision);

    fetch(endpoint, { method: 'POST', body: formData })
        .then(res => {
            if (!res.ok) throw new Error(`Server error: ${res.status}`);
            return res.json();
        })
        .then(data => {
            showLoading(false);
            if (data.error) {
                showNotification(data.error, 'error');
            } else {
                renderCallback(data);
                showNotification('Analysis complete!', 'success');
            }
        })
        .catch(err => {
            showLoading(false);
            showNotification('Analysis failed: ' + err.message, 'error');
        });
}

/**
 * LIVE side-by-side analysis for individual modules in modules.html
 */
let _liveAnalyzers = new Map(); // Store intervals/states
let _sessionResults = {
    vehicleResults: { ids: new Set(), detections: [] },
    anprResults: { plates: new Set(), detections: [] },
    faceResults: { keys: new Set(), detections: [] },
    peopleResults: { peakCount: 0, genderCounts: { Male: 0, Female: 0 } },
    weaponResults: { items: new Set(), detections: [] },
    behaviorResults: { ids: new Set(), detections: [] },
    helmetResults: { ids: new Set(), detections: [] }
};

function startLiveVideoAnalysis(file, endpoint, renderCallback, nightVision) {
    // Determine which module results panel contains the source
    let panelId = "";
    if (endpoint.includes('detect-vehicles')) panelId = "vehicleResults";
    else if (endpoint.includes('anpr')) panelId = "anprResults";
    else if (endpoint.includes('recognize-face')) panelId = "faceResults";
    else if (endpoint.includes('count-people')) panelId = "peopleResults";
    else if (endpoint.includes('detect-weapons')) panelId = "weaponResults";
    else if (endpoint.includes('analyze-behavior')) panelId = "behaviorResults";
    else if (endpoint.includes('detect-helmets')) panelId = "helmetResults";

    // Reset session results for this module
    if (_sessionResults[panelId]) {
        if (panelId === 'peopleResults') {
            _sessionResults[panelId] = { peakCount: 0, genderCounts: { Male: 0, Female: 0 } };
        } else {
            const listKey = 'detections';
            const key = panelId === 'vehicleResults' ? 'ids' :
                (panelId === 'anprResults' ? 'plates' :
                    (panelId === 'faceResults' ? 'keys' :
                        (panelId === 'behaviorResults' ? 'ids' : (panelId === 'helmetResults' ? 'ids' : 'items'))));
            _sessionResults[panelId] = { [key]: new Set(), [listKey]: [] };
        }
    }

    const panel = document.getElementById(panelId);
    if (!panel) return;

    panel.classList.add('visible');
    panel.classList.add('live-analysis-mode'); // Enable full-width/side-by-side layout

    let container = panel.querySelector('.result-image-container');
    const imgEl = panel.querySelector('.result-image');

    // Create a NEW container for the original video if it doesn't exist
    let videoContainer = panel.querySelector('.live-video-container');
    if (!videoContainer) {
        videoContainer = document.createElement('div');
        videoContainer.className = 'result-image-container live-video-container';
        videoContainer.innerHTML = `
            <div class="result-image-label">
                <i class="fas fa-video"></i> Original Video Feed
            </div>
        `;
        // Insert it BEFORE the result-image-container in the results-grid
        container.parentNode.insertBefore(videoContainer, container);
    }

    // Clear existing video if any
    const oldVideo = videoContainer.querySelector('video.live-module-video');
    if (oldVideo) oldVideo.remove();

    // Create video element
    const video = document.createElement('video');
    video.className = 'live-module-video';
    video.src = URL.createObjectURL(file);
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.controls = true;
    video.style.width = '100%';
    video.style.height = '100%';
    video.style.objectFit = 'contain';

    videoContainer.appendChild(video);

    const canvas = document.createElement('canvas');
    let isProcessing = false;

    const runFrameLoop = () => {
        // Clear any existing analyzer for transparency
        if (_liveAnalyzers.get(panelId)) {
            _liveAnalyzers.get(panelId).active = false;
        }

        const state = { active: true };
        _liveAnalyzers.set(panelId, state);

        const processFrame = async () => {
            if (!state.active || video.paused || video.ended) {
                if (state.active) setTimeout(processFrame, 500);
                return;
            }

            if (isProcessing || video.videoWidth === 0) {
                if (state.active) requestAnimationFrame(processFrame);
                return;
            }

            isProcessing = true;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            try {
                const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.8));
                if (!blob) throw new Error("Canvas to Blob failed");

                const fd = new FormData();
                fd.append('file', blob, 'frame.jpg');
                fd.append('night_vision', nightVision);
                fd.append('high_throughput', 'true');

                const res = await fetch(endpoint, { method: 'POST', body: fd });
                if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
                const data = await res.json();

                if (data.error) console.error("API Error:", data.error);
                else {
                    accumulateSessionResults(panelId, data);
                    renderCallback(data);
                    const annotated = data.annotated_image;
                    if (annotated && imgEl) {
                        imgEl.src = 'data:image/jpeg;base64,' + annotated;
                    }
                }
            } catch (e) {
                console.error("Frame analysis error:", e);
            } finally {
                isProcessing = false;
                if (state.active) setTimeout(processFrame, 125); // Faster processing for better responsiveness
            }
        };

        processFrame();
    };

    video.onplay = runFrameLoop;
    video.onpause = () => { if (_liveAnalyzers.get(panelId)) _liveAnalyzers.get(panelId).active = false; };
    video.onended = () => {
        if (_liveAnalyzers.get(panelId)) _liveAnalyzers.get(panelId).active = false;
        const session = _sessionResults[panelId];
        let summary = "";
        if (panelId === 'anprResults') summary = `${session.detections.length} plates detected.`;
        else if (panelId === 'faceResults') summary = `${session.detections.length} faces identified.`;
        else if (panelId === 'vehicleResults') summary = `${session.detections.length} vehicles tracked.`;
        else if (panelId === 'peopleResults') summary = `Peak count: ${session.peakCount}.`;
        else if (panelId === 'helmetResults') summary = `${session.detections.length} rider events tracked.`;
        showNotification(`Analysis Complete: ${summary}`, 'success');
    };

    video.play().catch(err => {
        console.warn("Autoplay blocked:", err);
        showNotification("Please click play on the video to start AI analysis", "info");
    });

    showNotification('Live Video Analysis Started', 'success');
}

function accumulateSessionResults(panelId, data) {
    if (panelId === 'vehicleResults' && data.detections) {
        data.detections.forEach(v => {
            if (!_sessionResults.vehicleResults.ids.has(v.id)) {
                _sessionResults.vehicleResults.ids.add(v.id);
                _sessionResults.vehicleResults.detections.push(v);
            }
        });
    } else if (panelId === 'anprResults' && data.plates) {
        data.plates.forEach(p => {
            if (!_sessionResults.anprResults.plates.has(p.text)) {
                _sessionResults.anprResults.plates.add(p.text);
                _sessionResults.anprResults.detections.push(p);
            }
        });
    } else if (panelId === 'faceResults' && data.faces) {
        data.faces.forEach(f => {
            if (f.name && f.name !== 'Unmatched Person') {
                if (!_sessionResults.faceResults.keys.has(f.name)) {
                    _sessionResults.faceResults.keys.add(f.name);
                    _sessionResults.faceResults.detections.unshift(f);
                }
            } else {
                // For Unmatched Person, use a time-bound ID to avoid duplicates but allow new ones
                const unmatchedKey = `Face_${f.id}_${Math.floor(Date.now() / 2000)}`;
                if (!_sessionResults.faceResults.keys.has(unmatchedKey)) {
                    _sessionResults.faceResults.keys.add(unmatchedKey);
                    _sessionResults.faceResults.detections.unshift(f);
                }
            }
        });
        if (_sessionResults.faceResults.detections.length > 100) {
            _sessionResults.faceResults.detections = _sessionResults.faceResults.detections.slice(0, 100);
        }
    } else if (panelId === 'peopleResults') {
        if (data.total_people > _sessionResults.peopleResults.peakCount) {
            _sessionResults.peopleResults.peakCount = data.total_people;
        }
        if (data.gender_counts) {
            for (let g in data.gender_counts) {
                _sessionResults.peopleResults.genderCounts[g] = Math.max(
                    _sessionResults.peopleResults.genderCounts[g] || 0,
                    data.gender_counts[g]
                );
            }
        }
    } else if (panelId === 'behaviorResults' && data.detections) {
        data.detections.forEach(b => {
            const behaviorPriority = {
                'FALL DETECTED': 5,
                'FIGHTING': 4,
                'AGGRESSIVE': 3,
                'LOITERING': 2,
                'Normal': 1
            };

            if (!_sessionResults.behaviorResults.ids.has(b.id)) {
                _sessionResults.behaviorResults.ids.add(b.id);
                _sessionResults.behaviorResults.detections.push(b);
            } else {
                const idx = _sessionResults.behaviorResults.detections.findIndex(d => d.id === b.id);
                if (idx !== -1) {
                    const currentBehavior = _sessionResults.behaviorResults.detections[idx].behavior;
                    const newBehavior = b.behavior;

                    // Only update if the new behavior is higher or equal priority
                    // This keeps the "FIGHTING" label even if they stop fighting later in the session log
                    if (behaviorPriority[newBehavior] >= behaviorPriority[currentBehavior]) {
                        _sessionResults.behaviorResults.detections[idx] = b;
                    }
                }
            }
        });
    } else if (panelId === 'helmetResults' && data.detections) {
        data.detections.forEach(h => {
            const hKey = `${h.class}_${h.bbox.join('_')}`;
            if (!_sessionResults.helmetResults.ids.has(hKey)) {
                _sessionResults.helmetResults.ids.add(hKey);
                _sessionResults.helmetResults.detections.push(h);
            }
        });
    }
}

/* ── Loading Overlay ──────────────────────────────────────── */
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.toggle('visible', show);
    }
}

/* ── Notification Toast ───────────────────────────────────── */
function showNotification(message, type = 'info') {
    // Remove existing
    const existing = document.querySelector('.notification-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `notification-toast ${type}`;
    toast.style.cssText = `
        position: fixed;
        top: 24px;
        right: 24px;
        z-index: 10000;
        padding: 14px 24px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 500;
        backdrop-filter: blur(16px);
        animation: slideInRight 0.3s ease;
        max-width: 400px;
        display: flex;
        align-items: center;
        gap: 10px;
    `;

    if (type === 'success') {
        toast.style.background = 'rgba(6, 214, 160, 0.15)';
        toast.style.border = '1px solid rgba(6, 214, 160, 0.3)';
        toast.style.color = '#06d6a0';
        toast.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    } else if (type === 'error') {
        toast.style.background = 'rgba(239, 68, 68, 0.15)';
        toast.style.border = '1px solid rgba(239, 68, 68, 0.3)';
        toast.style.color = '#ef4444';
        toast.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    } else {
        toast.style.background = 'rgba(59, 130, 246, 0.15)';
        toast.style.border = '1px solid rgba(59, 130, 246, 0.3)';
        toast.style.color = '#3b82f6';
        toast.innerHTML = `<i class="fas fa-info-circle"></i> ${message}`;
    }

    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Toast animations
const toastStyle = document.createElement('style');
toastStyle.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(toastStyle);

/* ── Dynamic System Status ────────────────────────────────── */
function syncSystemStatus() {
    fetch('/api/health')
        .then(r => r.json())
        .then(data => {
            // Update Online Status Dot
            const dots = document.querySelectorAll('.status-dot');
            dots.forEach(dot => {
                dot.style.background = data.status === 'online' ? '#06d6a0' : '#ef4444';
                dot.style.boxShadow = data.status === 'online' ? '0 0 10px #06d6a0' : '0 0 10px #ef4444';
            });

            const statusText = document.querySelector('.system-status');
            if (statusText) {
                statusText.innerHTML = `<span class="status-dot" style="background:${data.status === 'online' ? '#06d6a0' : '#ef4444'}"></span> System ${data.status.toUpperCase()} — ${data.modules.length} Modules Active`;
            }

            // Update Counter on Dashboard if present
            const moduleCountEl = document.querySelector('.stat-card.cyan .count-up');
            if (moduleCountEl) {
                moduleCountEl.textContent = data.modules.length;
            }
        })
        .catch(err => console.error('Health check failed:', err));
}

/* ── Render Functions for Module Results ──────────────────── */

// Vehicle Detection Results
function renderVehicleResults(data) {
    const panel = document.getElementById('vehicleResults');
    if (!panel) return;
    panel.classList.add('visible');

    // Use session totals if in live mode
    const isLive = panel.classList.contains('live-analysis-mode');
    const session = _sessionResults.vehicleResults;

    // Annotated image
    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    // Total count
    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = isLive ? session.detections.length : data.total;

    // Per-class counts / Detections
    const listEl = panel.querySelector('.detections-list');
    const detectionsToShow = isLive ? session.detections : (data.detections || []);

    if (listEl) {
        const colors = { car: '#06d6a0', motorcycle: '#f59e0b', bus: '#3b82f6', truck: '#8b5cf6' };

        // Calculate breakdown for the summary box
        const breakdown = { car: 0, motorcycle: 0, bus: 0, truck: 0 };
        detectionsToShow.forEach(det => {
            if (breakdown.hasOwnProperty(det.class)) breakdown[det.class]++;
        });

        // Insert Summary Box at the top
        let summaryHtml = `
            <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 12px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 8px;">Classification Breakdown</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px;">
                    ${Object.entries(breakdown).filter(([_, count]) => count > 0).map(([cls, count]) => `
                        <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.3); padding: 6px 12px; border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;">
                            <span style="font-size: 11px; color: ${colors[cls]}"><i class="fas fa-${cls === 'motorcycle' ? 'motorcycle' : cls === 'bus' ? 'bus' : cls === 'truck' ? 'truck' : 'car'}"></i> ${cls.toUpperCase()}</span>
                            <span style="font-weight: 700; font-family: 'JetBrains Mono'; color: var(--text-primary);">${count}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; padding-left: 4px;">Individual Logs</div>
        `;

        listEl.innerHTML = summaryHtml + (detectionsToShow.map(det => `
            <div class="detection-item" style="border-left:3px solid ${colors[det.class] || '#06d6a0'}; padding-left:12px">
                <div style="display:flex; flex-direction:column; gap:2px">
                    <span class="detection-item-label" style="font-weight:700">
                        ${det.class.charAt(0).toUpperCase() + det.class.slice(1)} #${det.id}
                    </span>
                    <div style="display:flex; gap:8px; font-size:11px; color:var(--text-muted)">
                        <span>Conf: ${(det.confidence * 100).toFixed(0)}%</span>
                        ${det.speed ? `<span style="color:var(--accent-cyan); font-weight:700"><i class="fas fa-tachometer-alt"></i> ${det.speed} km/h</span>` : ''}
                    </div>
                </div>
            </div>
        `).join('') || '<div style="color:var(--text-muted); font-size:13px; padding:10px">No vehicles identified</div>');
    }

    // Update chart if exists
    if (window.vehicleChart && data.counts) {
        window.vehicleChart.data.datasets[0].data = Object.values(data.counts);
        window.vehicleChart.update();
    }
}

// Helmet Detection Results
function renderHelmetResults(data) {
    const panel = document.getElementById('helmetResults');
    if (!panel) return;
    panel.style.display = 'block';

    const imgEl = panel.querySelector('.result-image');
    const totalEl = panel.querySelector('.total-count');
    const listEl = panel.querySelector('.detections-list');

    if (data.annotated_image && imgEl) {
        imgEl.src = `data:image/jpeg;base64,${data.annotated_image}`;
    }

    if (totalEl) {
        totalEl.textContent = data.total_riders || 0;
    }

    const detections = data.detections || [];
    if (listEl) {
        listEl.innerHTML = detections.map(d => `
            <div class="detection-item" style="border-left:3px solid ${d.is_violation ? '#ef4444' : '#06d6a0'}">
                <div style="display:flex; flex-direction:column; gap:4px">
                    <span style="font-weight:700; color:${d.is_violation ? '#ef4444' : '#06d6a0'}">
                        <i class="fas fa-${d.is_violation ? 'exclamation-triangle' : 'check-circle'}"></i> 
                        ${d.class}
                    </span>
                    <span style="font-size:11px; color:var(--text-muted)">Confidence: ${(d.confidence * 100).toFixed(1)}%</span>
                </div>
            </div>
        `).join('') || '<div style="color:var(--text-muted); text-align:center; padding:20px;">No pertinent objects detected</div>';
    }
}

// ANPR Results
function renderANPRResults(data) {
    const panel = document.getElementById('anprResults');
    if (!panel) return;
    panel.classList.add('visible');

    const isLive = panel.classList.contains('live-analysis-mode');
    const session = _sessionResults.anprResults;

    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = isLive ? session.detections.length : data.total_plates;

    const listEl = panel.querySelector('.detections-list');
    const platesToShow = isLive ? session.detections : (data.plates || []);

    if (listEl) {
        listEl.innerHTML = platesToShow.map(plate => `
            <div class="detection-item" style="border-left: 3px solid ${plate.is_blacklisted ? '#ef4444' : 'transparent'}; padding-left: 8px;">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:${plate.is_blacklisted ? '#ef4444' : '#3b82f6'}"></span>
                    <strong style="${plate.is_blacklisted ? 'color:#ef4444;' : ''}">${plate.text}</strong>
                    ${plate.is_blacklisted ? '<span style="background:rgba(239,68,68,0.2);color:#ef4444;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:6px;font-weight:700;">BLACKLISTED</span>' : ''}
                </span>
                <span class="detection-conf">${(plate.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('') || '<div class="detection-item"><span style="color:var(--text-muted)">No plates detected</span></div>';
    }
}

// Face Recognition Results
function renderFaceResults(data) {
    const panel = document.getElementById('faceResults');
    if (!panel) return;
    panel.classList.add('visible');

    const isLive = panel.classList.contains('live-analysis-mode');
    const session = _sessionResults.faceResults;

    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = isLive ? session.detections.length : data.total_faces;

    const listEl = panel.querySelector('.detections-list');
    const facesToShow = isLive ? session.detections : (data.faces || []);

    if (listEl) {
        listEl.innerHTML = facesToShow.map(face => `
            <div class="detection-item" style="flex-direction:column;align-items:flex-start;gap:6px;border-left: 3px solid ${face.is_blacklisted ? '#ef4444' : face.is_authorized ? '#06d6a0' : 'transparent'}; padding-left: 8px; margin-bottom: 8px;">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:${face.is_blacklisted ? '#ef4444' : face.is_authorized ? '#06d6a0' : '#8b5cf6'}"></span>
                    <strong style="${face.is_blacklisted ? 'color:#ef4444;' : ''}">${face.name && face.name !== 'Unmatched Person' ? face.name : 'Face #' + face.id}</strong>
                    ${face.is_blacklisted ? '<span style="background:rgba(239,68,68,0.2);color:#ef4444;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:6px;font-weight:700;">BLACKLISTED</span>' : face.is_authorized ? '<span style="background:rgba(6,214,160,0.2);color:#06d6a0;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:6px;font-weight:700;">AUTHORIZED</span>' : ''}
                </span>
                <div style="display:flex;gap:8px;flex-wrap:wrap;padding-left:16px">
                    <span class="meta-tag"><i class="fas fa-birthday-cake"></i> Age: ${face.age}</span>
                    <span class="meta-tag"><i class="fas fa-venus-mars"></i> ${face.gender}</span>
                    <span class="meta-tag"><i class="fas fa-smile"></i> ${face.emotion}</span>
                </div>
            </div>
        `).join('') || '<div style="color:var(--text-muted); font-size:13px; padding:10px">No faces identified</div>';
    }
}

// Weapon Detection Results
function renderWeaponResults(data) {
    const panel = document.getElementById('weaponResults');
    if (!panel) return;
    panel.classList.add('visible');

    const isLive = panel.classList.contains('live-analysis-mode');
    const session = _sessionResults.weaponResults;

    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = isLive ? session.detections.length : data.total_weapons;

    const statusEl = panel.querySelector('.status-text');
    if (statusEl) {
        statusEl.textContent = data.status === 'alert' ? 'DANGER DETECTED' : 'Clear';
        statusEl.style.color = data.status === 'alert' ? '#ef4444' : '#06d6a0';
    }

    const listEl = panel.querySelector('.detections-list');
    const itemsToShow = isLive ? session.detections : (data.detections || []);

    if (listEl) {
        listEl.innerHTML = itemsToShow.map((item, idx) => `
            <div class="detection-item" style="border-left: 3px solid #ef4444; padding-left: 10px;">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:#ef4444"></span>
                    <strong>${item.class.toUpperCase()}</strong>
                </span>
                <span class="meta-tag"><i class="fas fa-bullseye"></i> ${Math.round(item.confidence * 100)}% Conf.</span>
            </div>
        `).join('') || '<div style="color:var(--text-muted); font-size:13px; padding:10px">No high-risk items found</div>';
    }
}

// Behavior Analysis Results
function renderBehaviorResults(data) {
    const panel = document.getElementById('behaviorResults');
    if (!panel) return;
    panel.classList.add('visible');

    const isLive = panel.classList.contains('live-analysis-mode');
    const session = _sessionResults.behaviorResults;

    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = isLive ? session.detections.length : data.total_active;

    const listEl = panel.querySelector('.detections-list');
    const itemsToShow = isLive ? session.detections : (data.detections || []);

    const BEHAVIOR_COLORS = {
        'Normal': '#06d6a0',
        'FALL DETECTED': '#ef4444',
        'FIGHTING': '#f97316',
        'LOITERING': '#f59e0b',
        'AGGRESSIVE': '#8b5cf6', // Purple
    };
    const BEHAVIOR_ICONS = {
        'Normal': 'check-circle',
        'FALL DETECTED': 'person-falling',
        'FIGHTING': 'hand-fist',
        'LOITERING': 'clock',
        'AGGRESSIVE': 'bolt',
    };

    if (listEl) {
        // Sort by priority so Fighting and Falls are at the top
        const sortedItems = [...itemsToShow].sort((a, b) => {
            const pA = BEHAVIOR_COLORS[a.behavior] ? (a.behavior === 'FALL DETECTED' ? 10 : (a.behavior === 'FIGHTING' ? 9 : 5)) : 0;
            const pB = BEHAVIOR_COLORS[b.behavior] ? (b.behavior === 'FALL DETECTED' ? 10 : (b.behavior === 'FIGHTING' ? 9 : 5)) : 0;
            return pB - pA;
        });

        listEl.innerHTML = sortedItems.map((item) => {
            const bColor = BEHAVIOR_COLORS[item.behavior] || '#ef4444';
            const bIcon = BEHAVIOR_ICONS[item.behavior] || 'exclamation-triangle';
            const confPct = item.confidence ? (item.confidence * 100).toFixed(0) + '%' : '';
            return `
            <div class="detection-item" style="border-left: 3px solid ${bColor}; padding-left: 10px; background: ${bColor}08;">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:${bColor}"></span>
                    <strong>PERSON #${item.id}</strong>
                </span>
                <div style="display:flex; flex-direction:column; gap:4px">
                    <span class="meta-tag" style="background:${bColor}20; color:${bColor}; font-weight:600;">
                        <i class="fas fa-${bIcon}"></i> ${item.behavior}${confPct ? ' — ' + confPct : ''}
                    </span>
                    <span class="meta-tag"><i class="fas fa-clock"></i> Active: ${item.duration}s</span>
                </div>
            </div>`;
        }).join('') || '<div style="color:var(--text-muted); font-size:13px; padding:10px">No behavioral data</div>';
    }

    // Show summary banner
    const statusEl = panel.querySelector('.status-text');
    if (statusEl) {
        const alerts = (data.detections || []).filter(d => d.behavior !== 'Normal');
        if (alerts.length > 0) {
            const labels = [...new Set(alerts.map(a => a.behavior))].join(' + ');
            statusEl.textContent = '⚠ ' + labels;
            statusEl.style.color = '#ef4444';
        } else {
            statusEl.textContent = 'All Clear';
            statusEl.style.color = '#06d6a0';
        }
    }
}


// People Counter Results
function renderPeopleResults(data) {
    const panel = document.getElementById('peopleResults');
    if (!panel) return;
    panel.classList.add('visible');

    const isLive = panel.classList.contains('live-analysis-mode');
    const session = _sessionResults.peopleResults;

    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    const heatmapEl = panel.querySelector('.heatmap-image');
    if (heatmapEl) heatmapEl.src = 'data:image/jpeg;base64,' + data.heatmap_image;

    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = isLive ? session.peakCount : data.total_people;

    const densityEl = panel.querySelector('.density-value');
    if (densityEl) densityEl.textContent = data.density;

    // Gender Distribution
    const genderEl = panel.querySelector('.gender-stats');
    if (genderEl) {
        const counts = isLive ? session.genderCounts : (data.gender_counts || {});
        genderEl.innerHTML = `
            <div style="display:flex; gap:15px; margin-top:10px">
                <div style="display:flex; align-items:center; gap:6px; color:#3b82f6"><i class="fas fa-mars"></i> Male: ${counts.Male || 0}</div>
                <div style="display:flex; align-items:center; gap:6px; color:#ec4899"><i class="fas fa-venus"></i> Female: ${counts.Female || 0}</div>
            </div>
        `;
    }
}

/* ── Chart.js Initialization Helpers ──────────────────────── */
function createDoughnutChart(canvasId, labels, dataValues, colors) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: dataValues,
                backgroundColor: colors,
                borderColor: 'rgba(10,14,26,0.8)',
                borderWidth: 3,
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 },
                        padding: 16,
                        usePointStyle: true,
                        pointStyleWidth: 10
                    }
                }
            }
        }
    });
}

function createBarChart(canvasId, labels, datasets) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 },
                        usePointStyle: true
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
                    grid: { color: 'rgba(255,255,255,0.03)' }
                },
                y: {
                    ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
                    grid: { color: 'rgba(255,255,255,0.03)' }
                }
            }
        }
    });
}

function createRadarChart(canvasId, labels, dataValues, label) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: dataValues,
                backgroundColor: 'rgba(6, 214, 160, 0.1)',
                borderColor: '#06d6a0',
                borderWidth: 2,
                pointBackgroundColor: '#06d6a0',
                pointBorderColor: '#0a0e1a',
                pointBorderWidth: 2,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { display: false },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    pointLabels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 11 }
                    },
                    angleLines: { color: 'rgba(255,255,255,0.05)' }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 }
                    }
                }
            }
        }
    });
}

/* ── Progress Bar Animation ───────────────────────────────── */
function animateProgressBars() {
    const bars = document.querySelectorAll('.progress-bar-fill');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                bar.style.width = bar.dataset.width || '0%';
                observer.unobserve(bar);
            }
        });
    }, { threshold: 0.5 });

    bars.forEach(bar => {
        bar.style.width = '0%';
        observer.observe(bar);
    });
}

document.addEventListener('DOMContentLoaded', animateProgressBars);

/* ── Blacklist / Whitelist Management ─────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    loadBlacklist();
    loadWhitelist();

    const addBlacklistBtn = document.getElementById('addBlacklistBtn');
    if (addBlacklistBtn) {
        addBlacklistBtn.addEventListener('click', () => {
            const plate = document.getElementById('blacklistInput').value.trim();
            if (!plate) return;
            fetch('/api/blacklist/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plate })
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    document.getElementById('blacklistInput').value = '';
                    loadBlacklist();
                    showNotification('Plate added to blacklist', 'success');
                }
            });
        });
    }

    const addWhitelistBtn = document.getElementById('addWhitelistBtn');
    if (addWhitelistBtn) {
        addWhitelistBtn.addEventListener('click', () => {
            const name = document.getElementById('whitelistName').value.trim();
            const fileInput = document.getElementById('whitelistImage');
            if (!name || !fileInput.files.length) {
                showNotification('Please provide name and image', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('name', name);
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('image', fileInput.files[i]);
            }

            showLoading(true);
            fetch('/api/whitelist/add', {
                method: 'POST',
                body: formData
            }).then(r => r.json()).then(data => {
                showLoading(false);
                if (data.success) {
                    document.getElementById('whitelistName').value = '';
                    fileInput.value = '';
                    loadWhitelist();
                    showNotification('Person added to whitelist', 'success');
                } else {
                    showNotification(data.error || 'Failed to add', 'error');
                }
            });
        });
    }

    const addFaceBlacklistBtn = document.getElementById('addFaceBlacklistBtn');
    if (addFaceBlacklistBtn) {
        addFaceBlacklistBtn.addEventListener('click', () => {
            const name = document.getElementById('faceBlacklistName').value.trim();
            const fileInput = document.getElementById('faceBlacklistImage');
            if (!name || !fileInput.files.length) {
                showNotification('Please provide name and image', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('name', name);
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('image', fileInput.files[i]);
            }

            showLoading(true);
            fetch('/api/face-blacklist/add', {
                method: 'POST',
                body: formData
            }).then(r => r.json()).then(data => {
                showLoading(false);
                if (data.success) {
                    document.getElementById('faceBlacklistName').value = '';
                    fileInput.value = '';
                    loadFaceBlacklist();
                    showNotification('Person added to blacklist', 'success');
                } else {
                    showNotification(data.error || 'Failed to add', 'error');
                }
            });
        });
    }
});

// Add Face Blacklist initializer to DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    loadFaceBlacklist();
});

window.loadBlacklist = function () {
    const container = document.getElementById('blacklistContainer');
    if (!container) return;

    fetch('/api/blacklist/list')
        .then(r => r.json())
        .then(data => {
            if (!data.blacklist || data.blacklist.length === 0) {
                container.innerHTML = '<div style="color:var(--text-muted); padding: 8px;">No plates in blacklist.</div>';
                return;
            }
            container.innerHTML = data.blacklist.map(plate => `
            <div class="detection-item" style="justify-content: space-between;">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:#ef4444"></span>
                    <strong>${plate}</strong>
                </span>
                <button class="tab-btn" onclick="removeBlacklist('${plate}')" style="padding: 4px 8px; font-size:12px; border-color:#ef4444; color:#ef4444; min-width:unset;"><i class="fas fa-trash"></i></button>
            </div>
        `).join('');
        });
};

window.removeBlacklist = function (plate) {
    if (!confirm(`Remove ${plate} from blacklist?`)) return;
    fetch('/api/blacklist/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plate })
    }).then(() => loadBlacklist());
};

window.loadWhitelist = function () {
    const container = document.getElementById('whitelistContainer');
    if (!container) return;

    fetch('/api/whitelist/list')
        .then(r => r.json())
        .then(data => {
            if (!data.whitelist || data.whitelist.length === 0) {
                container.innerHTML = '<div style="color:var(--text-muted); padding: 8px;">No faces in whitelist.</div>';
                return;
            }
            container.innerHTML = data.whitelist.map(name => `
            <div class="detection-item" style="justify-content: space-between;">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:#06d6a0"></span>
                    <strong>${name}</strong>
                </span>
                <button class="tab-btn" onclick="removeWhitelist('${name}')" style="padding: 4px 8px; font-size:12px; border-color:#ef4444; color:#ef4444; min-width:unset;"><i class="fas fa-trash"></i></button>
            </div>
        `).join('');
        });
};

window.removeWhitelist = function (name) {
    if (!confirm(`Remove ${name} from whitelist?`)) return;
    fetch('/api/whitelist/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    }).then(() => loadWhitelist());
};

window.loadFaceBlacklist = function () {
    const container = document.getElementById('faceBlacklistContainer');
    if (!container) return;

    fetch('/api/face-blacklist/list')
        .then(r => r.json())
        .then(data => {
            if (!data.blacklist || data.blacklist.length === 0) {
                container.innerHTML = '<div style="color:var(--text-muted); padding: 8px;">No faces in blacklist.</div>';
                return;
            }
            container.innerHTML = data.blacklist.map(name => `
            <div class="detection-item" style="justify-content: space-between;">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:#ef4444"></span>
                    <strong>${name}</strong>
                </span>
                <button class="tab-btn" onclick="removeFaceBlacklist('${name}')" style="padding: 4px 8px; font-size:12px; border-color:#ef4444; color:#ef4444; min-width:unset;"><i class="fas fa-trash"></i></button>
            </div>
        `).join('');
        });
};

window.removeFaceBlacklist = function (name) {
    if (!confirm(`Remove ${name} from blacklist?`)) return;
    fetch('/api/face-blacklist/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    }).then(() => loadFaceBlacklist());
};
