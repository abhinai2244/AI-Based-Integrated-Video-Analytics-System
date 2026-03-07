/* ═══════════════════════════════════════════════════════════
   AI Video Analytics Dashboard — Main JavaScript
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initAnimatedCounters();
    initScrollAnimations();
    initMobileMenu();
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
            processUpload(e.dataTransfer.files[0], endpoint, renderCallback);
        }
    });

    // Click upload
    if (input) {
        input.addEventListener('change', (e) => {
            if (e.target.files.length) {
                processUpload(e.target.files[0], endpoint, renderCallback);
            }
        });
    }
}

function processUpload(file, endpoint, renderCallback) {
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
    const loadingText = isVideo ? 'Analyzing video frames with AI models (this may take a moment)...' : 'Analyzing image with AI models...';

    const loadingEl = document.querySelector('#loadingOverlay .loading-text');
    if (loadingEl) loadingEl.textContent = loadingText;

    showLoading(true);

    const formData = new FormData();
    // We use 'file' as a generic name that the updated backend supports
    formData.append('file', file);

    fetch(endpoint, {
        method: 'POST',
        body: formData
    })
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

/* ── Render Functions for Module Results ──────────────────── */

// Vehicle Detection Results
function renderVehicleResults(data) {
    const panel = document.getElementById('vehicleResults');
    if (!panel) return;
    panel.classList.add('visible');

    // Annotated image
    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    // Total count
    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = data.total;

    // Per-class counts
    const listEl = panel.querySelector('.detections-list');
    if (listEl && data.counts) {
        const colors = { car: '#06d6a0', motorcycle: '#f59e0b', bus: '#3b82f6', truck: '#8b5cf6' };
        listEl.innerHTML = Object.entries(data.counts).map(([type, count]) => `
            <div class="detection-item">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:${colors[type] || '#06d6a0'}"></span>
                    ${type.charAt(0).toUpperCase() + type.slice(1)}
                </span>
                <span class="detection-conf">${count}</span>
            </div>
        `).join('');
    }

    // Update chart if exists
    if (window.vehicleChart && data.counts) {
        window.vehicleChart.data.datasets[0].data = Object.values(data.counts);
        window.vehicleChart.update();
    }
}

// ANPR Results
function renderANPRResults(data) {
    const panel = document.getElementById('anprResults');
    if (!panel) return;
    panel.classList.add('visible');

    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = data.total_plates;

    const listEl = panel.querySelector('.detections-list');
    if (listEl && data.plates) {
        listEl.innerHTML = data.plates.map(plate => `
            <div class="detection-item">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:#3b82f6"></span>
                    <strong>${plate.text}</strong>
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

    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = data.total_faces;

    const listEl = panel.querySelector('.detections-list');
    if (listEl && data.faces) {
        listEl.innerHTML = data.faces.map(face => `
            <div class="detection-item" style="flex-direction:column;align-items:flex-start;gap:6px">
                <span class="detection-item-label">
                    <span class="detection-dot" style="background:#8b5cf6"></span>
                    <strong>Face #${face.id}</strong>
                </span>
                <div style="display:flex;gap:8px;flex-wrap:wrap;padding-left:16px">
                    <span class="meta-tag"><i class="fas fa-birthday-cake"></i> Age: ${face.age}</span>
                    <span class="meta-tag"><i class="fas fa-venus-mars"></i> ${face.gender}</span>
                    <span class="meta-tag"><i class="fas fa-smile"></i> ${face.emotion}</span>
                    <span class="meta-tag"><i class="fas fa-globe"></i> ${face.race}</span>
                </div>
            </div>
        `).join('') || '<div class="detection-item"><span style="color:var(--text-muted)">No faces detected</span></div>';
    }
}

// People Counter Results
function renderPeopleResults(data) {
    const panel = document.getElementById('peopleResults');
    if (!panel) return;
    panel.classList.add('visible');

    const imgEl = panel.querySelector('.result-image');
    if (imgEl) imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image;

    const heatmapEl = panel.querySelector('.heatmap-image');
    if (heatmapEl) heatmapEl.src = 'data:image/jpeg;base64,' + data.heatmap_image;

    const totalEl = panel.querySelector('.total-count');
    if (totalEl) totalEl.textContent = data.total_people;

    const densityEl = panel.querySelector('.density-value');
    if (densityEl) densityEl.textContent = data.density;
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
