document.addEventListener("DOMContentLoaded", () => {
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const initBtn = document.querySelector('.btn-minimal');
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    
    // --- FULL-SITE LENIS SMOOTH SCROLLING ENGINE ---
    let lenis = null;
    if (typeof Lenis !== 'undefined') {
        lenis = new Lenis({
            lerp: 0.085,
            orientation: 'vertical',
            gestureOrientation: 'vertical',
            smoothWheel: true,
            wheelMultiplier: 1.0,
            touchMultiplier: 1.6,
            syncTouch: true,
            syncTouchLerp: 0.075,
            touchInertiaMultiplier: 30,
            autoResize: true,
            infinite: false
        });

        window.lenis = lenis;

        function raf(time) {
            lenis.raf(time);
            requestAnimationFrame(raf);
        }
        requestAnimationFrame(raf);

        // Auto-recalculate scroll bounds on window resize & layout changes
        window.addEventListener('resize', () => lenis.resize());

        // Fluid inertia scrolling for all internal anchors
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href && href !== '#') {
                    const target = document.querySelector(href);
                    if (target) {
                        e.preventDefault();
                        lenis.scrollTo(target, { offset: -24, duration: 1.4 });
                    }
                }
            });
        });
    }

    const outputImage = document.getElementById('output-image');
    const placeholder = document.getElementById('placeholder');
    const spinner = document.getElementById('spinner');
    const displayCard = document.querySelector('.data-card.display-card');
    const displayPanel = document.querySelector('.display-card .display-panel');
    const pixelCanvas = document.getElementById('pixel-swap-canvas');
    let currentImageDimensions = null;

    // Initialize PixelSwap Engine on the canvas with Black & White middle-to-edges wave
    const pixelSwap = (window.PixelSwap && pixelCanvas) ? new PixelSwap(pixelCanvas, {
        gap: 16,
        pixelSize: 13.5
    }) : null;

    // Smoothly adjust the Output Card's width and height to match the image dimensions
    function adjustCardDimensions(naturalWidth, naturalHeight) {
        if (!displayCard || !displayPanel || !naturalWidth || !naturalHeight) return;

        currentImageDimensions = { width: naturalWidth, height: naturalHeight };

        const dashboardGrid = document.querySelector('.dashboard-grid');
        let availableColumnWidth = displayCard.parentElement 
            ? displayCard.parentElement.clientWidth 
            : window.innerWidth;

        if (dashboardGrid && dashboardGrid.clientWidth > 800) {
            availableColumnWidth = dashboardGrid.clientWidth - 340 - 32;
        }

        const maxCardWidth = Math.min(Math.max(availableColumnWidth, 340), 960);
        const cardPadding = 56; // 28px left + 28px right padding

        const maxPanelWidth = maxCardWidth - cardPadding;
        const maxPanelHeight = Math.min(window.innerHeight * 0.72, 640);
        const minPanelHeight = 280;

        const aspectRatio = naturalWidth / naturalHeight;

        let targetPanelWidth = maxPanelWidth;
        let targetPanelHeight = Math.round(targetPanelWidth / aspectRatio);

        if (targetPanelHeight > maxPanelHeight) {
            targetPanelHeight = maxPanelHeight;
            targetPanelWidth = Math.round(targetPanelHeight * aspectRatio);
        }

        if (targetPanelHeight < minPanelHeight) {
            targetPanelHeight = minPanelHeight;
            const desiredW = Math.round(targetPanelHeight * aspectRatio);
            if (desiredW <= maxPanelWidth) {
                targetPanelWidth = desiredW;
            }
        }

        const targetCardWidth = Math.round(targetPanelWidth + cardPadding);

        // Apply smooth transition values
        displayCard.style.maxWidth = `${targetCardWidth}px`;
        displayPanel.style.height = `${targetPanelHeight}px`;
        displayPanel.style.minHeight = `${targetPanelHeight}px`;
        if (window.lenis) window.lenis.resize();
    }

    // Auto re-adjust on resize
    window.addEventListener('resize', () => {
        if (currentImageDimensions) {
            adjustCardDimensions(currentImageDimensions.width, currentImageDimensions.height);
        }
    });

    const alertBanner = document.getElementById('alert-banner');
    const alertText = document.getElementById('alert-text');
    const systemErrorBanner = document.getElementById('system-error-banner');
    const systemErrorTitle = document.getElementById('system-error-title');
    const systemErrorCode = document.getElementById('system-error-code');
    const systemErrorText = document.getElementById('system-error-text');
    const systemErrorClose = document.getElementById('system-error-close');

    function showError(title, message, code = 'ERR_SYS') {
        if (systemErrorBanner && systemErrorText) {
            if (systemErrorTitle) systemErrorTitle.textContent = title || 'SYSTEM EXCEPTION';
            if (systemErrorCode) systemErrorCode.textContent = code;
            systemErrorText.textContent = message || 'An unexpected operational failure occurred.';
            systemErrorBanner.classList.remove('hidden');
        }
    }

    function clearError() {
        if (systemErrorBanner) {
            systemErrorBanner.classList.add('hidden');
        }
    }

    if (systemErrorClose) {
        systemErrorClose.addEventListener('click', (e) => {
            e.stopPropagation();
            clearError();
        });
    }

    let lastUploadedFile = null;

    // --- 3D PARAMETERS CONSOLE INTERACTIVITY ---
    const rangeTrackFill = document.getElementById('range-track-fill');
    const hudLedMeter = document.getElementById('hud-led-meter');
    const confStatusText = document.getElementById('conf-status-text');
    const paramModeTag = document.getElementById('param-mode-tag');
    const btnNudgeDown = document.getElementById('btn-nudge-down');
    const btnNudgeUp = document.getElementById('btn-nudge-up');
    const presetBtns = document.querySelectorAll('.btn-preset-3d');
    const scalePoints = document.querySelectorAll('.scale-point');

    function updateParameterDisplay(rawVal) {
        const val = parseFloat(rawVal);
        const formatted = val.toFixed(2);
        
        // 1. Numeric HUD Readout
        if (confVal) confVal.textContent = formatted;

        // 2. Track Fill Bar
        if (rangeTrackFill) {
            rangeTrackFill.style.width = `${Math.min(Math.max(val * 100, 0), 100)}%`;
        }

        // 3. 10-Segment LED Meter
        if (hudLedMeter) {
            const segs = hudLedMeter.querySelectorAll('.led-seg');
            const litCount = Math.round(val * 10);
            segs.forEach((seg, idx) => {
                if (idx < litCount) {
                    seg.classList.add('is-lit');
                    if (idx >= 7) {
                        seg.classList.add('is-orange');
                    } else {
                        seg.classList.remove('is-orange');
                    }
                } else {
                    seg.classList.remove('is-lit', 'is-orange');
                }
            });
        }

        // 4. Dynamic Status Readout in HUD Screen
        if (confStatusText) {
            if (val < 0.35) {
                confStatusText.textContent = "HIGH SENSITIVITY (MAX RECALL)";
            } else if (val <= 0.65) {
                confStatusText.textContent = "OPTIMAL ACCURACY (BALANCED)";
            } else {
                confStatusText.textContent = "STRICT FILTER (ZERO FALSE ALARMS)";
            }
        }

        // 5. Active Preset Button Sync
        presetBtns.forEach(btn => {
            const pVal = parseFloat(btn.getAttribute('data-preset'));
            if (Math.abs(pVal - val) < 0.02) {
                btn.classList.add('is-active');
            } else {
                btn.classList.remove('is-active');
            }
        });

        // 6. Active Scale Point Ticks Sync
        scalePoints.forEach(sp => {
            const spVal = parseFloat(sp.getAttribute('data-val'));
            if (Math.abs(spVal - val) < 0.12) {
                sp.classList.add('active');
            } else {
                sp.classList.remove('active');
            }
        });
    }

    // Initialize display on load
    if (confSlider) {
        updateParameterDisplay(confSlider.value);

        // Update slider value display on drag
        confSlider.addEventListener('input', (e) => {
            updateParameterDisplay(e.target.value);
        });

        // Re-run prediction when slider is released
        confSlider.addEventListener('change', () => {
            if (lastUploadedFile) {
                processImage(lastUploadedFile);
            }
        });
    }

    // Nudge Down Button [-]
    if (btnNudgeDown && confSlider) {
        btnNudgeDown.addEventListener('click', () => {
            let newVal = Math.max(0, parseFloat(confSlider.value) - 0.05);
            newVal = Math.round(newVal * 100) / 100;
            confSlider.value = newVal;
            updateParameterDisplay(newVal);
            if (lastUploadedFile) processImage(lastUploadedFile);
        });
    }

    // Nudge Up Button [+]
    if (btnNudgeUp && confSlider) {
        btnNudgeUp.addEventListener('click', () => {
            let newVal = Math.min(1.0, parseFloat(confSlider.value) + 0.05);
            newVal = Math.round(newVal * 100) / 100;
            confSlider.value = newVal;
            updateParameterDisplay(newVal);
            if (lastUploadedFile) processImage(lastUploadedFile);
        });
    }

    // 3D Preset Push-Buttons
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const pVal = parseFloat(btn.getAttribute('data-preset'));
            if (confSlider) {
                confSlider.value = pVal;
                updateParameterDisplay(pVal);
                if (lastUploadedFile) processImage(lastUploadedFile);
            }
        });
    });

    // Scale Points Click
    scalePoints.forEach(sp => {
        sp.addEventListener('click', () => {
            const spVal = parseFloat(sp.getAttribute('data-val'));
            if (confSlider) {
                confSlider.value = spVal;
                updateParameterDisplay(spVal);
                if (lastUploadedFile) processImage(lastUploadedFile);
            }
        });
    });

    // Bind Upload Button
    if (initBtn) {
        initBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    uploadBox.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadBox.addEventListener(eventName, () => uploadBox.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, () => uploadBox.classList.remove('dragover'), false);
    });

    uploadBox.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file) {
            processImage(file);
        }
    });

    // Handle File Browse
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            processImage(file);
        }
    });

    async function processImage(file) {
        lastUploadedFile = file;

        // --- CLIENT-SIDE SECURITY PRE-FLIGHT CHECKS ---
        clearError();

        if (!file) {
            showError("INVALID INPUT", "No image file provided for sensor scan.", "NO_PAYLOAD");
            return;
        }

        const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15 MB hardware cap
        if (file.size > MAX_FILE_SIZE) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
            showError("PAYLOAD REJECTED", `File size (${sizeMB} MB) exceeds maximum security buffer of 15 MB.`, "ERR_FILE_TOO_LARGE");
            if (spinner) spinner.classList.add('hidden');
            if (placeholder) placeholder.classList.remove('hidden');
            return;
        }

        const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'];
        const fileExt = '.' + (file.name.split('.').pop() || '').toLowerCase();
        if (!file.type.startsWith('image/') && !ALLOWED_EXTENSIONS.includes(fileExt)) {
            showError("UNSUPPORTED FORMAT", `MIME / extension '${fileExt || file.type}' is unauthorized. Allowed: JPG, PNG, WEBP, BMP, TIFF.`, "ERR_INVALID_MEDIA");
            if (spinner) spinner.classList.add('hidden');
            if (placeholder) placeholder.classList.remove('hidden');
            return;
        }
        
        // 1. Immediately preview the image dimensions to start the smooth morphing right away
        const previewImg = new Image();
        const tempUrl = URL.createObjectURL(file);
        previewImg.onload = () => {
            adjustCardDimensions(previewImg.naturalWidth, previewImg.naturalHeight);
            URL.revokeObjectURL(tempUrl);
        };
        previewImg.src = tempUrl;

        // UI State: Loading (show spinner, hide image, hide alert)
        outputImage.classList.add('hidden');
        alertBanner.classList.add('hidden');
        if (placeholder) placeholder.classList.add('hidden');
        if (spinner) spinner.classList.remove('hidden');

        const btnFullviewTrigger = document.getElementById('btn-fullview-trigger');
        const outputExpandHint = document.getElementById('output-expand-hint');
        if (btnFullviewTrigger) btnFullviewTrigger.classList.remove('is-ready');
        if (outputExpandHint) outputExpandHint.classList.remove('is-visible');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('conf_threshold', confSlider.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errorData = null;
                try {
                    errorData = await response.json();
                } catch (_) {}

                const detailMsg = errorData?.detail || `Inference pipeline returned HTTP ${response.status} ${response.statusText}`;
                const errCode = errorData?.error_code || (
                    response.status === 413 ? "FILE_TOO_LARGE" :
                    response.status === 429 ? "RATE_LIMIT_EXCEEDED" :
                    response.status === 503 ? "SERVER_BUSY" :
                    response.status === 422 ? "VALIDATION_FAILED" :
                    `HTTP_${response.status}`
                );
                
                let errorTitle = "INFERENCE REJECTED";
                if (response.status === 413) errorTitle = "PAYLOAD BUFFER EXCEEDED";
                else if (response.status === 429) errorTitle = "RATE LIMIT EXCEEDED";
                else if (response.status === 503) errorTitle = "ENGINE CONCURRENCY SATURATION";
                else if (response.status === 400) errorTitle = "CORRUPTED OR MALFORMED IMAGE";
                else if (response.status === 422) errorTitle = "PARAMETER VALIDATION ERROR";

                showError(errorTitle, detailMsg, errCode);
                if (spinner) spinner.classList.add('hidden');
                if (placeholder) placeholder.classList.remove('hidden');
                return;
            }

            const alertCount = parseInt(response.headers.get("X-Alerts-Count") || "0");
            
            // Get Image Blob
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);

            // Brief smoothing delay so processing transition is fluid
            await new Promise(r => setTimeout(r, 600));

            // Set image source
            outputImage.src = imageUrl;
            
            // When image is loaded, trigger the Pixel Swap reveal
            outputImage.onload = () => {
                adjustCardDimensions(outputImage.naturalWidth, outputImage.naturalHeight);

                if (pixelSwap) {
                    pixelSwap.swap(
                        // Midpoint: switch elements underneath during the pixel blanket
                        () => {
                            if (spinner) spinner.classList.add('hidden');
                            if (placeholder) placeholder.classList.add('hidden');
                            outputImage.classList.remove('hidden');

                            if (btnFullviewTrigger) btnFullviewTrigger.classList.add('is-ready');
                            if (outputExpandHint) outputExpandHint.classList.add('is-visible');

                            if (alertCount > 0) {
                                alertText.textContent = `${alertCount} INTRUSION(S) DETECTED`;
                                alertBanner.classList.remove('hidden');
                            }
                        },
                        // Done:
                        () => {}
                    );
                } else {
                    if (spinner) spinner.classList.add('hidden');
                    if (placeholder) placeholder.classList.add('hidden');
                    outputImage.classList.remove('hidden');

                    if (btnFullviewTrigger) btnFullviewTrigger.classList.add('is-ready');
                    if (outputExpandHint) outputExpandHint.classList.add('is-visible');

                    if (alertCount > 0) {
                        alertText.textContent = `${alertCount} INTRUSION(S) DETECTED`;
                        alertBanner.classList.remove('hidden');
                    }
                }
            };

        } catch (error) {
            console.error("Error processing image:", error);
            if (spinner) spinner.classList.add('hidden');
            if (placeholder) placeholder.classList.remove('hidden');
            showError("COMMUNICATION ERROR", "Could not establish connection to Edge AI server. Verify backend daemon status.", "ERR_NETWORK");
        }
    }
    
    // ---------------------------------------------------------
    // ATTACHED DROPUP DOCK INTERACTIVITY (Telemetry & Spectral)
    // ---------------------------------------------------------
    const dropupDock = document.getElementById('output-dropup-dock');
    const dropupTrigger = document.getElementById('dropup-trigger-btn');
    const dropupPanel = document.getElementById('attached-dropup-panel');
    const spectralModeBtns = document.querySelectorAll('.spectral-mode-btn');

    if (dropupDock && dropupTrigger) {
        dropupTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = dropupDock.classList.toggle('is-open');
            dropupTrigger.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            if (dropupPanel) {
                dropupPanel.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
            }
        });

        // Prevent clicks inside panel from closing it
        if (dropupPanel) {
            dropupPanel.addEventListener('click', (e) => {
                e.stopPropagation();
            });
        }

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (dropupDock.classList.contains('is-open') && !dropupDock.contains(e.target)) {
                dropupDock.classList.remove('is-open');
                dropupTrigger.setAttribute('aria-expanded', 'false');
                if (dropupPanel) dropupPanel.setAttribute('aria-hidden', 'true');
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && dropupDock.classList.contains('is-open')) {
                dropupDock.classList.remove('is-open');
                dropupTrigger.setAttribute('aria-expanded', 'false');
                if (dropupPanel) dropupPanel.setAttribute('aria-hidden', 'true');
            }
        });

        // Spectral mode buttons active switcher
        spectralModeBtns.forEach((btn) => {
            btn.addEventListener('click', () => {
                spectralModeBtns.forEach(b => b.classList.remove('is-active'));
                btn.classList.add('is-active');
            });
        });
    }

    /* ==========================================================================
       TOP-LEFT CORNER REAL BACKGROUND SPECTRUM ORB (54% EXPOSED // CENTER-TO-RADIUS)
       ========================================================================== */
    const cornerOrbDisc = document.getElementById('cornerOrbDisc');
    const orbLayerBase = document.getElementById('orbLayerBase');
    const orbLayerRipple = document.getElementById('orbLayerRipple');

    const SPECTRUM_PALETTE = [
        {
            id: "01",
            name: "Black",
            bg: "#000000",
            ringColor: "rgba(255, 255, 255, 0.22)",
            shadow: "0 20px 50px rgba(0, 0, 0, 0.32), inset 0 2px 0 rgba(255, 255, 255, 0.3)"
        },
        {
            id: "02",
            name: "Violet",
            bg: "#7D39EB",
            ringColor: "rgba(255, 255, 255, 0.28)",
            shadow: "0 20px 50px rgba(125, 57, 235, 0.38), inset 0 2px 0 rgba(255, 255, 255, 0.4)"
        },
        {
            id: "03",
            name: "Lime",
            bg: "#C6FF33",
            ringColor: "rgba(0, 0, 0, 0.18)",
            shadow: "0 20px 50px rgba(198, 255, 51, 0.3), inset 0 2px 0 rgba(255, 255, 255, 0.7)"
        },
        {
            id: "04",
            name: "White",
            bg: "#FFFFFF",
            ringColor: "rgba(0, 0, 0, 0.12)",
            shadow: "0 20px 50px rgba(0, 0, 0, 0.15), inset 0 2px 0 rgba(255, 255, 255, 1.0)"
        }
    ];

    let currentSpectrumIdx = 0;
    let isSpectrumTransitioning = false;
    let spectrumAutoTimer = null;

    function applySpectrumColor(index, animate = true) {
        if (!orbLayerBase || !orbLayerRipple) return;
        const color = SPECTRUM_PALETTE[index];
        if (!color) return;

        if (!animate) {
            orbLayerBase.style.backgroundColor = color.bg;
            updateOrbRings(color);
            return;
        }

        if (isSpectrumTransitioning) return;
        isSpectrumTransitioning = true;

        // 1. Prepare center-to-radius expanding wave
        orbLayerRipple.style.backgroundColor = color.bg;
        orbLayerRipple.classList.remove('is-expanding');
        // Force reflow
        void orbLayerRipple.offsetWidth;
        orbLayerRipple.classList.add('is-expanding');

        // 2. Halfway through ripple (at 360ms), update ring stroke tone
        setTimeout(() => {
            updateOrbRings(color);
        }, 360);

        // 3. On ripple completion (at 800ms), lock base layer and reset ripple
        setTimeout(() => {
            orbLayerBase.style.backgroundColor = color.bg;
            orbLayerRipple.classList.remove('is-expanding');
            if (cornerOrbDisc) {
                cornerOrbDisc.style.boxShadow = color.shadow;
            }
            isSpectrumTransitioning = false;
        }, 800);
    }

    function updateOrbRings(color) {
        const rings = document.querySelectorAll('.orb-rings-svg circle, .orb-rings-svg line');
        rings.forEach(r => {
            r.style.stroke = color.ringColor;
        });
    }

    function cycleNextSpectrum() {
        currentSpectrumIdx = (currentSpectrumIdx + 1) % SPECTRUM_PALETTE.length;
        applySpectrumColor(currentSpectrumIdx, true);
    }

    if (cornerOrbDisc) {
        // Initialize with first color (Black)
        applySpectrumColor(0, false);

        // Interactive click jumps to next color immediately
        cornerOrbDisc.addEventListener('click', () => {
            clearInterval(spectrumAutoTimer);
            cycleNextSpectrum();
            startSpectrumInterval();
        });

        // Automated continuous transition every 4.2 seconds
        function startSpectrumInterval() {
            spectrumAutoTimer = setInterval(() => {
                cycleNextSpectrum();
            }, 4200);
        }

        startSpectrumInterval();
    }

    /* ==========================================================================
       RESPONSIVE ORB CORNER PLACEMENT (TOP-LEFT <-> BOTTOM-LEFT WITH SMOOTH TRANSITION)
       Moves orb to bottom-left corner when overlapping cards on tablet/small devices
       ========================================================================== */
    const cornerOrbSystem = document.getElementById('cornerBgOrb');

    function checkOrbOverlap() {
        if (!cornerOrbSystem) return;

        const isMobile = window.innerWidth <= 680;
        const isTabletOrSmall = window.innerWidth <= 1120;

        // Calculate dynamic bottom Y offset based on current viewport
        const bottomOffset = isMobile ? 50 : 70;
        const targetY = window.innerHeight - bottomOffset;
        cornerOrbSystem.style.setProperty('--orb-bottom-y', `${targetY}px`);

        if (!isTabletOrSmall) {
            // Full desktop with ample clearance: keep in top-left
            cornerOrbSystem.classList.remove('is-bottom-left');
            return;
        }

        // On tablet or small screens: check if top-left zone overlaps any card/header
        const orbW = isMobile ? 170 : 215;
        const orbH = isMobile ? 170 : 215;

        const cardsToCheck = document.querySelectorAll(
            '.metallic-black-header-wrapper, .data-card, .flow-diagram'
        );

        let hasOverlap = false;
        for (const card of cardsToCheck) {
            if (card.offsetParent === null) continue;
            const r = card.getBoundingClientRect();
            // Check intersection with top-left corner zone [0, 0, orbW, orbH]
            if (r.left < orbW && r.right > 0 && r.top < orbH && r.bottom > 0) {
                hasOverlap = true;
                break;
            }
        }

        // On tablet / small device, if overlapping or header reaches edges:
        if (hasOverlap || isTabletOrSmall) {
            cornerOrbSystem.classList.add('is-bottom-left');
        } else {
            cornerOrbSystem.classList.remove('is-bottom-left');
        }
    }

    // Run overlap check on resize, scroll, orientationchange
    window.addEventListener('resize', checkOrbOverlap);
    window.addEventListener('orientationchange', checkOrbOverlap);
    window.addEventListener('scroll', checkOrbOverlap, { passive: true });
    if (window.lenis) {
        window.lenis.on('scroll', checkOrbOverlap);
    }

    // Trigger initial check after DOM layout
    setTimeout(checkOrbOverlap, 50);
    window.addEventListener('load', checkOrbOverlap);

    /* ==========================================================================
       FULL VIEW LIGHTBOX TERMINAL (SMOOTH 3D FLIP & UPSCALE)
       - Scale up with smooth motion page flip / upscaling animation
       - Close via close button or touching/clicking outside image on backdrop
       - Escape key support & Lenis scroll-lock integration
       ========================================================================== */
    const imageFullviewModal = document.getElementById('imageFullviewModal');
    const fullviewBackdrop = document.getElementById('fullviewBackdrop');
    const fullviewDialog = document.getElementById('fullviewDialog');
    const fullviewCloseBtn = document.getElementById('fullviewCloseBtn');
    const fullviewImg = document.getElementById('fullviewImg');
    const fullviewMetaDimensions = document.getElementById('fullviewMetaDimensions');
    const fullviewDownloadBtn = document.getElementById('fullviewDownloadBtn');
    const btnFullviewTriggerPill = document.getElementById('btn-fullview-trigger');
    const outputExpandHintBtn = document.getElementById('output-expand-hint');

    let fullviewClosingTimeout = null;

    function openFullView() {
        if (!imageFullviewModal || !outputImage) return;
        // Verify output image has a valid source and is visible
        if (outputImage.classList.contains('hidden') || !outputImage.src) {
            return;
        }

        if (fullviewClosingTimeout) {
            clearTimeout(fullviewClosingTimeout);
            fullviewClosingTimeout = null;
        }

        // Sync full view image source & download link
        if (fullviewImg) {
            fullviewImg.src = outputImage.src;
        }
        if (fullviewDownloadBtn) {
            fullviewDownloadBtn.href = outputImage.src;
            fullviewDownloadBtn.setAttribute('download', `thermal_scan_${Date.now()}.png`);
        }

        // Update resolution metadata chip
        if (fullviewMetaDimensions) {
            const w = outputImage.naturalWidth || 640;
            const h = outputImage.naturalHeight || 640;
            fullviewMetaDimensions.textContent = `${w} × ${h} PX // 4D SPECTRAL`;
        }

        // Trigger opening state & 3D animation
        imageFullviewModal.classList.remove('is-closing');
        imageFullviewModal.classList.add('is-open');
        imageFullviewModal.setAttribute('aria-hidden', 'false');

        // Lock viewport scrolling smoothly
        document.documentElement.style.overflow = 'hidden';
        document.body.style.overflow = 'hidden';
        if (window.lenis && typeof window.lenis.stop === 'function') {
            window.lenis.stop();
        }

        // Focus close button for accessibility
        setTimeout(() => {
            if (fullviewCloseBtn) fullviewCloseBtn.focus();
        }, 100);
    }

    function closeFullView() {
        if (!imageFullviewModal || !imageFullviewModal.classList.contains('is-open')) return;

        // Trigger closing 3D flip-out animation
        imageFullviewModal.classList.add('is-closing');

        fullviewClosingTimeout = setTimeout(() => {
            imageFullviewModal.classList.remove('is-open', 'is-closing');
            imageFullviewModal.setAttribute('aria-hidden', 'true');

            // Restore scroll
            document.documentElement.style.overflow = '';
            document.body.style.overflow = '';
            if (window.lenis && typeof window.lenis.start === 'function') {
                window.lenis.start();
            }
            fullviewClosingTimeout = null;
        }, 340);
    }

    // 1. Direct click on analyzed output image opens full view
    if (outputImage) {
        outputImage.addEventListener('click', (e) => {
            if (!outputImage.classList.contains('hidden') && outputImage.src) {
                e.stopPropagation();
                openFullView();
            }
        });
    }

    // 2. Click on "FULL VIEW" header pill opens full view
    if (btnFullviewTriggerPill) {
        btnFullviewTriggerPill.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            openFullView();
        });
    }

    // 3. Click on hover hint button opens full view
    if (outputExpandHintBtn) {
        outputExpandHintBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            openFullView();
        });
    }

    // 4. Close via dedicated Close button
    if (fullviewCloseBtn) {
        fullviewCloseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            closeFullView();
        });
    }

    // 5. Close when touching or clicking outside image on the backdrop
    if (fullviewBackdrop) {
        fullviewBackdrop.addEventListener('click', (e) => {
            e.stopPropagation();
            closeFullView();
        });

        fullviewBackdrop.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            closeFullView();
        }, { passive: false });
    }

    // Modal background fallback click
    if (imageFullviewModal) {
        imageFullviewModal.addEventListener('click', (e) => {
            if (e.target === imageFullviewModal) {
                closeFullView();
            }
        });
    }

    // Prevent clicks/touches inside the dialog card from bubbling up and closing the modal
    if (fullviewDialog) {
        fullviewDialog.addEventListener('click', (e) => {
            e.stopPropagation();
        });
        fullviewDialog.addEventListener('touchend', (e) => {
            e.stopPropagation();
        }, { passive: true });
    }

    // 6. Keyboard navigation: Escape key closes modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && imageFullviewModal && imageFullviewModal.classList.contains('is-open')) {
            e.preventDefault();
            closeFullView();
        }
    });

});

