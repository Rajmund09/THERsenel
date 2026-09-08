// pixel-swap.js — High-Performance Black & White Radial Wave Transition
// Expands smoothly from the middle to the edges with zero lag and zero hover effects.
(function () {
  'use strict';

  // Strict Black & White Palette
  const BW_PALETTE = [
    { color: '#000000', isWhite: false },
    { color: '#09090B', isWhite: false },
    { color: '#18181B', isWhite: false },
    { color: '#FFFFFF', isWhite: true },
    { color: '#F4F4F5', isWhite: true },
    { color: '#E4E4E7', isWhite: true }
  ];

  class Pixel {
    constructor(x, y, baseSize, colorDef, waveDelay) {
      this.x = x;
      this.y = y;
      this.baseSize = baseSize;
      this.color = colorDef.color;
      this.isWhite = colorDef.isWhite;
      this.waveDelay = waveDelay; // 0.0 (middle / center) -> 1.0 (outer edges / corners)

      this.scale = 0;
      this.opacity = 0;
    }

    // Ultra-fast GPU-friendly draw — zero shadowBlur, zero save/restore overhead
    draw(ctx) {
      if (this.scale <= 0.02 || this.opacity <= 0.02) return;

      const size = this.scale * this.baseSize;
      const offset = (this.baseSize - size) * 0.5;
      const rx = this.x + offset;
      const ry = this.y + offset;

      ctx.globalAlpha = this.opacity;
      ctx.fillStyle = this.color;
      ctx.fillRect(rx, ry, size, size);

      if (this.isWhite) {
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.16)';
        ctx.lineWidth = 1;
        ctx.strokeRect(rx, ry, size, size);
      }
    }
  }

  class PixelSwapEngine {
    constructor(canvasElement, options = {}) {
      if (!canvasElement) return;

      this.canvas = canvasElement;
      this.ctx = canvasElement.getContext('2d', { alpha: true });
      this.container = canvasElement.parentElement || canvasElement;

      this.gridGap = options.gap ?? 16;
      this.pixelSize = options.pixelSize ?? 13.5;

      this.pixels = [];
      this.animFrame = null;
      this.isTransitioning = false;

      this.width = 0;
      this.height = 0;
      this.lastWidth = 0;
      this.lastHeight = 0;
      this.resizeTimer = null;

      this.initSize();
      this.initGrid();
      this.bindResize();
    }

    initSize() {
      const rect = this.container.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.width = Math.max(10, Math.floor(rect.width));
      this.height = Math.max(10, Math.floor(rect.height));

      this.canvas.width = Math.round(this.width * dpr);
      this.canvas.height = Math.round(this.height * dpr);
      this.canvas.style.width = `${this.width}px`;
      this.canvas.style.height = `${this.height}px`;

      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.lastWidth = this.width;
      this.lastHeight = this.height;
    }

    initGrid() {
      const pxs = [];
      const step = this.gridGap;
      const cols = Math.ceil(this.width / step);
      const rows = Math.ceil(this.height / step);

      const startX = (this.width - (cols - 1) * step) * 0.5;
      const startY = (this.height - (rows - 1) * step) * 0.5;

      const centerX = this.width * 0.5;
      const centerY = this.height * 0.5;

      for (let c = 0; c < cols; c++) {
        for (let r = 0; r < rows; r++) {
          const x = startX + c * step - (this.pixelSize * 0.5);
          const y = startY + r * step - (this.pixelSize * 0.5);

          // Center of this pixel
          const px = x + this.pixelSize * 0.5;
          const py = y + this.pixelSize * 0.5;

          // Normalized distance from middle of card (0.0 at center, ~0.7 at edge midpoints, 1.0 at corners)
          const normX = (px - centerX) / Math.max(1, centerX);
          const normY = (py - centerY) / Math.max(1, centerY);
          const dist = Math.hypot(normX, normY) / Math.SQRT2;
          const waveDelay = Math.max(0, Math.min(1, dist));

          const colorDef = BW_PALETTE[Math.floor(Math.random() * BW_PALETTE.length)];
          pxs.push(new Pixel(x, y, this.pixelSize, colorDef, waveDelay));
        }
      }
      this.pixels = pxs;
    }

    bindResize() {
      this.observer = new ResizeObserver((entries) => {
        if (this.isTransitioning) return;
        for (const entry of entries) {
          const w = Math.floor(entry.contentRect.width);
          const h = Math.floor(entry.contentRect.height);
          if (Math.abs(w - this.lastWidth) > 8 || Math.abs(h - this.lastHeight) > 8) {
            clearTimeout(this.resizeTimer);
            this.resizeTimer = setTimeout(() => {
              this.initSize();
              this.initGrid();
              this.ctx.clearRect(0, 0, this.width, this.height);
            }, 60);
          }
        }
      });
      this.observer.observe(this.container);
    }

    // High-performance, 60fps radial wave transition from middle to edges
    swap(onMidpoint, onComplete) {
      if (this.isTransitioning) return;
      this.isTransitioning = true;

      // Ensure canvas matches latest container dimensions before starting wave
      this.initSize();
      this.initGrid();

      const startTime = performance.now();
      const totalDuration = 800; // ms: silky smooth, responsive and punchy
      let midpointFired = false;

      const animateSwap = (now) => {
        const elapsed = now - startTime;
        const progress = Math.min(1, elapsed / totalDuration);

        // Fire midpoint reveal when the expanding wave blankets the card
        if (progress >= 0.46 && !midpointFired) {
          midpointFired = true;
          if (onMidpoint) {
            try { onMidpoint(); } catch (err) { console.error("PixelSwap midpoint error:", err); }
          }
        }

        // Fast update: wave starts at center (delay = 0.0) and radiates out to edges (delay = 1.0)
        const waveSpeed = 0.44;     // wave takes 44% of duration to reach the edges
        const pixelLife = 0.52;     // each pixel's pulse duration

        for (let i = 0; i < this.pixels.length; i++) {
          const p = this.pixels[i];
          const start = p.waveDelay * waveSpeed;
          const end = start + pixelLife;

          if (progress < start || progress > end) {
            p.scale = 0;
            p.opacity = 0;
          } else {
            const localProgress = (progress - start) / pixelLife;
            // Hann window (raised cosine): ultra-smooth, zero pop at entry and exit
            const wave = 0.5 * (1 - Math.cos(2 * Math.PI * localProgress));
            p.scale = wave * 1.25;
            p.opacity = Math.min(1, wave * 1.30);
          }
        }

        // Fast GPU render: clear and draw active pixels
        this.ctx.clearRect(0, 0, this.width, this.height);
        for (let i = 0; i < this.pixels.length; i++) {
          this.pixels[i].draw(this.ctx);
        }
        this.ctx.globalAlpha = 1.0;

        if (progress < 1) {
          this.animFrame = requestAnimationFrame(animateSwap);
        } else {
          // Finished: clear canvas cleanly
          this.ctx.clearRect(0, 0, this.width, this.height);
          this.isTransitioning = false;
          if (onComplete) {
            try { onComplete(); } catch (err) { console.error("PixelSwap complete error:", err); }
          }
        }
      };

      this.animFrame = requestAnimationFrame(animateSwap);
    }

    destroy() {
      if (this.animFrame) {
        cancelAnimationFrame(this.animFrame);
      }
      if (this.observer) {
        this.observer.disconnect();
      }
      clearTimeout(this.resizeTimer);
      this.ctx.clearRect(0, 0, this.width, this.height);
    }
  }

  // Export globally
  window.PixelSwap = PixelSwapEngine;
})();
