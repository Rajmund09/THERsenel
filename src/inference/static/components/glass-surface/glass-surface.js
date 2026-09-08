// glass-surface.js — Vanilla JavaScript implementation of React Bits <GlassSurface />
// Features SVG displacement chromatic aberration, liquid refraction pulse, and 3D specular tilt animation.
(function () {
  'use strict';

  let nextId = 0;

  function supportsSVGFilters(filterId) {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return false;
    }
    const isWebkit = /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent);
    const isFirefox = /Firefox/.test(navigator.userAgent);
    if (isWebkit || isFirefox) {
      return false;
    }
    const div = document.createElement('div');
    div.style.backdropFilter = `url(#${filterId})`;
    return div.style.backdropFilter !== '';
  }

  class GlassSurface {
    constructor(element, options = {}) {
      if (!element) {
        console.warn('GlassSurface: Target element not found.');
        return;
      }

      this.container = element;
      this.id = ++nextId;
      this.uniqueId = `gs-${this.id}-${Math.random().toString(36).substr(2, 6)}`;
      this.filterId = `glass-filter-${this.uniqueId}`;
      this.redGradId = `red-grad-${this.uniqueId}`;
      this.blueGradId = `blue-grad-${this.uniqueId}`;

      // Options
      this.borderRadius = options.borderRadius ?? (parseInt(element.dataset.borderRadius, 10) || 32);
      this.borderWidth = options.borderWidth ?? (parseFloat(element.dataset.borderWidth) || 0.07);
      this.brightness = options.brightness ?? (parseInt(element.dataset.brightness, 10) || 50);
      this.opacity = options.opacity ?? (parseFloat(element.dataset.opacity) || 0.93);
      this.blur = options.blur ?? (parseInt(element.dataset.blur, 10) || 11);
      this.displace = options.displace ?? (parseFloat(element.dataset.displace) || 0);
      this.backgroundOpacity = options.backgroundOpacity ?? (parseFloat(element.dataset.backgroundOpacity) || 0.35);
      this.saturation = options.saturation ?? (parseFloat(element.dataset.saturation) || 1.6);
      this.baseDistortionScale = options.distortionScale ?? (parseFloat(element.dataset.distortionScale) || -160);
      this.redOffset = options.redOffset ?? (parseFloat(element.dataset.redOffset) || 0);
      this.greenOffset = options.greenOffset ?? (parseFloat(element.dataset.greenOffset) || 12);
      this.blueOffset = options.blueOffset ?? (parseFloat(element.dataset.blueOffset) || 24);
      this.xChannel = options.xChannel || element.dataset.xChannel || 'R';
      this.yChannel = options.yChannel || element.dataset.yChannel || 'G';
      this.mixBlendMode = options.mixBlendMode || element.dataset.mixBlendMode || 'difference';

      // 3D Tilt State
      this.tiltX = 0;
      this.tiltY = 0;
      this.targetTiltX = 0;
      this.targetTiltY = 0;
      this.isHovered = false;
      this.animFrame = null;

      this.buildDOM();
      this.updateDisplacementMap();
      this.bindEvents();
      this.startAnimation();
    }

    buildDOM() {
      // Create SVG Filter definition
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('class', 'glass-surface__filter');
      svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');

      svg.innerHTML = `
        <defs>
          <filter id="${this.filterId}" color-interpolation-filters="sRGB" x="0%" y="0%" width="100%" height="100%">
            <feImage id="fe-image-${this.uniqueId}" x="0" y="0" width="100%" height="100%" preserveAspectRatio="none" result="map" />
            <feDisplacementMap id="fe-red-${this.uniqueId}" in="SourceGraphic" in2="map" scale="${this.baseDistortionScale + this.redOffset}" xChannelSelector="${this.xChannel}" yChannelSelector="${this.yChannel}" result="dispRed" />
            <feColorMatrix in="dispRed" type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0" result="red" />
            
            <feDisplacementMap id="fe-green-${this.uniqueId}" in="SourceGraphic" in2="map" scale="${this.baseDistortionScale + this.greenOffset}" xChannelSelector="${this.xChannel}" yChannelSelector="${this.yChannel}" result="dispGreen" />
            <feColorMatrix in="dispGreen" type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0" result="green" />
            
            <feDisplacementMap id="fe-blue-${this.uniqueId}" in="SourceGraphic" in2="map" scale="${this.baseDistortionScale + this.blueOffset}" xChannelSelector="${this.xChannel}" yChannelSelector="${this.yChannel}" result="dispBlue" />
            <feColorMatrix in="dispBlue" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0" result="blue" />
            
            <feBlend in="red" in2="green" mode="screen" result="rg" />
            <feBlend in="rg" in2="blue" mode="screen" result="output" />
            <feGaussianBlur in="output" stdDeviation="${this.displace || 0.7}" />
          </filter>
        </defs>
      `;

      this.container.insertBefore(svg, this.container.firstChild);

      // Create interactive specular spot highlight element
      this.sheenSpot = document.createElement('div');
      this.sheenSpot.className = 'glass-surface-sheen-spot';
      this.container.appendChild(this.sheenSpot);

      // Cache DOM references
      this.feImage = svg.querySelector(`#fe-image-${this.uniqueId}`);
      this.feRed = svg.querySelector(`#fe-red-${this.uniqueId}`);
      this.feGreen = svg.querySelector(`#fe-green-${this.uniqueId}`);
      this.feBlue = svg.querySelector(`#fe-blue-${this.uniqueId}`);

      // Apply CSS variables
      this.container.style.borderRadius = `${this.borderRadius}px`;
      this.container.style.setProperty('--glass-frost', this.backgroundOpacity);
      this.container.style.setProperty('--glass-saturation', this.saturation);
      this.container.style.setProperty('--filter-id', `url(#${this.filterId})`);

      const svgSupported = supportsSVGFilters(this.filterId);
      if (svgSupported) {
        this.container.classList.add('glass-surface--svg');
      } else {
        this.container.classList.add('glass-surface--fallback');
      }
    }

    generateDisplacementMap() {
      const rect = this.container.getBoundingClientRect();
      const actualWidth = Math.max(10, Math.floor(rect.width || 400));
      const actualHeight = Math.max(10, Math.floor(rect.height || 160));
      const edgeSize = Math.min(actualWidth, actualHeight) * (this.borderWidth * 0.5);

      const svgContent = `
        <svg viewBox="0 0 ${actualWidth} ${actualHeight}" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="${this.redGradId}" x1="100%" y1="0%" x2="0%" y2="0%">
              <stop offset="0%" stop-color="#0000"/>
              <stop offset="100%" stop-color="red"/>
            </linearGradient>
            <linearGradient id="${this.blueGradId}" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#0000"/>
              <stop offset="100%" stop-color="blue"/>
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="${actualWidth}" height="${actualHeight}" fill="black"></rect>
          <rect x="0" y="0" width="${actualWidth}" height="${actualHeight}" rx="${this.borderRadius}" fill="url(#${this.redGradId})" />
          <rect x="0" y="0" width="${actualWidth}" height="${actualHeight}" rx="${this.borderRadius}" fill="url(#${this.blueGradId})" style="mix-blend-mode: ${this.mixBlendMode}" />
          <rect x="${edgeSize}" y="${edgeSize}" width="${actualWidth - edgeSize * 2}" height="${actualHeight - edgeSize * 2}" rx="${this.borderRadius}" fill="hsl(0 0% ${this.brightness}% / ${this.opacity})" style="filter:blur(${this.blur}px)" />
        </svg>
      `;

      return `data:image/svg+xml,${encodeURIComponent(svgContent)}`;
    }

    updateDisplacementMap() {
      if (this.feImage) {
        this.feImage.setAttribute('href', this.generateDisplacementMap());
      }
    }

    bindEvents() {
      // ResizeObserver
      this.resizeObserver = new ResizeObserver(() => {
        requestAnimationFrame(() => this.updateDisplacementMap());
      });
      this.resizeObserver.observe(this.container);

      // Mouse Parallax & 3D Tilt
      this.onMouseMove = (e) => {
        this.isHovered = true;
        const rect = this.container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        // Max tilt +/- 7 degrees
        const percentX = (x - centerX) / centerX;
        const percentY = (y - centerY) / centerY;

        this.targetTiltY = percentX * 7;
        this.targetTiltX = -percentY * 7;

        // Update sheen highlight position
        const sheenX = Math.round((x / rect.width) * 100);
        const sheenY = Math.round((y / rect.height) * 100);
        this.container.style.setProperty('--sheen-x', `${sheenX}%`);
        this.container.style.setProperty('--sheen-y', `${sheenY}%`);
      };

      this.onMouseEnter = () => {
        this.isHovered = true;
      };

      this.onMouseLeave = () => {
        this.isHovered = false;
        this.targetTiltX = 0;
        this.targetTiltY = 0;
      };

      this.container.addEventListener('mousemove', this.onMouseMove);
      this.container.addEventListener('mouseenter', this.onMouseEnter);
      this.container.addEventListener('mouseleave', this.onMouseLeave);
    }

    startAnimation() {
      const animate = (time) => {
        // Liquid Refraction Breathing (oscillate distortion scale gently)
        const pulse = Math.sin(time * 0.0015);
        const dynamicScale = this.baseDistortionScale + pulse * 14;

        if (this.feRed) {
          this.feRed.setAttribute('scale', (dynamicScale + this.redOffset).toFixed(1));
        }
        if (this.feGreen) {
          this.feGreen.setAttribute('scale', (dynamicScale + this.greenOffset).toFixed(1));
        }
        if (this.feBlue) {
          this.feBlue.setAttribute('scale', (dynamicScale + this.blueOffset).toFixed(1));
        }

        // 3D Tilt Interpolation
        this.tiltX += (this.targetTiltX - this.tiltX) * 0.1;
        this.tiltY += (this.targetTiltY - this.tiltY) * 0.1;

        if (Math.abs(this.tiltX) > 0.01 || Math.abs(this.tiltY) > 0.01 || this.isHovered) {
          this.container.style.transform = `perspective(1000px) rotateX(${this.tiltX.toFixed(2)}deg) rotateY(${this.tiltY.toFixed(2)}deg)`;
        } else {
          this.container.style.transform = 'none';
        }

        this.animFrame = requestAnimationFrame(animate);
      };

      this.animFrame = requestAnimationFrame(animate);
    }

    destroy() {
      if (this.animFrame) cancelAnimationFrame(this.animFrame);
      if (this.resizeObserver) this.resizeObserver.disconnect();
      this.container.removeEventListener('mousemove', this.onMouseMove);
      this.container.removeEventListener('mouseenter', this.onMouseEnter);
      this.container.removeEventListener('mouseleave', this.onMouseLeave);
    }
  }

  // Auto-initialize on elements with [data-glass-surface]
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-glass-surface]').forEach((el) => {
      new GlassSurface(el);
    });
  });

  window.GlassSurface = GlassSurface;
})();
