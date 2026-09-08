// pixel-card.js — Vanilla JavaScript implementation of React Bits <PixelCard />
(function () {
  'use strict';

  class Pixel {
    constructor(canvas, context, x, y, color, speed, delay, maxSizeInteger = 2.5) {
      this.width = canvas.width;
      this.height = canvas.height;
      this.ctx = context;
      this.x = x;
      this.y = y;
      this.color = color;
      this.speed = this.getRandomValue(0.1, 0.9) * speed;
      this.size = 0;
      this.sizeStep = Math.random() * 0.4 + 0.1;
      this.minSize = 0.5;
      this.maxSizeInteger = maxSizeInteger;
      this.maxSize = this.getRandomValue(this.minSize, this.maxSizeInteger);
      this.delay = delay;
      this.counter = 0;
      this.counterStep = Math.random() * 4 + (this.width + this.height) * 0.01;
      this.isIdle = false;
      this.isReverse = false;
      this.isShimmer = false;
    }

    getRandomValue(min, max) {
      return Math.random() * (max - min) + min;
    }

    draw() {
      const centerOffset = this.maxSizeInteger * 0.5 - this.size * 0.5;
      this.ctx.fillStyle = this.color;
      this.ctx.fillRect(this.x + centerOffset, this.y + centerOffset, this.size, this.size);
    }

    appear() {
      this.isIdle = false;
      if (this.counter <= this.delay) {
        this.counter += this.counterStep;
        return;
      }
      if (this.size >= this.maxSize) {
        this.isShimmer = true;
      }
      if (this.isShimmer) {
        this.shimmer();
      } else {
        this.size += this.sizeStep;
      }
      this.draw();
    }

    disappear() {
      this.isShimmer = false;
      this.counter = 0;
      if (this.size <= 0) {
        this.isIdle = true;
        return;
      } else {
        this.size -= 0.1;
      }
      this.draw();
    }

    shimmer() {
      if (this.size >= this.maxSize) {
        this.isReverse = true;
      } else if (this.size <= this.minSize) {
        this.isReverse = false;
      }
      if (this.isReverse) {
        this.size -= this.speed;
      } else {
        this.size += this.speed;
      }
    }
  }

  function getEffectiveSpeed(value, reducedMotion) {
    const min = 0;
    const max = 100;
    const throttle = 0.001;
    const parsed = parseInt(value, 10);

    if (parsed <= min || reducedMotion) {
      return min;
    } else if (parsed >= max) {
      return max * throttle;
    } else {
      return parsed * throttle;
    }
  }

  const VARIANTS = {
    default: {
      activeColor: null,
      gap: 5,
      speed: 35,
      colors: '#f8fafc,#f1f5f9,#cbd5e1',
      noFocus: false
    },
    blue: {
      activeColor: '#e0f2fe',
      gap: 10,
      speed: 25,
      colors: '#e0f2fe,#7dd3fc,#0ea5e9',
      noFocus: false
    },
    yellow: {
      activeColor: '#fef08a',
      gap: 3,
      speed: 20,
      colors: '#fef08a,#fde047,#eab308',
      noFocus: false
    },
    pink: {
      activeColor: '#fecdd3',
      gap: 6,
      speed: 80,
      colors: '#fecdd3,#fda4af,#e11d48',
      noFocus: true
    },
    tactical: {
      activeColor: 'rgba(234, 88, 12, 0.35)',
      gap: 6,
      speed: 40,
      colors: '#f97316,#22c55e,#ea580c,#16a34a,#fb923c,#4ade80,#18181b,#000000,#27272a,#ff8800',
      noFocus: false
    }
  };

  class PixelCard {
    constructor(element, options = {}) {
      if (!element) {
        console.warn('PixelCard: Target element not found.');
        return;
      }

      this.container = element;
      this.options = options;

      const variantName = options.variant || 'tactical';
      const variantCfg = VARIANTS[variantName] || VARIANTS.default;

      this.finalGap = options.gap ?? variantCfg.gap;
      this.finalSpeed = options.speed ?? variantCfg.speed;
      this.finalColors = options.colors ?? variantCfg.colors;
      this.finalNoFocus = options.noFocus ?? variantCfg.noFocus;
      this.activeColor = options.activeColor ?? variantCfg.activeColor;
      this.pixelSize = options.pixelSize ?? 2.5;

      this.pixels = [];
      this.animationFrame = null;
      this.timePrevious = performance.now();
      this.reducedMotion = typeof window !== 'undefined' && 
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

      this.initDOM();
      this.bindEvents();
      this.initPixels();
    }

    initDOM() {
      this.container.classList.add('pixel-card');
      
      if (this.activeColor) {
        this.container.style.setProperty('--pixel-card-active-color', this.activeColor);
      }

      // Check for existing canvas or create one
      let canvas = this.container.querySelector(':scope > .pixel-canvas');
      if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.className = 'pixel-canvas';
        // Insert as first child so it sits beneath card content
        this.container.insertBefore(canvas, this.container.firstChild);
      }
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');

      if (!this.finalNoFocus && !this.container.hasAttribute('tabindex')) {
        this.container.setAttribute('tabindex', '0');
      }
    }

    initPixels() {
      if (!this.container || !this.canvas) return;

      const rect = this.container.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));

      this.canvas.width = width;
      this.canvas.height = height;
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;

      const colorsArray = this.finalColors.split(',').map(c => c.trim());
      const pxs = [];
      const gap = parseInt(this.finalGap, 10) || 6;
      const effSpeed = getEffectiveSpeed(this.finalSpeed, this.reducedMotion);

      for (let x = 0; x < width; x += gap) {
        for (let y = 0; y < height; y += gap) {
          const color = colorsArray[Math.floor(Math.random() * colorsArray.length)];
          const dx = x - width / 2;
          const dy = y - height / 2;
          const distance = Math.sqrt(dx * dx + dy * dy);
          const delay = this.reducedMotion ? 0 : distance;

          pxs.push(new Pixel(this.canvas, this.ctx, x, y, color, effSpeed, delay, this.pixelSize));
        }
      }
      this.pixels = pxs;
    }

    doAnimate(fnName) {
      this.animationFrame = requestAnimationFrame(() => this.doAnimate(fnName));
      const timeNow = performance.now();
      const timePassed = timeNow - this.timePrevious;
      const timeInterval = 1000 / 60;

      if (timePassed < timeInterval) return;
      this.timePrevious = timeNow - (timePassed % timeInterval);

      if (!this.ctx || !this.canvas) return;

      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

      let allIdle = true;
      for (let i = 0; i < this.pixels.length; i++) {
        const pixel = this.pixels[i];
        pixel[fnName]();
        if (!pixel.isIdle) {
          allIdle = false;
        }
      }

      if (allIdle) {
        cancelAnimationFrame(this.animationFrame);
        this.animationFrame = null;
      }
    }

    handleAnimation(name) {
      if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame);
      }
      this.animationFrame = requestAnimationFrame(() => this.doAnimate(name));
    }

    appear() {
      this.handleAnimation('appear');
    }

    disappear() {
      this.handleAnimation('disappear');
    }

    bindEvents() {
      this.onMouseEnter = () => this.handleAnimation('appear');
      this.onMouseLeave = () => this.handleAnimation('disappear');

      this.onFocus = (e) => {
        if (e.currentTarget.contains(e.relatedTarget)) return;
        this.handleAnimation('appear');
      };

      this.onBlur = (e) => {
        if (e.currentTarget.contains(e.relatedTarget)) return;
        this.handleAnimation('disappear');
      };

      this.container.addEventListener('mouseenter', this.onMouseEnter);
      this.container.addEventListener('mouseleave', this.onMouseLeave);

      if (!this.finalNoFocus) {
        this.container.addEventListener('focus', this.onFocus);
        this.container.addEventListener('blur', this.onBlur);
      }

      this.observer = new ResizeObserver(() => {
        this.initPixels();
      });
      this.observer.observe(this.container);
    }

    destroy() {
      if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame);
      }
      if (this.observer) {
        this.observer.disconnect();
      }
      this.container.removeEventListener('mouseenter', this.onMouseEnter);
      this.container.removeEventListener('mouseleave', this.onMouseLeave);
      this.container.removeEventListener('focus', this.onFocus);
      this.container.removeEventListener('blur', this.onBlur);
      if (this.canvas && this.canvas.parentNode) {
        this.canvas.parentNode.removeChild(this.canvas);
      }
    }
  }

  // Export globally
  window.PixelCard = PixelCard;
})();
