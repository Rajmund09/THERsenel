// infinite-spiral.js — Premium Apple-level 3D Spiral Gallery
(function() {

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const modulo = (v, d) => ((v % d) + d) % d;
const lerp = (a, b, t) => a + (b - a) * t;

// Apple-style spring interpolation
const springBlend = (delta, stiffness) => 1 - Math.exp(-delta * stiffness);

class InfiniteSpiral {
  constructor(container, options = {}) {
    this.container = container;
    const rawItems = options.items || [];
    this.items = rawItems.map((item, i) =>
      typeof item === 'string'
        ? { src: item, alt: `Image ${i + 1}` }
        : { alt: `Image ${i + 1}`, ...item }
    );

    this.options = {
      speed: 0.5,
      direction: 'up',
      animationMode: 'scroll',
      radius: 260,
      cardWidth: 260,
      cardHeight: 180,
      verticalSpacing: 88,
      perspective: 1100,
      cardsPerTurn: 9,
      rotation: 0,
      cardRadius: 18,
      centerScale: 1.22,
      edgeFade: 0.28,
      edgeBlur: 10,
      pauseOnHover: false,
      imageFit: 'cover',
      grayscale: 0,
      scrollVelocityMultiplier: 0.012,
      ...options
    };

    // Physics state
    this.progress = 0;
    this.targetProgress = 0;
    this.velocity = 0;       // smooth velocity accumulator
    this.autoSpeed = 0;
    this.hovered = false;
    this.isVisible = true;
    this.dragging = false;
    this.lastPointerY = 0;
    this.dragMoved = false;

    this.cardNodes = [];
    this.glossNodes = [];
    this.bounds = { width: 1, height: 1 };
    this.frameId = null;
    this.previousTime = performance.now();

    this._initDOM();
    this._bindEvents();
    this.render = this.render.bind(this);
    this.frameId = requestAnimationFrame(this.render);
  }

  _initDOM() {
    this.container.classList.add('infinite-spiral');
    this.container.style.perspective = `${this.options.perspective}px`;
    this.container.style.perspectiveOrigin = '50% 50%';

    const drag = this.options.animationMode === 'drag' || this.options.animationMode === 'all';
    this.container.style.cursor = drag ? 'grab' : 'default';
    this.container.style.touchAction = drag ? 'none' : 'auto';
    this.container.style.userSelect = 'none';

    this.innerContainer = document.createElement('div');
    this.innerContainer.className = 'infinite-spiral__inner';

    this.items.forEach((item, index) => {
      const card = document.createElement(item.href ? 'a' : 'div');
      if (item.href) {
        card.href = item.href;
        if (item.target) { card.target = item.target; card.rel = 'noreferrer'; }
      }
      card.className = 'infinite-spiral__card';

      // Card base styles
      card.style.width = `${this.options.cardWidth}px`;
      card.style.height = `${this.options.cardHeight}px`;
      card.style.borderRadius = `${this.options.cardRadius}px`;
      card.style.border = '1px solid rgba(255,255,255,0.12)';
      card.style.boxShadow = '0 24px 60px rgba(0,0,0,0.55), 0 4px 16px rgba(0,0,0,0.4)';
      card.style.background = '#111';
      card.style.overflow = 'hidden';

      // Image
      const img = document.createElement('img');
      img.src = item.src;
      img.alt = item.alt;
      img.loading = index < 8 ? 'eager' : 'lazy';
      img.draggable = false;
      img.className = 'infinite-spiral__image';
      img.style.objectFit = this.options.imageFit;
      img.style.filter = this.options.grayscale > 0 ? `grayscale(${this.options.grayscale})` : 'none';

      // Glass gloss overlay
      const gloss = document.createElement('div');
      gloss.className = 'infinite-spiral__gloss';

      card.appendChild(img);
      card.appendChild(gloss);
      this.innerContainer.appendChild(card);
      this.cardNodes.push(card);
      this.glossNodes.push(gloss);
    });

    this.container.appendChild(this.innerContainer);
    this._measure();
  }

  _measure() {
    const r = this.container.getBoundingClientRect();
    this.bounds = { width: Math.max(r.width, 1), height: Math.max(r.height, 1) };
  }

  _bindEvents() {
    this.resizeObserver = new ResizeObserver(() => this._measure());
    this.resizeObserver.observe(this.container);

    this.intersectionObserver = new IntersectionObserver(
      ([e]) => { this.isVisible = e.isIntersecting; },
      { threshold: 0.01 }
    );
    this.intersectionObserver.observe(this.container);

    // ── Wheel: intercept and drive velocity, prevent page scroll ──
    this.handleWheel = (event) => {
      if (!this.isVisible) return;
      const scrollEnabled = this.options.animationMode === 'scroll' || this.options.animationMode === 'all';
      if (!scrollEnabled) return;
      event.preventDefault(); // stop page from scrolling
      // Apple trackpad delta is already smooth; mouse wheel deltaMode=0 is pixels
      const norm = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY; // line → px
      this.velocity += norm * this.options.scrollVelocityMultiplier;
    };
    // Must be { passive: false } to allow preventDefault
    this.container.addEventListener('wheel', this.handleWheel, { passive: false });

    // Hover
    this.container.addEventListener('mouseenter', () => { this.hovered = true; });
    this.container.addEventListener('mouseleave', () => { this.hovered = false; });

    // Drag
    const drag = this.options.animationMode === 'drag' || this.options.animationMode === 'all';
    this.stopDragging = (e) => {
      if (!this.dragging) return;
      this.dragging = false;
      if (e.currentTarget && e.currentTarget.hasPointerCapture?.(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      this.container.style.cursor = drag ? 'grab' : 'default';
    };
    this.container.addEventListener('pointerdown', (e) => {
      if (!drag || e.button !== 0) return;
      this.dragging = true;
      this.dragMoved = false;
      this.lastPointerY = e.clientY;
      this.targetProgress = this.progress;
      e.currentTarget.setPointerCapture(e.pointerId);
      this.container.style.cursor = 'grabbing';
    });
    this.container.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const dy = e.clientY - this.lastPointerY;
      this.lastPointerY = e.clientY;
      if (Math.abs(dy) > 0.5) this.dragMoved = true;
      this.targetProgress -= dy / Math.max(this.options.verticalSpacing, 1);
    });
    this.container.addEventListener('pointerup', this.stopDragging);
    this.container.addEventListener('pointercancel', this.stopDragging);
    this.container.addEventListener('click', (e) => {
      if (!this.dragMoved) return;
      e.preventDefault(); e.stopPropagation();
      this.dragMoved = false;
    }, true);
  }

  render(time) {
    const delta = Math.min((time - this.previousTime) / 1000, 0.05);
    this.previousTime = time;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const autoEnabled = this.options.animationMode === 'auto' || this.options.animationMode === 'all';
    const motionPaused = this.dragging || (this.options.pauseOnHover && this.hovered);
    const dirMul = this.options.direction === 'down' ? -1 : 1;

    // ── Auto speed ──
    const desiredAuto = (autoEnabled && this.isVisible && !reducedMotion && !motionPaused)
      ? this.options.speed * dirMul : 0;
    this.autoSpeed = lerp(this.autoSpeed, desiredAuto, springBlend(delta, 5));

    // ── Velocity momentum: Apple-style — high friction glide ──
    this.velocity *= Math.pow(0.88, delta * 60); // frame-rate independent friction
    this.targetProgress += (this.autoSpeed + this.velocity) * delta;

    // ── Progress follows target with spring ──
    // stiffness 8 = silky smooth, 16 = snappy
    this.progress = lerp(this.progress, this.targetProgress, springBlend(delta, this.dragging ? 18 : 8));

    // ── Layout constants ──
    const count = this.items.length;
    const half = count / 2;
    const { width, height } = this.bounds;
    const fit = Math.min(1,
      width  / (this.options.cardWidth  * 2.6),
      height / (this.options.cardHeight * 2.2)
    );
    const R = Math.min(this.options.radius, Math.max(80, width * 0.34)) * fit;
    const turnSize = Math.max(this.options.cardsPerTurn, 1);
    const vSpacing = this.options.verticalSpacing * fit;
    const fadeStart = clamp(1 - this.options.edgeFade, 0, 0.97);

    this.cardNodes.forEach((card, index) => {
      const gloss = this.glossNodes[index];

      let offset = index - this.progress;
      offset = modulo(offset + half, count) - half;

      // Angular position on the spiral circle
      const angle = offset * (360 / turnSize) + this.options.rotation;
      const rad   = (angle * Math.PI) / 180;

      // 3D position
      const xPos = Math.sin(rad) * R;
      const zPos = Math.cos(rad) * R;
      const yPos = offset * vSpacing;

      // ── Depth scale (perspective foreshortening) ──
      const depthScale = clamp(
        this.options.perspective / Math.max(this.options.perspective - zPos, 1),
        0.5, 1.6
      );

      // ── Focus / edge weights ──
      const edgeWeight = Math.min(Math.abs(offset) / Math.max(half, 1), 1);
      const focusWeight = 1 - Math.min(Math.abs(offset) / Math.max(turnSize * 0.55, 1), 1);

      // ── Scale ──
      const scale = (1 + (this.options.centerScale - 1) * focusWeight) * fit * depthScale;

      // ── Opacity: fade edges ──
      const opacityRaw = 1 - this._smoothstep(fadeStart, 1, edgeWeight);
      const opacity = clamp(opacityRaw, 0, 1);

      // ── True helix: rotateY exactly matches angular position ──
      // angle=0  → front card faces viewer flat
      // angle=90 → card is seen nearly edge-on from the side
      // This perfectly recreates the reference image helix structure
      const tiltY = -angle; // full 1:1 mapping

      // No Z roll — reference shows clean Y-axis-only tilt
      const tiltZ = 0;

      // ── Blur: only kicks in well into the edges ──
      const blur = this.options.edgeBlur * this._smoothstep(0.5, 1, edgeWeight);

      // ── Depth (z-index) ──
      const depth = (zPos / Math.max(R, 1) + 1) / 2;

      // Apply transform — no Z translate so cards don't clip into each other
      card.style.transform = [
        'translate(-50%, -50%)',
        `translate3d(${xPos}px, ${yPos}px, 0px)`,
        `rotateY(${tiltY}deg)`,
        `scale(${scale})`
      ].join(' ');

      card.style.opacity  = opacity.toFixed(3);
      card.style.zIndex   = String(Math.round(depth * 100000) + index);
      card.style.pointerEvents = opacity > 0.15 ? 'auto' : 'none';

      // Blur for cards fading to edges
      if (blur > 0.05) {
        card.style.filter = `blur(${blur.toFixed(2)}px)`;
      } else {
        card.style.filter = 'none';
      }

      // ── Glassy concave/convex gloss effect based on Z position ──
      // Front cards: bright top-edge specular (convex)
      // Back/side cards: darker, subtle inner shadow (concave)
      const glossT = (zPos / Math.max(R, 1) + 1) / 2; // 0=back, 1=front
      const specularAlpha = lerp(0.04, 0.22, glossT);
      const specularAngle = 125 + angle * 0.3;

      // Concave shadow (for back/side cards)
      const concaveAlpha = lerp(0.25, 0.0, glossT);

      gloss.style.background = [
        `linear-gradient(${specularAngle}deg,`,
        `  rgba(255,255,255,${specularAlpha.toFixed(3)}) 0%,`,
        `  rgba(255,255,255,0.0) 55%,`,
        `  rgba(0,0,0,${concaveAlpha.toFixed(3)}) 100%)`
      ].join('');

      // Border shimmer based on depth
      const borderAlpha = lerp(0.06, 0.3, glossT);
      card.style.border = `1px solid rgba(255,255,255,${borderAlpha.toFixed(3)})`;
      card.style.boxShadow = `0 ${12 + glossT * 24}px ${40 + glossT * 40}px rgba(0,0,0,${lerp(0.7, 0.3, glossT).toFixed(2)})`;
    });

    this.frameId = requestAnimationFrame(this.render);
  }

  _smoothstep(min, max, val) {
    const x = clamp((val - min) / (max - min || 1), 0, 1);
    return x * x * (3 - 2 * x);
  }

  destroy() {
    cancelAnimationFrame(this.frameId);
    this.resizeObserver.disconnect();
    this.intersectionObserver.disconnect();
    this.container.removeEventListener('wheel', this.handleWheel);
    this.container.innerHTML = '';
  }
}

window.InfiniteSpiral = InfiniteSpiral;
})();
