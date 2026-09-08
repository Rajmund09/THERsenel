/**
 * Cross-Dither Background Element
 * Renders an organic 1-bit halftone cross-stitch topographic sculpture
 * spanning the full right flank from top 0 to bottom 100vh.
 * Features dynamic fluid motion, breathing contour undulation, 
 * Lenis scroll parallax, and touch/mouse interactive energy waves.
 */

(function () {
  const canvas = document.getElementById('crossDitherBgCanvas');
  if (!canvas) return;

  const wrapper = document.getElementById('crossDitherBackdrop');
  const ctx = canvas.getContext('2d', { alpha: true });
  let width = 0;
  let height = 0;
  let dpr = 1;
  const step = 6; // Compact 6px grid for crisp halftone detail

  let time = 0;
  let lastTimestamp = performance.now();
  let animId = null;
  let isVisible = true;

  // Scroll Parallax & Velocity Tracking
  let currentScroll = 0;
  let targetScroll = 0;
  let scrollVelocity = 0;

  // Pointer & Touch Interaction
  const pointer = {
    x: -1000,
    y: -1000,
    targetX: -1000,
    targetY: -1000,
    active: false,
    strength: 0
  };

  const COLOR_INK = '#111111';
  const COLOR_CANVAS = '#F4F3EF';

  function resize() {
    if (!wrapper) return;
    const rect = wrapper.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = Math.floor(rect.width);
    height = Math.floor(rect.height);

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
  }

  // Topographic Multi-Feature Scalar Field with Dynamic Fluid Motion
  function sampleField(nx, ny, t, scrollNorm, scrollVelNorm) {
    // 1. Dynamic Left Boundary Contour Curve x_edge(ny)
    let x_edge = 0.44;

    // Fluid vertical wave drift combining time and scroll velocity
    const flowY = ny + Math.sin(t * 0.4 + ny * 3.5) * 0.03 + scrollNorm * 0.08;

    // Concave neck indentation at upper third (breathing oscillation)
    const neckCenter = 0.22 + Math.sin(t * 0.8) * 0.035 - scrollNorm * 0.04;
    const neckDist = (flowY - neckCenter) / 0.12;
    x_edge += 0.22 * Math.exp(-neckDist * neckDist);

    // Convex belly lobe protruding outwards at mid-height (undulating pulse)
    const lobeCenter = 0.48 + Math.cos(t * 0.65) * 0.045 + scrollVelNorm * 0.05;
    const lobeDist = (flowY - lobeCenter) / 0.16;
    x_edge -= (0.26 + Math.sin(t * 0.9) * 0.04) * Math.exp(-lobeDist * lobeDist);

    // Sharp diagonal tuck-in beneath the lobe
    const tuckCenter = 0.65 + Math.sin(t * 0.75) * 0.025;
    const tuckDist = (flowY - tuckCenter) / 0.09;
    x_edge += 0.15 * Math.exp(-tuckDist * tuckDist);

    // Terraced lower peninsula (fluid wave roll)
    const lowerCenter = 0.85 + Math.cos(t * 0.55) * 0.035;
    const lowerDist = (flowY - lowerCenter) / 0.15;
    x_edge -= 0.12 * Math.exp(-lowerDist * lowerDist);

    // Multi-harmonic organic contour ripples (traveling waves)
    x_edge += Math.sin(ny * 18.0 - t * 1.6 + scrollNorm * 4.0) * 0.024;
    x_edge += Math.cos(ny * 36.0 + t * 1.1) * 0.012;
    x_edge += Math.sin(ny * 60.0 - t * 2.4) * 0.006;

    // Boundary check: left of silhouette
    if (nx < x_edge) {
      const distOut = x_edge - nx;
      if (distOut > 0.12) {
        return 0.0;
      }
      // Fluid stipple dissipation at the boundary with traveling shimmer
      const scatter = Math.sin(nx * 75.0 + ny * 55.0 + t * 2.0) * 0.06;
      return Math.max(0.0, (1.0 - distOut / 0.12) * 0.24 + scatter);
    }

    // Inside the mass:
    // Penetration depth from outer edge toward right screen edge
    const depth = (nx - x_edge) / (1.0 - x_edge + 0.0001);
    let v = 0.32 + depth * 0.46;

    // A. Dark Core inside the middle belly lobe (pulsating core)
    const lobeCoreX = 0.46 + Math.sin(t * 0.5) * 0.03;
    const lobeCoreY = 0.49 + Math.cos(t * 0.6) * 0.04;
    const dxLobe = (nx - lobeCoreX) * 1.5;
    const dyLobe = (ny - lobeCoreY) * 1.7;
    const dLobe = Math.hypot(dxLobe, dyLobe);
    if (dLobe < 0.52) {
      v += Math.pow(Math.max(0.0, 1.0 - dLobe * 1.9), 1.2) * (0.44 + Math.sin(t * 1.2) * 0.06);
    }

    // B. Upper Right Shoulder Mass (swirling mass)
    const dxTop = (nx - 0.88) * 1.2;
    const dyTop = (ny - (0.10 + Math.sin(t * 0.4) * 0.03)) * 2.0;
    const dTop = Math.hypot(dxTop, dyTop);
    if (dTop < 0.65) {
      v += Math.pow(Math.max(0.0, 1.0 - dTop * 1.5), 1.1) * 0.42;
    }

    // C. Lower Right Mass
    const dxBot = (nx - 0.85) * 1.2;
    const dyBot = (ny - (0.88 + Math.cos(t * 0.45) * 0.03)) * 1.8;
    const dBot = Math.hypot(dxBot, dyBot);
    if (dBot < 0.60) {
      v += Math.pow(Math.max(0.0, 1.0 - dBot * 1.6), 1.1) * 0.40;
    }

    // D. Diagonal White Canyon / Highlight Chasm (rippling fluid fault)
    const chasmShift = Math.sin(t * 0.7) * 0.04;
    const p1x = 0.56 + chasmShift * 0.5, p1y = 0.25 - chasmShift * 0.3;
    const p2x = 0.94 - chasmShift * 0.3, p2y = 0.48 + chasmShift * 0.5;
    const vx = p2x - p1x, vy = p2y - p1y;
    const lenSq = vx * vx + vy * vy;
    const tProj = Math.max(0.0, Math.min(1.0, ((nx - p1x) * vx + (ny - p1y) * vy) / lenSq));
    const closestX = p1x + tProj * vx;
    const closestY = p1y + tProj * vy;
    const distChasm = Math.hypot(nx - closestX, ny - closestY);

    const chasmW = 0.055 + 0.085 * tProj + Math.sin(t * 1.5 + ny * 10.0) * 0.012;
    if (distChasm < chasmW) {
      const chasmStrength = Math.pow(1.0 - distChasm / chasmW, 1.4);
      v -= chasmStrength * (0.74 + Math.sin(t * 1.1) * 0.08);
    }

    // E. Upper Light Island (floating eddy)
    const dIsland = Math.hypot((nx - (0.66 + Math.cos(t * 0.6) * 0.03)) * 1.6, (ny - (0.14 + Math.sin(t * 0.7) * 0.03)) * 2.2);
    if (dIsland < 0.12) {
      v -= Math.pow(1.0 - dIsland / 0.12, 1.4) * 0.40;
    }

    // F. Lower Crease
    const dCrease = Math.hypot((nx - 0.70) * 1.4, (ny - (0.68 + Math.sin(t * 0.5) * 0.02)) * 2.0);
    if (dCrease < 0.14) {
      v -= Math.pow(1.0 - dCrease / 0.14, 1.3) * 0.28;
    }

    // G. Continuous Micro-Topographic Flow Currents
    v += Math.sin(nx * 20.0 + ny * 16.0 - t * 1.4 + scrollNorm * 2.0) * 0.06;
    v += Math.cos(nx * 36.0 - ny * 28.0 + t * 0.9) * 0.04;

    // H. Autonomous Organic Wandering Pulse (Attractor when idle)
    const autox = 0.62 + Math.cos(t * 0.5) * 0.18;
    const autoy = 0.50 + Math.sin(t * 0.4) * 0.32;
    const autoDist = Math.hypot((nx - autox) * 1.4, (ny - autoy) * 1.6);
    if (autoDist < 0.35) {
      const aForce = Math.pow(1.0 - autoDist / 0.35, 1.5);
      v += Math.sin(autoDist * 22.0 - t * 2.8) * 0.14 * aForce;
    }

    // I. Mouse & Touch Interactive Ripple Shockwave
    if (pointer.strength > 0.01) {
      const pdx = (nx * width - pointer.x) / width;
      const pdy = (ny * height - pointer.y) / height;
      const pDist = Math.hypot(pdx, pdy);
      if (pDist < 0.32) {
        const force = (1.0 - pDist / 0.32) * pointer.strength;
        v += Math.sin(pDist * 32.0 - t * 5.0) * 0.28 * force;
      }
    }

    return Math.max(0.0, Math.min(1.0, v));
  }

  function render(now) {
    if (!isVisible) {
      animId = requestAnimationFrame(render);
      return;
    }

    const dt = Math.min((now - lastTimestamp) / 1000, 0.08);
    lastTimestamp = now;
    time += dt * 1.6; // Organic living motion speed

    // Smooth scroll interpolation
    const pageMaxScroll = Math.max(document.body.scrollHeight - window.innerHeight, 1);
    targetScroll = window.scrollY || window.pageYOffset || 0;
    const prevScroll = currentScroll;
    currentScroll += (targetScroll - currentScroll) * 0.12;
    scrollVelocity = (currentScroll - prevScroll);
    const scrollNorm = currentScroll / pageMaxScroll;
    const scrollVelNorm = Math.max(-0.2, Math.min(0.2, scrollVelocity / 30));

    // Smooth pointer interpolation
    pointer.x += (pointer.targetX - pointer.x) * 0.14;
    pointer.y += (pointer.targetY - pointer.y) * 0.14;
    pointer.strength += ((pointer.active ? 1.0 : 0.0) - pointer.strength) * 0.08;

    ctx.save();
    ctx.scale(dpr, dpr);

    // Transparent clear so background grid shows through
    ctx.clearRect(0, 0, width, height);

    const cols = Math.floor(width / step);
    const rows = Math.floor(height / step);

    for (let r = 0; r < rows; r++) {
      const cy = r * step + step / 2;
      const ny = cy / height;

      for (let c = 0; c < cols; c++) {
        const cx = c * step + step / 2;
        const nx = cx / width;

        const val = sampleField(nx, ny, time, scrollNorm, scrollVelNorm);

        if (val < 0.07) {
          continue; // Transparent space
        } else if (val < 0.20) {
          // Level 1: Checkerboard stipple dots at the outer fringe
          if ((r + c) % 2 === 0) {
            ctx.fillStyle = COLOR_INK;
            ctx.beginPath();
            ctx.arc(cx, cy, 0.95, 0, Math.PI * 2);
            ctx.fill();
          }
        } else if (val < 0.36) {
          // Level 2: Crisp dot on every cell
          ctx.fillStyle = COLOR_INK;
          ctx.beginPath();
          ctx.arc(cx, cy, 1.15, 0, Math.PI * 2);
          ctx.fill();
        } else if (val < 0.52) {
          // Level 3: Fine cross '+'
          ctx.fillStyle = COLOR_INK;
          const arm = 2.2;
          ctx.fillRect(cx - arm, cy - 0.5, arm * 2, 1.0);
          ctx.fillRect(cx - 0.5, cy - arm, 1.0, arm * 2);
        } else if (val < 0.72) {
          // Level 4: Medium cross '+' with center weight
          ctx.fillStyle = COLOR_INK;
          const arm = step / 2;
          ctx.fillRect(cx - arm, cy - 0.75, arm * 2, 1.5);
          ctx.fillRect(cx - 0.75, cy - arm, 1.5, arm * 2);
          ctx.fillRect(cx - 1, cy - 1, 2, 2);
        } else if (val < 0.88) {
          // Level 5: Interlocking heavy woven cross
          ctx.fillStyle = COLOR_INK;
          const arm = step / 2 + 0.6;
          ctx.fillRect(cx - arm, cy - 1.1, arm * 2, 2.2);
          ctx.fillRect(cx - 1.1, cy - arm, 2.2, arm * 2);
          ctx.fillRect(cx - 1.6, cy - 1.6, 3.2, 3.2);
        } else {
          // Level 6: Solid black block with canvas color inverted dot
          ctx.fillStyle = COLOR_INK;
          ctx.fillRect(cx - step / 2, cy - step / 2, step, step);
          ctx.fillStyle = COLOR_CANVAS;
          ctx.beginPath();
          ctx.arc(cx, cy, 1.1, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    ctx.restore();
    animId = requestAnimationFrame(render);
  }

  // Pointer position updater helper
  function updatePointer(clientX, clientY) {
    if (!wrapper) return;
    const rect = wrapper.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    if (x >= -80 && x <= rect.width + 80 && y >= -80 && y <= rect.height + 80) {
      pointer.targetX = x;
      pointer.targetY = y;
      pointer.active = true;
    } else {
      pointer.active = false;
    }
  }

  // Mouse event listeners
  window.addEventListener('mousemove', (e) => {
    updatePointer(e.clientX, e.clientY);
  });

  window.addEventListener('mouseleave', () => {
    pointer.active = false;
    pointer.targetX = -1000;
    pointer.targetY = -1000;
  });

  // Touch event listeners for mobile devices
  window.addEventListener('touchstart', (e) => {
    if (e.touches && e.touches[0]) {
      updatePointer(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (e.touches && e.touches[0]) {
      updatePointer(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  window.addEventListener('touchend', () => {
    pointer.active = false;
  }, { passive: true });

  // Hook Lenis scroll events if present
  if (window.lenis) {
    window.lenis.on('scroll', (e) => {
      targetScroll = e.scroll;
    });
  }

  // Resize handler
  window.addEventListener('resize', () => {
    resize();
  });

  // Page visibility optimization
  document.addEventListener('visibilitychange', () => {
    isVisible = !document.hidden;
    if (isVisible) {
      lastTimestamp = performance.now();
    }
  });

  // Initialize
  resize();
  animId = requestAnimationFrame(render);
})();
