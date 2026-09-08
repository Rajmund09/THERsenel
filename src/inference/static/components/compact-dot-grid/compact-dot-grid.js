// compact-dot-grid.js — High-performance 3D Compact Pixel Dot Matrix Background
// Lightweight native WebGL engine creating a textured, 3D perspective dot topography
(function () {
  'use strict';

  // Matrix math helpers
  function mat4Perspective(out, fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = (far + near) * nf;
    out[11] = -1;
    out[12] = 0;
    out[13] = 0;
    out[14] = (2 * far * near) * nf;
    out[15] = 0;
    return out;
  }

  function mat4LookAt(out, eye, center, up) {
    let x0, x1, x2, y0, y1, y2, z0, z1, z2, len;
    const eyex = eye[0], eyey = eye[1], eyez = eye[2];
    const upx = up[0], upy = up[1], upz = up[2];
    const centerx = center[0], centery = center[1], centerz = center[2];

    z0 = eyex - centerx;
    z1 = eyey - centery;
    z2 = eyez - centerz;
    len = 1 / Math.hypot(z0, z1, z2);
    z0 *= len; z1 *= len; z2 *= len;

    x0 = upy * z2 - upz * z1;
    x1 = upz * z0 - upx * z2;
    x2 = upx * z1 - upy * z0;
    len = Math.hypot(x0, x1, x2);
    if (!len) {
      x0 = 0; x1 = 0; x2 = 0;
    } else {
      len = 1 / len;
      x0 *= len; x1 *= len; x2 *= len;
    }

    y0 = z1 * x2 - z2 * x1;
    y1 = z2 * x0 - z0 * x2;
    y2 = z0 * x1 - z1 * x0;
    len = Math.hypot(y0, y1, y2);
    if (!len) {
      y0 = 0; y1 = 0; y2 = 0;
    } else {
      len = 1 / len;
      y0 *= len; y1 *= len; y2 *= len;
    }

    out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
    out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
    out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
    out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
    out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
    out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
    out[15] = 1;
    return out;
  }

  const VERTEX_SHADER = `
attribute vec3 aPosition;

uniform mat4 uProjection;
uniform mat4 uView;
uniform float uTime;
uniform float uDpr;
uniform float uDotSize;
uniform float uElevation;
uniform float uSpeed;

varying float vAlpha;
varying float vLight;

void main() {
  vec3 pos = aPosition;

  // Gentle, hypnotic 3D harmonic wave undulation
  float t = uTime * uSpeed;
  float wave1 = sin(pos.x * 0.012 + t * 0.7) * cos(pos.z * 0.012 + t * 0.5) * uElevation;
  float wave2 = sin(length(pos.xz) * 0.010 - t * 0.6) * (uElevation * 0.65);
  float wave3 = sin(pos.x * 0.025 - pos.z * 0.018 + t * 0.4) * (uElevation * 0.4);
  float totalWave = wave1 + wave2 + wave3;

  pos.y += totalWave;

  vec4 mvPos = uView * vec4(pos, 1.0);
  gl_Position = uProjection * mvPos;

  // 3D Point Size calculation with depth scaling
  float dist = -mvPos.z;
  float baseSize = uDotSize * uDpr;
  float pSize = (baseSize * 480.0) / max(dist, 50.0);
  gl_PointSize = clamp(pSize, 1.0, 4.2 * uDpr);

  // Smooth atmospheric horizon depth attenuation and near-camera fade
  float depthFade = smoothstep(1750.0, 140.0, dist) * smoothstep(30.0, 140.0, dist);
  float crest = clamp(0.35 + (totalWave + uElevation * 1.5) / (uElevation * 3.5), 0.1, 1.0);
  vLight = crest;
  vAlpha = depthFade * (0.25 + 0.70 * crest);
}
`;

  const FRAGMENT_SHADER = `
precision mediump float;

varying float vAlpha;
varying float vLight;

uniform vec3 uColor;

void main() {
  // Crisp compact pixel dot with anti-aliased edge
  vec2 coord = gl_PointCoord - vec2(0.5);
  float r = length(coord);
  if (r > 0.5) discard;

  float alpha = smoothstep(0.5, 0.35, r) * vAlpha;

  // Monochromatic silver/white highlight
  vec3 col = mix(vec3(0.35, 0.38, 0.44), uColor, vLight);

  gl_FragColor = vec4(col, alpha);
}
`;

  function createShader(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`CompactDotGrid shader error: ${err}`);
    }
    return s;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    const vs = createShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const err = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error(`CompactDotGrid program error: ${err}`);
    }
    return p;
  }

  class CompactDotGrid {
    constructor(container, options = {}) {
      if (!container) return;
      this.container = container;

      // Configurable properties
      this.dotSize = options.dotSize ?? (container.dataset.dotSize ? parseFloat(container.dataset.dotSize) : 1.8);
      this.spacing = options.spacing ?? (container.dataset.spacing ? parseFloat(container.dataset.spacing) : 14.0);
      this.speed = options.speed ?? (container.dataset.speed ? parseFloat(container.dataset.speed) : 0.45);
      this.elevation = options.elevation ?? (container.dataset.elevation ? parseFloat(container.dataset.elevation) : 14.0);

      // Monochrome white color
      const hex = (options.color || container.dataset.color || '#FFFFFF').replace('#', '');
      this.color = [
        parseInt(hex.slice(0, 2), 16) / 255,
        parseInt(hex.slice(2, 4), 16) / 255,
        parseInt(hex.slice(4, 6), 16) / 255
      ];

      // Lag-free DPR clamped to 1.0
      const sysDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      this.dpr = options.dpr ?? (container.dataset.dpr ? parseFloat(container.dataset.dpr) : Math.min(sysDpr, 1.0));

      this.rafId = null;
      this.init();
    }

    init() {
      const canvas = document.createElement('canvas');
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.display = 'block';
      canvas.style.pointerEvents = 'none'; // No hover effect
      this.container.appendChild(canvas);
      this.canvas = canvas;

      const gl = canvas.getContext('webgl', {
        alpha: true,
        antialias: false,
        depth: false,
        powerPreference: 'high-performance'
      }) || canvas.getContext('experimental-webgl');

      if (!gl) {
        console.warn('CompactDotGrid: WebGL unavailable.');
        return;
      }
      this.gl = gl;

      // Enable additive blending for glowing dot matrix
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

      const program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
      this.program = program;
      gl.useProgram(program);

      // Generate dense 3D Grid of Points covering entire viewport in perspective
      const spacing = this.spacing;
      const xRange = 1100;
      const zMin = -1200;
      const zMax = 750;

      const points = [];
      for (let z = zMin; z <= zMax; z += spacing) {
        for (let x = -xRange; x <= xRange; x += spacing) {
          points.push(x, 0, z);
        }
      }

      this.pointCount = points.length / 3;
      const posBuffer = gl.createBuffer();
      this.posBuffer = posBuffer;
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.STATIC_DRAW);

      const aPosLoc = gl.getAttribLocation(program, 'aPosition');
      gl.enableVertexAttribArray(aPosLoc);
      gl.vertexAttribPointer(aPosLoc, 3, gl.FLOAT, false, 0, 0);

      // Uniform Locations
      this.uProjLoc = gl.getUniformLocation(program, 'uProjection');
      this.uViewLoc = gl.getUniformLocation(program, 'uView');
      this.uTimeLoc = gl.getUniformLocation(program, 'uTime');
      this.uDprLoc = gl.getUniformLocation(program, 'uDpr');
      this.uDotSizeLoc = gl.getUniformLocation(program, 'uDotSize');
      this.uElevationLoc = gl.getUniformLocation(program, 'uElevation');
      this.uSpeedLoc = gl.getUniformLocation(program, 'uSpeed');
      this.uColorLoc = gl.getUniformLocation(program, 'uColor');

      // Static uniforms
      gl.uniform1f(this.uDprLoc, this.dpr);
      gl.uniform1f(this.uDotSizeLoc, this.dotSize);
      gl.uniform1f(this.uElevationLoc, this.elevation);
      gl.uniform1f(this.uSpeedLoc, this.speed);
      gl.uniform3fv(this.uColorLoc, this.color);

      // Pre-allocated matrices
      this.projMatrix = new Float32Array(16);
      this.viewMatrix = new Float32Array(16);

      // Camera settings: tilted 3D perspective with sweeping depth
      const eye = [0, 260, 480];
      const center = [0, -50, -120];
      const up = [0, 1, 0];
      mat4LookAt(this.viewMatrix, eye, center, up);
      gl.uniformMatrix4fv(this.uViewLoc, false, this.viewMatrix);

      // Resizing
      this.resize = this.resize.bind(this);
      this.resize();
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.container);

      // Loop
      this.loop = this.loop.bind(this);
      this.rafId = requestAnimationFrame(this.loop);
    }

    resize() {
      if (!this.container || !this.canvas || !this.gl) return;
      const rect = this.container.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width * this.dpr));
      const height = Math.max(1, Math.floor(rect.height * this.dpr));

      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.gl.viewport(0, 0, width, height);

        const aspect = width / height;
        mat4Perspective(this.projMatrix, (58 * Math.PI) / 180, aspect, 10.0, 1800.0);
        this.gl.uniformMatrix4fv(this.uProjLoc, false, this.projMatrix);
      }
    }

    loop(t) {
      this.rafId = requestAnimationFrame(this.loop);
      if (document.hidden) return; // Pause when tab is inactive

      const gl = this.gl;
      gl.clearColor(0.0, 0.0, 0.0, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.uniform1f(this.uTimeLoc, t * 0.001);
      gl.drawArrays(gl.POINTS, 0, this.pointCount);
    }

    destroy() {
      if (this.rafId) cancelAnimationFrame(this.rafId);
      if (this.ro) this.ro.disconnect();
      if (this.canvas && this.canvas.parentElement === this.container) {
        this.container.removeChild(this.canvas);
      }
      if (this.gl) {
        if (this.posBuffer) this.gl.deleteBuffer(this.posBuffer);
        if (this.program) this.gl.deleteProgram(this.program);
      }
    }
  }

  function initCompactDotGrid() {
    const targets = document.querySelectorAll('[data-compact-dot-3d], #site-compact-dot-bg');
    targets.forEach((el) => {
      if (!el._compactDotInstance) {
        el._compactDotInstance = new CompactDotGrid(el);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCompactDotGrid);
  } else {
    initCompactDotGrid();
  }

  window.CompactDotGrid = CompactDotGrid;
})();
