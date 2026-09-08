// ferrofluid.js — High-performance Vanilla WebGL implementation of React Bits <Ferrofluid />
// Optimized for smooth, lag-free rendering as a full-page background or container element
(function () {
  'use strict';

  const MAX_COLORS = 8;

  const hexToRGB = (hex) => {
    const c = hex.replace('#', '').padEnd(6, '0');
    const r = parseInt(c.slice(0, 2), 16) / 255;
    const g = parseInt(c.slice(2, 4), 16) / 255;
    const b = parseInt(c.slice(4, 6), 16) / 255;
    return [r, g, b];
  };

  const prepColors = (input) => {
    const base = (input && input.length ? input : ['#ffffff', '#ffffff', '#ffffff']).slice(0, MAX_COLORS);
    const count = base.length;
    const arr = [];
    for (let i = 0; i < MAX_COLORS; i++) {
      arr.push(hexToRGB(base[Math.min(i, base.length - 1)]));
    }
    const avg = [0, 0, 0];
    for (let i = 0; i < count; i++) {
      avg[0] += arr[i][0];
      avg[1] += arr[i][1];
      avg[2] += arr[i][2];
    }
    avg[0] /= count;
    avg[1] /= count;
    avg[2] /= count;
    return { arr, count, avg };
  };

  const flowVec = (d) => {
    switch (d) {
      case 'up':
        return [0, 1];
      case 'down':
        return [0, -1];
      case 'left':
        return [-1, 0];
      case 'right':
        return [1, 0];
      default:
        return [0, -1];
    }
  };

  const VERTEX_SHADER = `
attribute vec2 position;
attribute vec2 uv;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

  const FRAGMENT_SHADER = `
precision highp float;

uniform vec3  iResolution;
uniform vec2  iMouse;
uniform float iTime;

uniform vec3  uColor0;
uniform vec3  uColor1;
uniform vec3  uColor2;
uniform vec3  uColor3;
uniform vec3  uColor4;
uniform vec3  uColor5;
uniform vec3  uColor6;
uniform vec3  uColor7;
uniform int   uColorCount;

uniform vec3  uMouseColor;
uniform vec2  uFlow;
uniform float uSpeed;
uniform float uScale;
uniform float uTurbulence;
uniform float uFluidity;
uniform float uRimWidth;
uniform float uSharpness;
uniform float uShimmer;
uniform float uGlow;
uniform float uOpacity;
uniform float uMouseEnabled;
uniform float uMouseStrength;
uniform float uMouseRadius;

varying vec2 vUv;

#define PI 3.14159265

vec3 palette(float h) {
  int count = uColorCount;
  if (count < 1) count = 1;
  int idx = int(floor(clamp(h, 0.0, 0.999999) * float(count)));
  if (idx <= 0) return uColor0;
  if (idx == 1) return uColor1;
  if (idx == 2) return uColor2;
  if (idx == 3) return uColor3;
  if (idx == 4) return uColor4;
  if (idx == 5) return uColor5;
  if (idx == 6) return uColor6;
  return uColor7;
}

float hash(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float smin(float a, float b, float k) {
  float r = exp2(-a / k) + exp2(-b / k);
  return -k * log2(r);
}

float sinlerp(float a, float b, float w) {
  return mix(a, b, (sin(w * PI - PI / 2.0) + 1.0) / 2.0);
}

float vn(vec2 p, float s, float seed) {
  vec2 cellp = floor(p / s);
  vec2 relp = mod(p, s);
  float g1 = hash(vec3(cellp, seed));
  float g2 = hash(vec3(cellp.x + 1.0, cellp.y, seed));
  float g3 = hash(vec3(cellp.x + 1.0, cellp.y + 1.0, seed));
  float g4 = hash(vec3(cellp.x, cellp.y + 1.0, seed));
  float bx = sinlerp(g1, g2, relp.x / s);
  float tx = sinlerp(g4, g3, relp.x / s);
  return sinlerp(bx, tx, relp.y / s);
}

float dbn(vec2 p, float s, float seed) {
  float o = s / 2.0;
  float n0 = vn(p, s, seed);
  float n1 = vn(p + vec2(o, o), s, seed + 0.1);
  float n2 = vn(p + vec2(-o, o), s, seed + 0.2);
  float n3 = vn(p + vec2(o, -o), s, seed + 0.3);
  float n4 = vn(p + vec2(-o, -o), s, seed + 0.4);
  return (2.0 * n0 + 1.5 * n1 + 1.25 * n2 + 1.125 * n3 + n4) / 7.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  float ref = 700.0 / max(uScale, 0.05);
  vec2 p = fragCoord / iResolution.y * ref;

  float spd = 200.0 * uSpeed;
  float t = iTime;

  vec2 dir = uFlow;
  vec2 perp = vec2(-dir.y, dir.x);

  float distort1 = vn(p + perp * (t * spd), 60.0, 10.0) * 50.0 * uTurbulence;
  float distort2 = vn(p - perp * (t * spd), 120.0, 15.0) * 100.0 * uTurbulence;

  float peaks = dbn(p + distort1 + dir * (t * spd * 0.5), 40.0, 1.0);
  float peaks2 = dbn(p + distort2 - dir * (t * spd * 0.5), 40.0, 0.0);

  float mapeaks = smin(peaks, peaks2, max(uFluidity, 0.001));

  float mGlow = 0.0;
  if (uMouseEnabled > 0.5) {
    vec2 mp = iMouse / iResolution.y * ref;
    float md = length(p - mp) / ref;
    float rr = max(uMouseRadius, 0.02);
    mGlow = exp(-md * md / (rr * rr)) * uMouseStrength;
  }

  float band = (uRimWidth - abs((mapeaks - 0.4) * 2.0)) * 5.0;
  float ltn = clamp(band - vn(p + dir * (t * spd * 0.5), 60.0, 12.0) * uShimmer, 0.0, 1.0);
  ltn = pow(ltn, uSharpness) * uGlow;
  if (uMouseEnabled > 0.5) {
    ltn *= clamp(1.0 - mGlow, 0.0, 1.0);
  }

  float h = clamp(0.5 + (peaks - peaks2) * 0.8, 0.0, 1.0);
  vec3 col = palette(h);

  vec3 outc = col * ltn;
  float a = clamp(max(outc.r, max(outc.g, outc.b)), 0.0, 1.0);
  fragColor = vec4(outc, a * uOpacity);
}

void main() {
  vec4 color;
  mainImage(color, vUv * iResolution.xy);
  gl_FragColor = color;
}
`;

  function createShader(gl, type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`Ferrofluid shader error: ${err}`);
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
      throw new Error(`Ferrofluid program error: ${err}`);
    }
    return p;
  }

  class Ferrofluid {
    constructor(container, options = {}) {
      if (!container) {
        console.warn('Ferrofluid: Container element not found.');
        return;
      }
      this.container = container;

      // Color scheme: Black and White by default
      let colors = options.colors;
      if (!colors && container.dataset.colors) {
        try {
          colors = JSON.parse(container.dataset.colors);
        } catch (e) {
          colors = ['#ffffff', '#ffffff', '#ffffff'];
        }
      }
      if (!colors || !colors.length) {
        colors = ['#ffffff', '#ffffff', '#ffffff'];
      }

      this.colors = colors;
      this.speed = options.speed ?? (container.dataset.speed ? parseFloat(container.dataset.speed) : 0.4);
      this.scale = options.scale ?? (container.dataset.scale ? parseFloat(container.dataset.scale) : 0.3);
      this.turbulence = options.turbulence ?? (container.dataset.turbulence ? parseFloat(container.dataset.turbulence) : 1.0);
      this.fluidity = options.fluidity ?? (container.dataset.fluidity ? parseFloat(container.dataset.fluidity) : 0.07);
      this.rimWidth = options.rimWidth ?? (container.dataset.rimWidth ? parseFloat(container.dataset.rimWidth) : 0.25);
      this.sharpness = options.sharpness ?? (container.dataset.sharpness ? parseFloat(container.dataset.sharpness) : 1.0);
      this.shimmer = options.shimmer ?? (container.dataset.shimmer ? parseFloat(container.dataset.shimmer) : 1.3);
      this.glow = options.glow ?? (container.dataset.glow ? parseFloat(container.dataset.glow) : 4.5);
      this.flowDirection = options.flowDirection ?? (container.dataset.flowDirection || 'down');
      this.opacity = options.opacity ?? (container.dataset.opacity ? parseFloat(container.dataset.opacity) : 1.0);

      // Disable hover/mouse effect on background as requested
      this.mouseInteraction = options.mouseInteraction ?? (container.dataset.mouseInteraction === 'true');
      this.mouseStrength = options.mouseStrength ?? (container.dataset.mouseStrength ? parseFloat(container.dataset.mouseStrength) : 0.0);
      this.mouseRadius = options.mouseRadius ?? (container.dataset.mouseRadius ? parseFloat(container.dataset.mouseRadius) : 0.4);
      this.mouseDampening = options.mouseDampening ?? 0.15;
      this.paused = options.paused ?? false;

      // High-performance DPR clamp: Max 1.0 for lag-free 60fps rendering across all displays
      const sysDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      this.dpr = options.dpr ?? (container.dataset.dpr ? parseFloat(container.dataset.dpr) : Math.min(sysDpr, 1.0));

      this.mouseTarget = [0, 0];
      this.currentMouse = [0, 0];
      this.lastTime = 0;
      this.rafId = null;

      this.init();
    }

    init() {
      const canvas = document.createElement('canvas');
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.display = 'block';
      if (!this.mouseInteraction) {
        canvas.style.pointerEvents = 'none';
      }
      this.container.appendChild(canvas);
      this.canvas = canvas;

      const gl = canvas.getContext('webgl', {
        alpha: true,
        antialias: false,
        depth: false,
        stencil: false,
        powerPreference: 'high-performance'
      }) || canvas.getContext('experimental-webgl');

      if (!gl) {
        console.warn('Ferrofluid: WebGL not supported.');
        return;
      }
      this.gl = gl;

      // Create Shader Program
      const program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
      this.program = program;
      gl.useProgram(program);

      // Full-screen Quad (2 triangles)
      const vertices = new Float32Array([
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
        -1.0,  1.0,  0.0, 1.0,
        -1.0,  1.0,  0.0, 1.0,
         1.0, -1.0,  1.0, 0.0,
         1.0,  1.0,  1.0, 1.0
      ]);

      const buffer = gl.createBuffer();
      this.buffer = buffer;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

      const posLoc = gl.getAttribLocation(program, 'position');
      const uvLoc = gl.getAttribLocation(program, 'uv');
      const FSIZE = Float32Array.BYTES_PER_ELEMENT;

      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, FSIZE * 4, 0);

      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, FSIZE * 4, FSIZE * 2);

      // Cache Uniform Locations
      this.uniforms = {
        iResolution: gl.getUniformLocation(program, 'iResolution'),
        iMouse: gl.getUniformLocation(program, 'iMouse'),
        iTime: gl.getUniformLocation(program, 'iTime'),
        uColorCount: gl.getUniformLocation(program, 'uColorCount'),
        uMouseColor: gl.getUniformLocation(program, 'uMouseColor'),
        uFlow: gl.getUniformLocation(program, 'uFlow'),
        uSpeed: gl.getUniformLocation(program, 'uSpeed'),
        uScale: gl.getUniformLocation(program, 'uScale'),
        uTurbulence: gl.getUniformLocation(program, 'uTurbulence'),
        uFluidity: gl.getUniformLocation(program, 'uFluidity'),
        uRimWidth: gl.getUniformLocation(program, 'uRimWidth'),
        uSharpness: gl.getUniformLocation(program, 'uSharpness'),
        uShimmer: gl.getUniformLocation(program, 'uShimmer'),
        uGlow: gl.getUniformLocation(program, 'uGlow'),
        uOpacity: gl.getUniformLocation(program, 'uOpacity'),
        uMouseEnabled: gl.getUniformLocation(program, 'uMouseEnabled'),
        uMouseStrength: gl.getUniformLocation(program, 'uMouseStrength'),
        uMouseRadius: gl.getUniformLocation(program, 'uMouseRadius'),
        colors: []
      };

      for (let i = 0; i < MAX_COLORS; i++) {
        this.uniforms.colors.push(gl.getUniformLocation(program, `uColor${i}`));
      }

      // Initialize Colors and Static Uniforms
      this.applyColors();
      this.applyUniforms();

      // Setup Resizing
      this.resize = this.resize.bind(this);
      this.resize();

      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.container);

      // Pointer moves only if explicitly enabled
      if (this.mouseInteraction) {
        this.onPointerMove = (e) => {
          const rect = this.canvas.getBoundingClientRect();
          const x = (e.clientX - rect.left) * this.dpr;
          const y = (rect.height - (e.clientY - rect.top)) * this.dpr;
          this.mouseTarget = [x, y];
          if (this.mouseDampening <= 0) {
            this.currentMouse = [x, y];
          }
        };
        this.canvas.addEventListener('pointermove', this.onPointerMove);
      }

      // Render Loop
      this.loop = this.loop.bind(this);
      this.rafId = requestAnimationFrame(this.loop);
    }

    applyColors() {
      const gl = this.gl;
      const { arr, count, avg } = prepColors(this.colors);
      gl.uniform1i(this.uniforms.uColorCount, count);
      gl.uniform3fv(this.uniforms.uMouseColor, avg);
      for (let i = 0; i < MAX_COLORS; i++) {
        gl.uniform3fv(this.uniforms.colors[i], arr[i]);
      }
    }

    applyUniforms() {
      const gl = this.gl;
      gl.uniform2fv(this.uniforms.uFlow, flowVec(this.flowDirection));
      gl.uniform1f(this.uniforms.uSpeed, this.speed);
      gl.uniform1f(this.uniforms.uScale, this.scale);
      gl.uniform1f(this.uniforms.uTurbulence, this.turbulence);
      gl.uniform1f(this.uniforms.uFluidity, this.fluidity);
      gl.uniform1f(this.uniforms.uRimWidth, this.rimWidth);
      gl.uniform1f(this.uniforms.uSharpness, this.sharpness);
      gl.uniform1f(this.uniforms.uShimmer, this.shimmer);
      gl.uniform1f(this.uniforms.uGlow, this.glow);
      gl.uniform1f(this.uniforms.uOpacity, this.opacity);
      gl.uniform1f(this.uniforms.uMouseEnabled, this.mouseInteraction ? 1.0 : 0.0);
      gl.uniform1f(this.uniforms.uMouseStrength, this.mouseStrength);
      gl.uniform1f(this.uniforms.uMouseRadius, this.mouseRadius);
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
        this.gl.uniform3f(this.uniforms.iResolution, width, height, 1.0);
      }
    }

    loop(t) {
      this.rafId = requestAnimationFrame(this.loop);

      // Pause rendering when tab is hidden to save GPU & battery
      if (this.paused || document.hidden) return;

      const gl = this.gl;
      gl.uniform1f(this.uniforms.iTime, t * 0.001);

      if (this.mouseInteraction) {
        if (this.mouseDampening > 0) {
          if (!this.lastTime) this.lastTime = t;
          const dt = (t - this.lastTime) / 1000;
          this.lastTime = t;
          const tau = Math.max(1e-4, this.mouseDampening);
          let factor = 1 - Math.exp(-dt / tau);
          if (factor > 1) factor = 1;
          this.currentMouse[0] += (this.mouseTarget[0] - this.currentMouse[0]) * factor;
          this.currentMouse[1] += (this.mouseTarget[1] - this.currentMouse[1]) * factor;
        }
        gl.uniform2f(this.uniforms.iMouse, this.currentMouse[0], this.currentMouse[1]);
      }

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    destroy() {
      if (this.rafId) cancelAnimationFrame(this.rafId);
      if (this.ro) this.ro.disconnect();
      if (this.canvas && this.onPointerMove) {
        this.canvas.removeEventListener('pointermove', this.onPointerMove);
      }
      if (this.canvas && this.canvas.parentElement === this.container) {
        this.container.removeChild(this.canvas);
      }
      if (this.gl) {
        if (this.buffer) this.gl.deleteBuffer(this.buffer);
        if (this.program) this.gl.deleteProgram(this.program);
      }
    }
  }

  // Auto-initialize any element with [data-ferrofluid] or #site-ferrofluid-bg
  function initFerrofluid() {
    const targets = document.querySelectorAll('[data-ferrofluid], #site-ferrofluid-bg');
    targets.forEach((el) => {
      if (!el._ferrofluidInstance) {
        el._ferrofluidInstance = new Ferrofluid(el);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFerrofluid);
  } else {
    initFerrofluid();
  }

  window.Ferrofluid = Ferrofluid;
})();
