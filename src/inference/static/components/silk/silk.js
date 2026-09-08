// silk.js — High-performance Vanilla WebGL implementation of React Bits <Silk />
// Optimized for smooth, lag-free rendering as a unified full-site background
(function () {
  'use strict';

  const hexToNormalizedRGB = (hex) => {
    hex = hex.replace('#', '').padEnd(6, '0');
    return [
      parseInt(hex.slice(0, 2), 16) / 255,
      parseInt(hex.slice(2, 4), 16) / 255,
      parseInt(hex.slice(4, 6), 16) / 255
    ];
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

varying vec2 vUv;

uniform float uTime;
uniform vec3  uColor;
uniform float uSpeed;
uniform float uScale;
uniform float uRotation;
uniform float uNoiseIntensity;
uniform float uLightMode;

const float e = 2.71828182845904523536;

float noise(vec2 texCoord) {
  float G = e;
  vec2  r = (G * sin(G * texCoord));
  return fract(r.x * r.y * (1.0 + texCoord.x));
}

vec2 rotateUvs(vec2 uv, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  mat2  rot = mat2(c, -s, s, c);
  return rot * uv;
}

void main() {
  float rnd        = noise(gl_FragCoord.xy);
  vec2  uv         = rotateUvs(vUv * uScale, uRotation);
  vec2  tex        = uv * uScale;
  float tOffset    = uSpeed * uTime;

  tex.y += 0.03 * sin(8.0 * tex.x - tOffset);

  float pattern = 0.6 +
                  0.4 * sin(5.0 * (tex.x + tex.y +
                                   cos(3.0 * tex.x + 5.0 * tex.y) +
                                   0.02 * tOffset) +
                           sin(20.0 * (tex.x + tex.y - 0.1 * tOffset)));

  float grain = rnd / 15.0 * uNoiseIntensity;
  vec3 result = uColor * pattern - vec3(grain);

  if (uLightMode > 0.5) {
    float fold = smoothstep(0.28, 0.9, pattern);
    float specular = smoothstep(0.72, 0.98, pattern);
    vec3 shadowColor = uColor * 0.72;
    vec3 bodyColor = min(uColor * 1.18, vec3(1.0));
    vec3 lightBase = mix(shadowColor, bodyColor, fold);
    lightBase = mix(lightBase, vec3(1.0), specular * 0.92);
    float fineNoise = noise(gl_FragCoord.xy * 0.63 + vec2(17.0, 41.0));
    float grainSignal = (rnd + fineNoise - 1.0);
    float grainStrength = clamp(uNoiseIntensity * 0.038, 0.0, 0.16);
    result = lightBase + grainSignal * grainStrength;
  }

  gl_FragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
}
`;

  function createShader(gl, type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`Silk shader error: ${err}`);
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
      throw new Error(`Silk program error: ${err}`);
    }
    return p;
  }

  class Silk {
    constructor(container, options = {}) {
      if (!container) return;
      this.container = container;

      // Props parsing
      this.speed = options.speed ?? (container.dataset.speed ? parseFloat(container.dataset.speed) : 3.1);
      this.scale = options.scale ?? (container.dataset.scale ? parseFloat(container.dataset.scale) : 5.0);
      this.noiseIntensity = options.noiseIntensity ?? (container.dataset.noiseIntensity ? parseFloat(container.dataset.noiseIntensity) : 8.9);
      this.rotation = options.rotation ?? (container.dataset.rotation ? parseFloat(container.dataset.rotation) : 3.61);
      this.lightMode = options.lightMode ?? (container.dataset.lightMode === 'true');

      const colorHex = options.color || container.dataset.color || '#ffffff';
      this.color = hexToNormalizedRGB(colorHex);

      const sysDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      this.dpr = options.dpr ?? (container.dataset.dpr ? parseFloat(container.dataset.dpr) : Math.min(sysDpr, 1.0));

      this.uTime = 0;
      this.lastTime = 0;
      this.rafId = null;

      this.init();
    }

    init() {
      const canvas = document.createElement('canvas');
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.display = 'block';
      canvas.style.pointerEvents = 'none'; // Background has no pointer events
      this.container.appendChild(canvas);
      this.canvas = canvas;

      const gl = canvas.getContext('webgl', {
        alpha: false,
        antialias: false,
        depth: false,
        stencil: false,
        powerPreference: 'high-performance'
      }) || canvas.getContext('experimental-webgl');

      if (!gl) {
        console.warn('Silk: WebGL unavailable.');
        return;
      }
      this.gl = gl;

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
        uTime: gl.getUniformLocation(program, 'uTime'),
        uColor: gl.getUniformLocation(program, 'uColor'),
        uSpeed: gl.getUniformLocation(program, 'uSpeed'),
        uScale: gl.getUniformLocation(program, 'uScale'),
        uRotation: gl.getUniformLocation(program, 'uRotation'),
        uNoiseIntensity: gl.getUniformLocation(program, 'uNoiseIntensity'),
        uLightMode: gl.getUniformLocation(program, 'uLightMode')
      };

      // Set static uniforms
      gl.uniform3fv(this.uniforms.uColor, this.color);
      gl.uniform1f(this.uniforms.uSpeed, this.speed);
      gl.uniform1f(this.uniforms.uScale, this.scale);
      gl.uniform1f(this.uniforms.uRotation, this.rotation);
      gl.uniform1f(this.uniforms.uNoiseIntensity, this.noiseIntensity);
      gl.uniform1f(this.uniforms.uLightMode, this.lightMode ? 1.0 : 0.0);

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
      }
    }

    loop(now) {
      this.rafId = requestAnimationFrame(this.loop);
      if (document.hidden) return; // Save GPU when inactive

      if (!this.lastTime) this.lastTime = now;
      const delta = Math.min((now - this.lastTime) / 1000, 0.1);
      this.lastTime = now;

      // Increment time by 0.1 * delta (exact match to R3F Silk implementation)
      this.uTime += 0.1 * delta;

      const gl = this.gl;
      gl.uniform1f(this.uniforms.uTime, this.uTime);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    destroy() {
      if (this.rafId) cancelAnimationFrame(this.rafId);
      if (this.ro) this.ro.disconnect();
      if (this.canvas && this.canvas.parentElement === this.container) {
        this.container.removeChild(this.canvas);
      }
      if (this.gl) {
        if (this.buffer) this.gl.deleteBuffer(this.buffer);
        if (this.program) this.gl.deleteProgram(this.program);
      }
    }
  }

  function initSilk() {
    const targets = document.querySelectorAll('[data-silk], #site-silk-bg');
    targets.forEach((el) => {
      if (!el._silkInstance) {
        el._silkInstance = new Silk(el);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSilk);
  } else {
    initSilk();
  }

  window.Silk = Silk;
})();
