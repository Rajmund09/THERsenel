// liquid-chrome.js — Vanilla WebGL implementation of React Bits <LiquidChrome />
(function () {
  'use strict';

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
    uniform float uTime;
    uniform vec3 uResolution;
    uniform vec3 uBaseColor;
    uniform float uAmplitude;
    uniform float uFrequencyX;
    uniform float uFrequencyY;
    uniform vec2 uMouse;
    varying vec2 vUv;

    vec4 renderImage(vec2 uvCoord) {
        vec2 fragCoord = uvCoord * uResolution.xy;
        vec2 uv = (2.0 * fragCoord - uResolution.xy) / min(uResolution.x, uResolution.y);

        for (float i = 1.0; i < 10.0; i++){
            uv.x += uAmplitude / i * cos(i * uFrequencyX * uv.y + uTime + uMouse.x * 3.14159);
            uv.y += uAmplitude / i * cos(i * uFrequencyY * uv.x + uTime + uMouse.y * 3.14159);
        }

        vec2 diff = (uvCoord - uMouse);
        float dist = length(diff);
        float falloff = exp(-dist * 20.0);
        float ripple = sin(10.0 * dist - uTime * 2.0) * 0.03;
        uv += (diff / (dist + 0.0001)) * ripple * falloff;

        vec3 color = uBaseColor / abs(sin(uTime - uv.y - uv.x));
        return vec4(color, 1.0);
    }

    void main() {
        vec4 col = vec4(0.0);
        int samples = 0;
        for (int i = -1; i <= 1; i++){
            for (int j = -1; j <= 1; j++){
                vec2 offset = vec2(float(i), float(j)) * (1.0 / min(uResolution.x, uResolution.y));
                col += renderImage(vUv + offset);
                samples++;
            }
        }
        gl_FragColor = col / float(samples);
    }
  `;

  function createShader(gl, type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`LiquidChrome shader error: ${err}`);
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
      throw new Error(`LiquidChrome program link error: ${err}`);
    }
    return p;
  }

  class LiquidChrome {
    constructor(container, options = {}) {
      if (!container) {
        console.warn('LiquidChrome: Container element not found.');
        return;
      }

      this.container = container;

      // Props parsing
      let baseColor = options.baseColor;
      if (!baseColor && container.dataset.baseColor) {
        const raw = container.dataset.baseColor.trim();
        if (raw.startsWith('#')) {
          const hex = raw.replace('#', '');
          const r = parseInt(hex.substring(0, 2), 16) / 255;
          const g = parseInt(hex.substring(2, 4), 16) / 255;
          const b = parseInt(hex.substring(4, 6), 16) / 255;
          baseColor = [r, g, b];
        } else {
          baseColor = raw.split(',').map(Number);
        }
      }
      this.baseColor = baseColor || [0.137255, 0.007843, 0.007843];
      this.speed = options.speed ?? (parseFloat(container.dataset.speed) || 0.2);
      this.amplitude = options.amplitude ?? (parseFloat(container.dataset.amplitude) || 0.3);
      this.frequencyX = options.frequencyX ?? (parseFloat(container.dataset.frequencyX) || 3.0);
      this.frequencyY = options.frequencyY ?? (parseFloat(container.dataset.frequencyY) || 3.0);
      this.interactive = options.interactive ?? (container.dataset.interactive !== 'false');

      this.mouse = [0.5, 0.5];
      this.targetMouse = [0.5, 0.5];
      this.animFrame = null;

      this.initCanvas();
      this.initGL();
      this.bindEvents();
      this.resize();
      this.start();
    }

    initCanvas() {
      this.canvas = document.createElement('canvas');
      this.canvas.style.display = 'block';
      this.canvas.style.width = '100%';
      this.canvas.style.height = '100%';
      this.container.appendChild(this.canvas);
    }

    initGL() {
      const gl = this.canvas.getContext('webgl', { antialias: true, alpha: true }) ||
                 this.canvas.getContext('experimental-webgl', { antialias: true, alpha: true });
      if (!gl) {
        console.warn('LiquidChrome: WebGL not supported on this browser.');
        return;
      }
      this.gl = gl;

      gl.clearColor(0.05, 0.05, 0.06, 1.0);

      this.program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);

      // Full-screen Quad geometry
      const positions = new Float32Array([
        -1, -1,
         1, -1,
        -1,  1,
        -1,  1,
         1, -1,
         1,  1
      ]);

      const uvs = new Float32Array([
        0, 0,
        1, 0,
        0, 1,
        0, 1,
        1, 0,
        1, 1
      ]);

      this.posBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

      this.posLoc = gl.getAttribLocation(this.program, 'position');
      gl.enableVertexAttribArray(this.posLoc);
      gl.vertexAttribPointer(this.posLoc, 2, gl.FLOAT, false, 0, 0);

      this.uvBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);

      this.uvLoc = gl.getAttribLocation(this.program, 'uv');
      gl.enableVertexAttribArray(this.uvLoc);
      gl.vertexAttribPointer(this.uvLoc, 2, gl.FLOAT, false, 0, 0);

      // Uniform Locations
      this.uTimeLoc = gl.getUniformLocation(this.program, 'uTime');
      this.uResolutionLoc = gl.getUniformLocation(this.program, 'uResolution');
      this.uBaseColorLoc = gl.getUniformLocation(this.program, 'uBaseColor');
      this.uAmplitudeLoc = gl.getUniformLocation(this.program, 'uAmplitude');
      this.uFrequencyXLoc = gl.getUniformLocation(this.program, 'uFrequencyX');
      this.uFrequencyYLoc = gl.getUniformLocation(this.program, 'uFrequencyY');
      this.uMouseLoc = gl.getUniformLocation(this.program, 'uMouse');
    }

    resize() {
      if (!this.gl || !this.container) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const w = Math.max(10, Math.floor(this.container.clientWidth));
      const h = Math.max(10, Math.floor(this.container.clientHeight));

      this.width = Math.round(w * dpr);
      this.height = Math.round(h * dpr);

      this.canvas.width = this.width;
      this.canvas.height = this.height;
      this.gl.viewport(0, 0, this.width, this.height);
    }

    bindEvents() {
      this.resizeObserver = new ResizeObserver(() => {
        this.resize();
      });
      this.resizeObserver.observe(this.container);

      if (this.interactive) {
        // Track mouse over container and parent header
        const target = this.container.closest('.metallic-black-header') || this.container;

        this.handleMouseMove = (e) => {
          const rect = target.getBoundingClientRect();
          const x = (e.clientX - rect.left) / rect.width;
          const y = 1.0 - (e.clientY - rect.top) / rect.height;
          this.targetMouse = [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))];
        };

        this.handleTouchMove = (e) => {
          if (e.touches.length > 0) {
            const touch = e.touches[0];
            const rect = target.getBoundingClientRect();
            const x = (touch.clientX - rect.left) / rect.width;
            const y = 1.0 - (touch.clientY - rect.top) / rect.height;
            this.targetMouse = [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))];
          }
        };

        target.addEventListener('mousemove', this.handleMouseMove);
        target.addEventListener('touchmove', this.handleTouchMove, { passive: true });
      }
    }

    start() {
      if (!this.gl) return;
      const gl = this.gl;

      const render = (now) => {
        this.animFrame = requestAnimationFrame(render);

        // Smooth mouse interpolation
        this.mouse[0] += (this.targetMouse[0] - this.mouse[0]) * 0.08;
        this.mouse[1] += (this.targetMouse[1] - this.mouse[1]) * 0.08;

        gl.useProgram(this.program);

        gl.uniform1f(this.uTimeLoc, now * 0.001 * this.speed);
        gl.uniform3f(this.uResolutionLoc, this.width, this.height, this.width / Math.max(1, this.height));
        gl.uniform3f(this.uBaseColorLoc, this.baseColor[0], this.baseColor[1], this.baseColor[2]);
        gl.uniform1f(this.uAmplitudeLoc, this.amplitude);
        gl.uniform1f(this.uFrequencyXLoc, this.frequencyX);
        gl.uniform1f(this.uFrequencyYLoc, this.frequencyY);
        gl.uniform2f(this.uMouseLoc, this.mouse[0], this.mouse[1]);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
        gl.vertexAttribPointer(this.posLoc, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
        gl.vertexAttribPointer(this.uvLoc, 2, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.TRIANGLES, 0, 6);
      };

      this.animFrame = requestAnimationFrame(render);
    }

    destroy() {
      if (this.animFrame) cancelAnimationFrame(this.animFrame);
      if (this.resizeObserver) this.resizeObserver.disconnect();
      const target = this.container.closest('.metallic-black-header') || this.container;
      if (this.handleMouseMove) target.removeEventListener('mousemove', this.handleMouseMove);
      if (this.handleTouchMove) target.removeEventListener('touchmove', this.handleTouchMove);
      if (this.canvas && this.canvas.parentNode) {
        this.canvas.parentNode.removeChild(this.canvas);
      }
      this.gl?.getExtension('WEBGL_lose_context')?.loseContext();
    }
  }

  // Auto-initialize on elements with [data-liquid-chrome]
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-liquid-chrome]').forEach((el) => {
      new LiquidChrome(el);
    });
  });

  window.LiquidChrome = LiquidChrome;
})();
