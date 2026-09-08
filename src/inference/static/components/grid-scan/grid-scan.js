// grid-scan.js — Vanilla WebGL implementation of React Bits <GridScan />
// High-performance perspective scanning grid with interactive mouse tilt & click-scan
(function () {
  'use strict';

  const VERTEX_SHADER = `
    attribute vec2 position;
    attribute vec2 uv;
    varying vec2 vUv;
    void main(){
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `;

  const FRAGMENT_SHADER = `
    #extension GL_OES_standard_derivatives : enable
    precision highp float;
    uniform vec3 iResolution;
    uniform float iTime;
    uniform vec2 uSkew;
    uniform float uTilt;
    uniform float uYaw;
    uniform float uLineThickness;
    uniform vec3 uLinesColor;
    uniform vec3 uScanColor;
    uniform float uGridScale;
    uniform float uLineStyle;
    uniform float uLineJitter;
    uniform float uScanOpacity;
    uniform float uScanDirection;
    uniform float uNoise;
    uniform float uBloomOpacity;
    uniform float uScanGlow;
    uniform float uScanSoftness;
    uniform float uPhaseTaper;
    uniform float uScanDuration;
    uniform float uScanDelay;
    uniform float uLightMode;
    varying vec2 vUv;

    uniform float uScanStarts[8];
    uniform float uScanCount;

    const int MAX_SCANS = 8;

    float smoother01(float a, float b, float x){
      float t = clamp((x - a) / max(1e-5, (b - a)), 0.0, 1.0);
      return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }

    void mainImage(out vec4 fragColor, in vec2 fragCoord)
    {
        vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;

        vec3 ro = vec3(0.0);
        vec3 rd = normalize(vec3(p, 2.0));

        float cR = cos(uTilt), sR = sin(uTilt);
        rd.xy = mat2(cR, -sR, sR, cR) * rd.xy;

        float cY = cos(uYaw), sY = sin(uYaw);
        rd.xz = mat2(cY, -sY, sY, cY) * rd.xz;

        vec2 skew = clamp(uSkew, vec2(-0.7), vec2(0.7));
        rd.xy += skew * rd.z;

        vec3 color = vec3(0.0);
        float minT = 1e20;
        float gridScale = max(1e-5, uGridScale);
        float fadeStrength = 2.0;
        vec2 gridUV = vec2(0.0);

        float hitIsY = 1.0;
        for (int i = 0; i < 4; i++)
        {
            float isY = float(i < 2);
            float pos = mix(-0.2, 0.2, float(i)) * isY + mix(-0.5, 0.5, float(i - 2)) * (1.0 - isY);
            float num = pos - (isY * ro.y + (1.0 - isY) * ro.x);
            float den = isY * rd.y + (1.0 - isY) * rd.x;
            float t = num / den;
            vec3 h = ro + rd * t;

            float depthBoost = smoothstep(0.0, 3.0, h.z);
            h.xy += skew * 0.15 * depthBoost;

            bool use = t > 0.0 && t < minT;
            gridUV = use ? mix(h.zy, h.xz, isY) / gridScale : gridUV;
            minT = use ? t : minT;
            hitIsY = use ? isY : hitIsY;
        }

        vec3 hit = ro + rd * minT;
        float dist = length(hit - ro);

        float jitterAmt = clamp(uLineJitter, 0.0, 1.0);
        if (jitterAmt > 0.0) {
          vec2 j = vec2(
            sin(gridUV.y * 2.7 + iTime * 1.8),
            cos(gridUV.x * 2.3 - iTime * 1.6)
          ) * (0.15 * jitterAmt);
          gridUV += j;
        }
        float fx = fract(gridUV.x);
        float fy = fract(gridUV.y);
        float ax = min(fx, 1.0 - fx);
        float ay = min(fy, 1.0 - fy);
        float wx = fwidth(gridUV.x);
        float wy = fwidth(gridUV.y);
        float halfPx = max(0.0, uLineThickness) * 0.5;

        float tx = halfPx * wx;
        float ty = halfPx * wy;

        float aax = wx;
        float aay = wy;

        float lineX = 1.0 - smoothstep(tx, tx + aax, ax);
        float lineY = 1.0 - smoothstep(ty, ty + aay, ay);
        if (uLineStyle > 0.5) {
          float dashRepeat = 4.0;
          float dashDuty = 0.5;
          float vy = fract(gridUV.y * dashRepeat);
          float vx = fract(gridUV.x * dashRepeat);
          float dashMaskY = step(vy, dashDuty);
          float dashMaskX = step(vx, dashDuty);
          if (uLineStyle < 1.5) {
            lineX *= dashMaskY;
            lineY *= dashMaskX;
          } else {
            float dotRepeat = 6.0;
            float dotWidth = 0.18;
            float cy = abs(fract(gridUV.y * dotRepeat) - 0.5);
            float cx = abs(fract(gridUV.x * dotRepeat) - 0.5);
            float dotMaskY = 1.0 - smoothstep(dotWidth, dotWidth + fwidth(gridUV.y * dotRepeat), cy);
            float dotMaskX = 1.0 - smoothstep(dotWidth, dotWidth + fwidth(gridUV.x * dotRepeat), cx);
            lineX *= dotMaskY;
            lineY *= dotMaskX;
          }
        }
        float primaryMask = max(lineX, lineY);

        vec2 gridUV2 = (hitIsY > 0.5 ? hit.xz : hit.zy) / gridScale;
        if (jitterAmt > 0.0) {
          vec2 j2 = vec2(
            cos(gridUV2.y * 2.1 - iTime * 1.4),
            sin(gridUV2.x * 2.5 + iTime * 1.7)
          ) * (0.15 * jitterAmt);
          gridUV2 += j2;
        }
        float fx2 = fract(gridUV2.x);
        float fy2 = fract(gridUV2.y);
        float ax2 = min(fx2, 1.0 - fx2);
        float ay2 = min(fy2, 1.0 - fy2);
        float wx2 = fwidth(gridUV2.x);
        float wy2 = fwidth(gridUV2.y);
        float tx2 = halfPx * wx2;
        float ty2 = halfPx * wy2;
        float aax2 = wx2;
        float aay2 = wy2;
        float lineX2 = 1.0 - smoothstep(tx2, tx2 + aax2, ax2);
        float lineY2 = 1.0 - smoothstep(ty2, ty2 + aay2, ay2);
        if (uLineStyle > 0.5) {
          float dashRepeat2 = 4.0;
          float dashDuty2 = 0.5;
          float vy2m = fract(gridUV2.y * dashRepeat2);
          float vx2m = fract(gridUV2.x * dashRepeat2);
          float dashMaskY2 = step(vy2m, dashDuty2);
          float dashMaskX2 = step(vx2m, dashDuty2);
          if (uLineStyle < 1.5) {
            lineX2 *= dashMaskY2;
            lineY2 *= dashMaskX2;
          } else {
            float dotRepeat2 = 6.0;
            float dotWidth2 = 0.18;
            float cy2 = abs(fract(gridUV2.y * dotRepeat2) - 0.5);
            float cx2 = abs(fract(gridUV2.x * dotRepeat2) - 0.5);
            float dotMaskY2 = 1.0 - smoothstep(dotWidth2, dotWidth2 + fwidth(gridUV2.y * dotRepeat2), cy2);
            float dotMaskX2 = 1.0 - smoothstep(dotWidth2, dotWidth2 + fwidth(gridUV2.x * dotRepeat2), cx2);
            lineX2 *= dotMaskY2;
            lineY2 *= dotMaskX2;
          }
        }
        float altMask = max(lineX2, lineY2);

        float edgeDistX = min(abs(hit.x - (-0.5)), abs(hit.x - 0.5));
        float edgeDistY = min(abs(hit.y - (-0.2)), abs(hit.y - 0.2));
        float edgeDist = mix(edgeDistY, edgeDistX, hitIsY);
        float edgeGate = 1.0 - smoothstep(gridScale * 0.5, gridScale * 2.0, edgeDist);
        altMask *= edgeGate;

        float lineMask = max(primaryMask, altMask);

        float fade = exp(-dist * fadeStrength);

        float dur = max(0.05, uScanDuration);
        float del = max(0.0, uScanDelay);
        float scanZMax = 2.0;
        float widthScale = max(0.1, uScanGlow);
        float sigma = max(0.001, 0.18 * widthScale * uScanSoftness);
        float sigmaA = sigma * 2.0;

        float combinedPulse = 0.0;
        float combinedAura = 0.0;

        float cycle = dur + del;
        float tCycle = mod(iTime, cycle);
        float scanPhase = clamp((tCycle - del) / dur, 0.0, 1.0);
        float phase = scanPhase;
        if (uScanDirection > 0.5 && uScanDirection < 1.5) {
          phase = 1.0 - phase;
        } else if (uScanDirection > 1.5) {
          float t2 = mod(max(0.0, iTime - del), 2.0 * dur);
          phase = (t2 < dur) ? (t2 / dur) : (1.0 - (t2 - dur) / dur);
        }
        float scanZ = phase * scanZMax;
        float dz = abs(hit.z - scanZ);
        float lineBand = exp(-0.5 * (dz * dz) / (sigma * sigma));
        float taper = clamp(uPhaseTaper, 0.0, 0.49);
        float headW = taper;
        float tailW = taper;
        float headFade = smoother01(0.0, headW, phase);
        float tailFade = 1.0 - smoother01(1.0 - tailW, 1.0, phase);
        float phaseWindow = headFade * tailFade;
        float pulseBase = lineBand * phaseWindow;
        combinedPulse += pulseBase * clamp(uScanOpacity, 0.0, 1.0);
        float auraBand = exp(-0.5 * (dz * dz) / (sigmaA * sigmaA));
        combinedAura += (auraBand * 0.25) * phaseWindow * clamp(uScanOpacity, 0.0, 1.0);

        for (int i = 0; i < MAX_SCANS; i++) {
          if (float(i) >= uScanCount) break;
          float tActiveI = iTime - uScanStarts[i];
          float phaseI = clamp(tActiveI / dur, 0.0, 1.0);
          if (uScanDirection > 0.5 && uScanDirection < 1.5) {
            phaseI = 1.0 - phaseI;
          } else if (uScanDirection > 1.5) {
            phaseI = (phaseI < 0.5) ? (phaseI * 2.0) : (1.0 - (phaseI - 0.5) * 2.0);
          }
          float scanZI = phaseI * scanZMax;
          float dzI = abs(hit.z - scanZI);
          float lineBandI = exp(-0.5 * (dzI * dzI) / (sigma * sigma));
          float headFadeI = smoother01(0.0, headW, phaseI);
          float tailFadeI = 1.0 - smoother01(1.0 - tailW, 1.0, phaseI);
          float phaseWindowI = headFadeI * tailFadeI;
          combinedPulse += lineBandI * phaseWindowI * clamp(uScanOpacity, 0.0, 1.0);
          float auraBandI = exp(-0.5 * (dzI * dzI) / (sigmaA * sigmaA));
          combinedAura += (auraBandI * 0.25) * phaseWindowI * clamp(uScanOpacity, 0.0, 1.0);
        }

        float lineVis = lineMask;
        vec3 gridCol = uLinesColor * lineVis * fade;
        vec3 scanCol = uScanColor * combinedPulse;
        vec3 scanAura = uScanColor * combinedAura;

        color = gridCol + scanCol + scanAura;

        float n = fract(sin(dot(gl_FragCoord.xy + vec2(iTime * 123.4), vec2(12.9898,78.233))) * 43758.5453123);
        color += (n - 0.5) * uNoise;
        color = clamp(color, 0.0, 1.0);
        float alpha = clamp(max(lineVis, combinedPulse), 0.0, 1.0);
        float gx = 1.0 - smoothstep(tx * 2.0, tx * 2.0 + aax * 2.0, ax);
        float gy = 1.0 - smoothstep(ty * 2.0, ty * 2.0 + aay * 2.0, ay);
        float halo = max(gx, gy) * fade;
        alpha = max(alpha, halo * clamp(uBloomOpacity, 0.0, 1.0));
        if (uLightMode > 0.5) {
          float energy = max(max(color.r, color.g), color.b);
          float coverage = clamp(max(alpha, smoothstep(0.0, 0.55, energy) * 0.82), 0.0, 0.9);
          coverage *= smoothstep(0.015, 0.12, energy);
          vec3 chroma = clamp(color / max(energy, 0.0001), 0.0, 1.0);
          chroma = pow(chroma, vec3(1.2));
          fragColor = vec4(mix(vec3(1.0), chroma, coverage * 0.94), 1.0);
        } else {
          fragColor = vec4(color, alpha);
        }
    }

    void main(){
      vec4 c;
      mainImage(c, vUv * iResolution.xy);
      gl_FragColor = c;
    }
  `;

  function hexToRGB(hex) {
    if (!hex) return [0.1, 0.1, 0.1];
    const h = hex.replace('#', '').trim();
    if (h.length === 3) {
      return [
        parseInt(h[0] + h[0], 16) / 255,
        parseInt(h[1] + h[1], 16) / 255,
        parseInt(h[2] + h[2], 16) / 255
      ];
    }
    return [
      parseInt(h.substring(0, 2), 16) / 255,
      parseInt(h.substring(2, 4), 16) / 255,
      parseInt(h.substring(4, 6), 16) / 255
    ];
  }

  function createShader(gl, type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`GridScan shader error: ${err}`);
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
      throw new Error(`GridScan program error: ${err}`);
    }
    return p;
  }

  class GridScan {
    constructor(container, options = {}) {
      if (!container) return;
      this.container = container;

      // Parse dataset or options
      const ds = container.dataset;
      this.lineThickness = options.lineThickness ?? (parseFloat(ds.lineThickness) || 0.95);
      this.linesColor = hexToRGB(options.linesColor || ds.linesColor || '#18181B');
      this.scanColor = hexToRGB(options.scanColor || ds.scanColor || '#FFFFFF');
      // Compact grid scale (default: 0.045 for dense, compact tactical grid)
      this.gridScale = options.gridScale ?? (parseFloat(ds.gridScale) || 0.045);
      this.scanOpacity = options.scanOpacity ?? (parseFloat(ds.scanOpacity) || 0.55);
      this.scanDuration = options.scanDuration ?? (parseFloat(ds.scanDuration) || 2.2);
      this.scanDelay = options.scanDelay ?? (parseFloat(ds.scanDelay) || 0.8);
      this.scanDirection = options.scanDirection === 'backward' ? 1 : options.scanDirection === 'forward' ? 0 : 2;
      this.lineStyle = options.lineStyle === 'dashed' ? 1 : options.lineStyle === 'dotted' ? 2 : 0;
      this.lineJitter = options.lineJitter ?? (parseFloat(ds.lineJitter) || 0.05);
      this.noise = options.noiseIntensity ?? (parseFloat(ds.noiseIntensity) || 0.015);
      this.bloomOpacity = options.bloomIntensity ?? (parseFloat(ds.bloomIntensity) || 0.35);
      this.scanGlow = options.scanGlow ?? (parseFloat(ds.scanGlow) || 0.5);
      this.scanSoftness = options.scanSoftness ?? (parseFloat(ds.scanSoftness) || 1.8);
      this.scanPhaseTaper = options.scanPhaseTaper ?? (parseFloat(ds.scanPhaseTaper) || 0.35);
      this.lightMode = options.lightMode ?? (ds.lightMode === 'true' ? 1 : 0);
      this.sensitivity = options.sensitivity ?? (parseFloat(ds.sensitivity) || 0.55);
      this.scanOnClick = options.scanOnClick ?? (ds.scanOnClick !== 'false');

      this.MAX_SCANS = 8;
      this.scanStarts = new Float32Array(8);
      this.scanCount = 0;

      // Mouse tracking state
      this.targetLook = [0, 0];
      this.currentLook = [0, 0];
      this.targetTilt = 0;
      this.currentTilt = 0;
      this.targetYaw = 0;
      this.currentYaw = 0;

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
      this.canvas.style.pointerEvents = 'none';
      this.container.appendChild(this.canvas);
    }

    initGL() {
      const gl = this.canvas.getContext('webgl', { antialias: true, alpha: true }) ||
                 this.canvas.getContext('experimental-webgl', { antialias: true, alpha: true });
      if (!gl) {
        console.warn('GridScan: WebGL not supported.');
        return;
      }
      this.gl = gl;

      // Enable derivatives for fwidth in WebGL 1
      gl.getExtension('OES_standard_derivatives');

      gl.clearColor(0, 0, 0, 0);

      this.program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);

      // Fullscreen Quad Geometry
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

      this.uvBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);

      // Cache Uniform Locations
      this.uniforms = {
        iResolution: gl.getUniformLocation(this.program, 'iResolution'),
        iTime: gl.getUniformLocation(this.program, 'iTime'),
        uSkew: gl.getUniformLocation(this.program, 'uSkew'),
        uTilt: gl.getUniformLocation(this.program, 'uTilt'),
        uYaw: gl.getUniformLocation(this.program, 'uYaw'),
        uLineThickness: gl.getUniformLocation(this.program, 'uLineThickness'),
        uLinesColor: gl.getUniformLocation(this.program, 'uLinesColor'),
        uScanColor: gl.getUniformLocation(this.program, 'uScanColor'),
        uGridScale: gl.getUniformLocation(this.program, 'uGridScale'),
        uLineStyle: gl.getUniformLocation(this.program, 'uLineStyle'),
        uLineJitter: gl.getUniformLocation(this.program, 'uLineJitter'),
        uScanOpacity: gl.getUniformLocation(this.program, 'uScanOpacity'),
        uScanDirection: gl.getUniformLocation(this.program, 'uScanDirection'),
        uNoise: gl.getUniformLocation(this.program, 'uNoise'),
        uBloomOpacity: gl.getUniformLocation(this.program, 'uBloomOpacity'),
        uScanGlow: gl.getUniformLocation(this.program, 'uScanGlow'),
        uScanSoftness: gl.getUniformLocation(this.program, 'uScanSoftness'),
        uPhaseTaper: gl.getUniformLocation(this.program, 'uPhaseTaper'),
        uScanDuration: gl.getUniformLocation(this.program, 'uScanDuration'),
        uScanDelay: gl.getUniformLocation(this.program, 'uScanDelay'),
        uLightMode: gl.getUniformLocation(this.program, 'uLightMode'),
        uScanStarts: gl.getUniformLocation(this.program, 'uScanStarts'),
        uScanCount: gl.getUniformLocation(this.program, 'uScanCount')
      };

      this.posAttr = gl.getAttribLocation(this.program, 'position');
      this.uvAttr = gl.getAttribLocation(this.program, 'uv');
    }

    pushScan(t) {
      if (this.scanCount >= this.MAX_SCANS) {
        for (let i = 0; i < this.MAX_SCANS - 1; i++) {
          this.scanStarts[i] = this.scanStarts[i + 1];
        }
        this.scanStarts[this.MAX_SCANS - 1] = t;
      } else {
        this.scanStarts[this.scanCount] = t;
        this.scanCount++;
      }
    }

    bindEvents() {
      // Global mouse tracking across the site
      this.onMouseMove = (e) => {
        const nx = (e.clientX / window.innerWidth) * 2 - 1;
        const ny = -((e.clientY / window.innerHeight) * 2 - 1);
        this.targetLook = [nx, ny];
        this.targetYaw = nx * 0.35 * this.sensitivity;
        this.targetTilt = ny * 0.25 * this.sensitivity;
      };

      this.onClick = () => {
        if (this.scanOnClick) {
          const nowSec = performance.now() / 1000;
          this.pushScan(nowSec);
        }
      };

      this.onResize = () => this.resize();

      window.addEventListener('mousemove', this.onMouseMove, { passive: true });
      window.addEventListener('click', this.onClick);
      window.addEventListener('resize', this.onResize);
    }

    resize() {
      if (!this.canvas || !this.gl) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;

      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    start() {
      const startTime = performance.now();
      let lastTime = startTime;

      const render = (now) => {
        const dt = Math.min(0.1, (now - lastTime) / 1000);
        lastTime = now;
        const t = (now - startTime) / 1000;

        // Smooth damping towards cursor
        const lerpFactor = Math.min(1, dt * 4.5);
        this.currentLook[0] += (this.targetLook[0] - this.currentLook[0]) * lerpFactor;
        this.currentLook[1] += (this.targetLook[1] - this.currentLook[1]) * lerpFactor;
        this.currentTilt += (this.targetTilt - this.currentTilt) * lerpFactor;
        this.currentYaw += (this.targetYaw - this.currentYaw) * lerpFactor;

        const gl = this.gl;
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.program);

        // Bind attributes
        gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
        gl.enableVertexAttribArray(this.posAttr);
        gl.vertexAttribPointer(this.posAttr, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
        gl.enableVertexAttribArray(this.uvAttr);
        gl.vertexAttribPointer(this.uvAttr, 2, gl.FLOAT, false, 0, 0);

        // Upload uniforms
        gl.uniform3f(this.uniforms.iResolution, this.canvas.width, this.canvas.height, 1.0);
        gl.uniform1f(this.uniforms.iTime, t);

        const skewScale = 0.12 * this.sensitivity;
        gl.uniform2f(this.uniforms.uSkew, this.currentLook[0] * skewScale, -this.currentLook[1] * skewScale * 1.4);
        gl.uniform1f(this.uniforms.uTilt, this.currentTilt);
        gl.uniform1f(this.uniforms.uYaw, this.currentYaw);

        gl.uniform1f(this.uniforms.uLineThickness, this.lineThickness);
        gl.uniform3fv(this.uniforms.uLinesColor, this.linesColor);
        gl.uniform3fv(this.uniforms.uScanColor, this.scanColor);
        gl.uniform1f(this.uniforms.uGridScale, this.gridScale);
        gl.uniform1f(this.uniforms.uLineStyle, this.lineStyle);
        gl.uniform1f(this.uniforms.uLineJitter, this.lineJitter);
        gl.uniform1f(this.uniforms.uScanOpacity, this.scanOpacity);
        gl.uniform1f(this.uniforms.uScanDirection, this.scanDirection);
        gl.uniform1f(this.uniforms.uNoise, this.noise);
        gl.uniform1f(this.uniforms.uBloomOpacity, this.bloomOpacity);
        gl.uniform1f(this.uniforms.uScanGlow, this.scanGlow);
        gl.uniform1f(this.uniforms.uScanSoftness, this.scanSoftness);
        gl.uniform1f(this.uniforms.uPhaseTaper, this.scanPhaseTaper);
        gl.uniform1f(this.uniforms.uScanDuration, this.scanDuration);
        gl.uniform1f(this.uniforms.uScanDelay, this.scanDelay);
        gl.uniform1f(this.uniforms.uLightMode, this.lightMode);

        gl.uniform1fv(this.uniforms.uScanStarts, this.scanStarts);
        gl.uniform1f(this.uniforms.uScanCount, this.scanCount);

        gl.drawArrays(gl.TRIANGLES, 0, 6);

        this.animFrame = requestAnimationFrame(render);
      };

      this.animFrame = requestAnimationFrame(render);
    }

    destroy() {
      if (this.animFrame) cancelAnimationFrame(this.animFrame);
      window.removeEventListener('mousemove', this.onMouseMove);
      window.removeEventListener('click', this.onClick);
      window.removeEventListener('resize', this.onResize);
      if (this.canvas && this.canvas.parentElement) {
        this.canvas.parentElement.removeChild(this.canvas);
      }
    }
  }

  // Auto-init on DOMContentLoaded or immediate
  function initAll() {
    const targets = document.querySelectorAll('[data-grid-scan], #site-gridscan-bg');
    targets.forEach((el) => {
      if (!el._gridScanInstance) {
        el._gridScanInstance = new GridScan(el);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

  window.GridScan = GridScan;
})();
