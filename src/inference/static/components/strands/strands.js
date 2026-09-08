// strands.js — Vanilla WebGL2 implementation of React Bits <Strands />
(function () {
  'use strict';

  const MAX_STRANDS = 12;
  const MAX_COLORS = 8;

  const VERT = `#version 300 es
in vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

  const FRAG = `#version 300 es
precision highp float;

uniform float uTime;
uniform vec2 uResolution;
uniform vec3 uColors[${MAX_COLORS}];
uniform int uColorCount;
uniform int uStrandCount;
uniform float uSpeed;
uniform float uAmplitude;
uniform float uWaviness;
uniform float uThickness;
uniform float uGlow;
uniform float uTaper;
uniform float uSpread;
uniform float uHueShift;
uniform float uIntensity;
uniform float uOpacity;
uniform float uScale;
uniform float uSaturation;

out vec4 fragColor;

const float PI = 3.14159265;

vec3 spectrum(float t) {
  return 0.5 + 0.5 * cos(2.0 * PI * (t + vec3(0.00, 0.33, 0.67)));
}

vec3 samplePalette(float t) {
  t = fract(t);
  float scaled = t * float(uColorCount);
  int idx = int(floor(scaled));
  float blend = fract(scaled);
  int nextIdx = idx + 1;
  if (nextIdx >= uColorCount) nextIdx = 0;
  return mix(uColors[idx], uColors[nextIdx], blend);
}

vec3 strandColor(float t) {
  if (uColorCount > 0) return samplePalette(t);
  return spectrum(t);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  uv /= max(uScale, 0.0001);

  float e = 0.06 + uIntensity * 0.94;
  float env = pow(max(cos(uv.x * PI * 1.3), 0.0), uTaper);

  vec3 col = vec3(0.0);

  for (int i = 0; i < ${MAX_STRANDS}; i++) {
    if (i >= uStrandCount) break;

    float fi = float(i);
    float ph = fi * 1.7 * uSpread;
    float freq = (2.0 + fi * 0.35) * uWaviness;
    float spd = 1.4 + fi * 1.2;

    float tt = uTime * uSpeed;
    float w = sin(uv.x * freq + tt * spd + ph) * 0.60
            + sin(uv.x * freq * 1.1 - tt * spd * 0.7 + ph * 1.7) * 0.40;

    float amp = (0.1 + 0.02 * e) * env * uAmplitude;
    float y = w * amp;

    float d = abs(uv.y - y);
    float thick = (0.001 + 0.05 * e) * (0.35 + env) * uThickness;
    float g = thick / (d + thick * 0.45);
    g = g * g;

    float h = fi / float(uStrandCount) + uv.x * 0.30 + uTime * 0.04 + uHueShift;
    col += strandColor(h) * g * env;
  }

  col *= 0.45 + 0.7 * e;
  col = 1.0 - exp(-col * uGlow);

  float gray = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col = max(mix(vec3(gray), col, uSaturation), 0.0);

  float lum = max(max(col.r, col.g), col.b);
  float alpha = clamp(lum, 0.0, 1.0) * uOpacity;

  fragColor = vec4(col * uOpacity, alpha);
}
`;

  const GLASS_FRAG = `#version 300 es
precision highp float;

uniform sampler2D uScene;
uniform vec2 uResolution;
uniform float uRadius;
uniform float uRefraction;
uniform float uDispersion;

out vec4 fragColor;

vec2 toUv(vec2 p) {
  return p * (uResolution.y / uResolution) + 0.5;
}

void main() {
  vec2 p = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  float d = length(p);
  float r = uRadius;

  float edge = fwidth(d) * 1.5;
  float mask = 1.0 - smoothstep(r - edge, r + edge, d);
  if (mask <= 0.0) {
    fragColor = vec4(0.0);
    return;
  }

  float z = sqrt(max(r * r - d * d, 0.0)) / r;
  float nd = d / r;

  vec2 dir = d > 0.0 ? p / d : vec2(0.0);
  float lens = smoothstep(0.85, 1.0, nd) * pow(nd, 6.0);
  vec2 offset = -dir * lens * uRefraction * 0.15;
  vec2 disp = -dir * lens * uDispersion * 0.012;

  vec3 light;
  light.r = texture(uScene, toUv(p + offset - disp)).r;
  light.g = texture(uScene, toUv(p + offset)).g;
  light.b = texture(uScene, toUv(p + offset + disp)).b;

  float fres = pow(1.0 - z, 3.0);
  vec3 rim = vec3(1.0) * fres * 0.18;

  vec2 lightDir = normalize(vec2(-0.55, 0.6));
  float spec = pow(max(dot(p / max(r, 1e-4), lightDir), 0.0), 6.0);
  spec *= smoothstep(r, r * 0.55, d);

  vec3 emissive = light + rim + vec3(spec) * 0.4;
  float emissiveA = clamp(max(max(emissive.r, emissive.g), emissive.b), 0.0, 1.0);

  float bodyA = 0.05 + fres * 0.05;
  float outA = emissiveA + bodyA * (1.0 - emissiveA);
  vec3 outRGB = emissive;

  outRGB *= mask;
  outA *= mask;

  fragColor = vec4(outRGB, outA);
}
`;

  function hexToRgb(hex) {
    let clean = hex.replace('#', '').trim();
    if (clean.length === 3) {
      clean = clean.split('').map(c => c + c).join('');
    }
    const num = parseInt(clean, 16);
    return [
      ((num >> 16) & 255) / 255,
      ((num >> 8) & 255) / 255,
      (num & 255) / 255
    ];
  }

  function buildPaletteFlat(colors) {
    const list = colors && colors.length ? colors : ['#ffffff'];
    const flat = [];
    for (let i = 0; i < MAX_COLORS; i++) {
      const hex = list[i] ?? list[list.length - 1];
      const [r, g, b] = hexToRgb(hex);
      flat.push(r, g, b);
    }
    return new Float32Array(flat);
  }

  function compileShader(gl, type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`Shader compile error: ${err}`);
    }
    return s;
  }

  function createProgram(gl, vertSrc, fragSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const err = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error(`Program link error: ${err}`);
    }
    return p;
  }

  class Strands {
    constructor(container, options = {}) {
      if (!container) {
        console.warn('Strands: Container element not found.');
        return;
      }

      this.container = container;

      // Props / dataset attributes
      this.colors = options.colors || (container.dataset.colors ? container.dataset.colors.split(',') : ['#00E676', '#FF6B00', '#00FF88', '#FFA726']);
      this.count = options.count ?? (parseInt(container.dataset.count, 10) || 3);
      this.speed = options.speed ?? (parseFloat(container.dataset.speed) || 0.45);
      this.amplitude = options.amplitude ?? (parseFloat(container.dataset.amplitude) || 0.9);
      this.waviness = options.waviness ?? (parseFloat(container.dataset.waviness) || 1.1);
      this.thickness = options.thickness ?? (parseFloat(container.dataset.thickness) || 0.7);
      this.glow = options.glow ?? (parseFloat(container.dataset.glow) || 2.5);
      this.taper = options.taper ?? (parseFloat(container.dataset.taper) || 2.8);
      this.spread = options.spread ?? (parseFloat(container.dataset.spread) || 1.0);
      this.hueShift = options.hueShift ?? (parseFloat(container.dataset.hueShift) || 0.0);
      this.intensity = options.intensity ?? (parseFloat(container.dataset.intensity) || 0.65);
      this.saturation = options.saturation ?? (parseFloat(container.dataset.saturation) || 1.4);
      this.opacity = options.opacity ?? (parseFloat(container.dataset.opacity) || 0.9);
      this.scale = options.scale ?? (parseFloat(container.dataset.scale) || 1.4);
      this.glass = options.glass ?? (container.dataset.glass === 'true');
      this.refraction = options.refraction ?? (parseFloat(container.dataset.refraction) || 1.0);
      this.dispersion = options.dispersion ?? (parseFloat(container.dataset.dispersion) || 1.0);
      this.glassSize = options.glassSize ?? (parseFloat(container.dataset.glassSize) || 1.0);

      this.canvas = document.createElement('canvas');
      this.canvas.style.display = 'block';
      this.canvas.style.width = '100%';
      this.canvas.style.height = '100%';
      this.container.appendChild(this.canvas);

      const gl = this.canvas.getContext('webgl2', {
        alpha: true,
        premultipliedAlpha: true,
        antialias: true
      });

      if (!gl) {
        console.warn('Strands: WebGL2 not supported on this device.');
        return;
      }
      this.gl = gl;

      gl.clearColor(0, 0, 0, 0);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      // Full-screen triangle geometry
      this.quadVao = gl.createVertexArray();
      gl.bindVertexArray(this.quadVao);
      const posBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1.0, -1.0,
         3.0, -1.0,
        -1.0,  3.0
      ]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);

      // Compile Strands Program
      this.program = createProgram(gl, VERT, FRAG);
      this.uniforms = this.getUniformLocations(gl, this.program, [
        'uTime', 'uResolution', 'uColors', 'uColorCount', 'uStrandCount',
        'uSpeed', 'uAmplitude', 'uWaviness', 'uThickness', 'uGlow',
        'uTaper', 'uSpread', 'uHueShift', 'uIntensity', 'uOpacity',
        'uScale', 'uSaturation'
      ]);

      // Glass Pass (if requested)
      if (this.glass) {
        this.initGlassPass();
      }

      this.bindEvents();
      this.resize();
      this.start();
    }

    getUniformLocations(gl, prog, names) {
      const locs = {};
      names.forEach(n => {
        locs[n] = gl.getUniformLocation(prog, n);
      });
      return locs;
    }

    initGlassPass() {
      const gl = this.gl;
      this.glassProgram = createProgram(gl, VERT, GLASS_FRAG);
      this.glassUniforms = this.getUniformLocations(gl, this.glassProgram, [
        'uScene', 'uResolution', 'uRadius', 'uRefraction', 'uDispersion'
      ]);

      this.fbo = gl.createFramebuffer();
      this.fboTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.fboTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTexture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    resize() {
      if (!this.container || !this.gl) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(10, Math.floor(this.container.clientWidth));
      const h = Math.max(10, Math.floor(this.container.clientHeight));

      this.width = Math.round(w * dpr);
      this.height = Math.round(h * dpr);

      this.canvas.width = this.width;
      this.canvas.height = this.height;
      this.gl.viewport(0, 0, this.width, this.height);

      if (this.glass && this.fboTexture) {
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.fboTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      }
    }

    bindEvents() {
      this.resizeObserver = new ResizeObserver(() => {
        this.resize();
      });
      this.resizeObserver.observe(this.container);
    }

    start() {
      const gl = this.gl;
      const palette = buildPaletteFlat(this.colors);

      const render = (time) => {
        this.animFrame = requestAnimationFrame(render);
        const t = time * 0.001;

        if (this.glass && this.fbo) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
          gl.clear(gl.COLOR_BUFFER_BIT);
        } else {
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          gl.clear(gl.COLOR_BUFFER_BIT);
        }

        // Render Strands
        gl.useProgram(this.program);
        gl.uniform1f(this.uniforms.uTime, t);
        gl.uniform2f(this.uniforms.uResolution, this.width, this.height);
        gl.uniform3fv(this.uniforms.uColors, palette);
        gl.uniform1i(this.uniforms.uColorCount, Math.min(this.colors.length, MAX_COLORS));
        gl.uniform1i(this.uniforms.uStrandCount, Math.min(Math.max(Math.round(this.count), 1), MAX_STRANDS));
        gl.uniform1f(this.uniforms.uSpeed, this.speed);
        gl.uniform1f(this.uniforms.uAmplitude, this.amplitude);
        gl.uniform1f(this.uniforms.uWaviness, this.waviness);
        gl.uniform1f(this.uniforms.uThickness, this.thickness);
        gl.uniform1f(this.uniforms.uGlow, this.glow);
        gl.uniform1f(this.uniforms.uTaper, this.taper);
        gl.uniform1f(this.uniforms.uSpread, this.spread);
        gl.uniform1f(this.uniforms.uHueShift, this.hueShift);
        gl.uniform1f(this.uniforms.uIntensity, this.intensity);
        gl.uniform1f(this.uniforms.uOpacity, this.opacity);
        gl.uniform1f(this.uniforms.uScale, this.scale);
        gl.uniform1f(this.uniforms.uSaturation, this.saturation);

        gl.bindVertexArray(this.quadVao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        // Glass Ball pass
        if (this.glass && this.fbo) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          gl.clear(gl.COLOR_BUFFER_BIT);

          gl.useProgram(this.glassProgram);
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, this.fboTexture);
          gl.uniform1i(this.glassUniforms.uScene, 0);
          gl.uniform2f(this.glassUniforms.uResolution, this.width, this.height);
          gl.uniform1f(this.glassUniforms.uRadius, 0.46 * this.glassSize);
          gl.uniform1f(this.glassUniforms.uRefraction, this.refraction);
          gl.uniform1f(this.glassUniforms.uDispersion, this.dispersion);

          gl.drawArrays(gl.TRIANGLES, 0, 3);
        }
      };

      this.animFrame = requestAnimationFrame(render);
    }

    destroy() {
      if (this.animFrame) cancelAnimationFrame(this.animFrame);
      if (this.resizeObserver) this.resizeObserver.disconnect();
      if (this.canvas && this.canvas.parentNode) {
        this.canvas.parentNode.removeChild(this.canvas);
      }
      this.gl?.getExtension('WEBGL_lose_context')?.loseContext();
    }
  }

  // Auto-initialize on [data-strands] elements
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-strands]').forEach((el) => {
      new Strands(el);
    });
  });

  window.Strands = Strands;
})();
