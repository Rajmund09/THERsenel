/**
 * Tactical Web Audio Alarm Synthesizer (tactical-audio.js)
 * Synthesizes warning siren pulses without external audio files
 * using the HTML5 Web Audio API (AudioContext + OscillatorNode).
 */

class TacticalAudioSiren {
    constructor() {
        this.ctx = null;
        this.isMuted = localStorage.getItem('thersenel_alarm_muted') === 'true';
        this.isPlaying = false;
        this.activeOscillator = null;
        this.gainNode = null;
    }

    initContext() {
        if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                this.ctx = new AudioCtx();
            }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        localStorage.setItem('thersenel_alarm_muted', this.isMuted.toString());
        if (this.isMuted) {
            this.stopAlert();
        }
        return this.isMuted;
    }

    playIntrusionAlert(durationSec = 1.6) {
        if (this.isMuted) return;

        try {
            this.initContext();
            if (!this.ctx) return;

            const now = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();

            osc.type = 'sawtooth';
            // Alternating two-tone defense perimeter pulse (880Hz / 440Hz)
            osc.frequency.setValueAtTime(880, now);
            osc.frequency.setValueAtTime(440, now + 0.25);
            osc.frequency.setValueAtTime(880, now + 0.50);
            osc.frequency.setValueAtTime(440, now + 0.75);
            osc.frequency.setValueAtTime(880, now + 1.00);

            // Smooth envelope attack and decay to prevent audio pops
            gain.gain.setValueAtTime(0.001, now);
            gain.gain.exponentialRampToValueAtTime(0.25, now + 0.05);
            gain.gain.setValueAtTime(0.25, now + durationSec - 0.1);
            gain.gain.exponentialRampToValueAtTime(0.001, now + durationSec);

            osc.connect(gain);
            gain.connect(this.ctx.destination);

            osc.start(now);
            osc.stop(now + durationSec);
        } catch (e) {
            console.warn("Tactical Web Audio playback throttled or unsupported:", e);
        }
    }

    stopAlert() {
        if (this.activeOscillator) {
            try {
                this.activeOscillator.stop();
            } catch (_) {}
            this.activeOscillator = null;
        }
    }
}

window.TacticalAudioSiren = TacticalAudioSiren;
