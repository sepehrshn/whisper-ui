class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.nextIndex = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    const ratio = sampleRate / 16000;
    let index = this.nextIndex;
    const pcm = [];
    while (index < channel.length) {
      const left = Math.floor(index);
      const right = Math.min(left + 1, channel.length - 1);
      const fraction = index - left;
      const mixed = channel[left] * (1 - fraction) + channel[right] * fraction;
      const clamped = Math.max(-1, Math.min(1, mixed));
      const scaled = Math.round(clamped * 32768);
      pcm.push(Math.max(-32768, Math.min(32767, scaled)));
      index += ratio;
    }
    this.nextIndex = index - channel.length;

    if (pcm.length > 0) {
      const frame = new Int16Array(pcm);
      this.port.postMessage(frame.buffer, [frame.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
