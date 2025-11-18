class PCMRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const channelData = input[0];
    if (!channelData) {
      return true;
    }

    const copy = channelData.slice();
    this.port.postMessage(
      {
        buffer: copy.buffer,
        sampleRate,
      },
      [copy.buffer]
    );

    return true;
  }
}

registerProcessor("pcm-recorder", PCMRecorderProcessor);
