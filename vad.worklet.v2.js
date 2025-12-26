class VadHelperWorklet extends AudioWorkletProcessor {
  constructor() { super(); }
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0 && input[0]) {
      this.port.postMessage(input[0]);
    }
    return true;
  }
}
registerProcessor("vad-helper-worklet", VadHelperWorklet);
