// Socket.IO接続
const socket = io({ transports: ["websocket", "polling"] });

// DOM要素
const tabRealtime = document.getElementById("tab-realtime");
const tabFile = document.getElementById("tab-file");
const contentRealtime = document.getElementById("content-realtime");
const contentFile = document.getElementById("content-file");

const startRecordingBtn = document.getElementById("start-recording");
const stopRecordingBtn = document.getElementById("stop-recording");
const realtimeLanguage = document.getElementById("realtime-language");
const realtimeStatus = document.getElementById("realtime-status");
const realtimeStatusText = document.getElementById("realtime-status-text");
const realtimeDeviceHint = document.getElementById("realtime-device-hint");
const realtimeTranscript = document.getElementById("realtime-transcript");
const clearRealtimeBtn = document.getElementById("clear-realtime");
const preciseTranscript = document.getElementById("precise-transcript");

const fileUpload = document.getElementById("file-upload");
const fileName = document.getElementById("file-name");
const uploadBtn = document.getElementById("upload-btn");
const fileLanguage = document.getElementById("file-language");
const fileBeamSize = document.getElementById("file-beam-size");
const fileProgress = document.getElementById("file-progress");
const fileResult = document.getElementById("file-result");
const fileTranscript = document.getElementById("file-transcript");
const fileError = document.getElementById("file-error");
const fileErrorMessage = document.getElementById("file-error-message");
const resultInfo = document.getElementById("result-info");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");

// グローバル状態
let audioStream;
let audioContext;
let workletNode;
let workletSilenceNode;
let realtimeSessionId = null;
let activeRealtimeSessionId = null;
let realtimeChunkIndex = 0;
let isRealtimeActive = false;
let currentFile = null;
let lastResult = null;
let realtimeSegmentsLog = [];
let preciseSegmentsLog = [];

const PCM_WORKLET_URL = "/static/pcm-recorder.worklet.js";
const PCM_TARGET_SAMPLE_RATE = 16000;
const STREAM_CHUNK_INTERVAL_MS = 800; // 0.8s chunks for low latency

let pendingSampleRate = 0;
let pendingSamples = [];
let pendingSampleCount = 0;

// タブ切り替え
tabRealtime?.addEventListener("click", () => switchTab("realtime"));
tabFile?.addEventListener("click", () => switchTab("file"));

function switchTab(tab) {
  if (tab === "realtime") {
    tabRealtime.className =
      "tab-active whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium transition-colors";
    tabFile.className =
      "tab-inactive whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium transition-colors";
    contentRealtime.classList.remove("hidden");
    contentFile.classList.add("hidden");
  } else {
    tabFile.className =
      "tab-active whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium transition-colors";
    tabRealtime.className =
      "tab-inactive whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium transition-colors";
    contentFile.classList.remove("hidden");
    contentRealtime.classList.add("hidden");
  }
}

// リアルタイム録音
startRecordingBtn?.addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast("ブラウザがマイク録音に対応していません", "error");
    return;
  }
  if (isRealtimeActive) return;

  try {
    resetRealtimePipeline();
    realtimeSessionId = crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}`;
    activeRealtimeSessionId = realtimeSessionId;
    realtimeTranscript.textContent = "";

    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule(PCM_WORKLET_URL);
    const source = audioContext.createMediaStreamSource(audioStream);
    workletNode = new AudioWorkletNode(audioContext, "pcm-recorder");
    workletNode.port.onmessage = handleWorkletChunk;
    workletSilenceNode = audioContext.createGain();
    workletSilenceNode.gain.value = 0;
    source.connect(workletNode);
    workletNode.connect(workletSilenceNode).connect(audioContext.destination);
    await audioContext.resume();

    isRealtimeActive = true;
    startRecordingBtn.classList.add("hidden");
    stopRecordingBtn.classList.remove("hidden");
    realtimeStatus.classList.remove("hidden");
    setRealtimeStatus("録音中...", "live");
  } catch (error) {
    toast(`マイクの初期化に失敗しました: ${error.message}`, "error");
    await teardownRealtimeStream();
  }
});

stopRecordingBtn?.addEventListener("click", async () => {
  if (!isRealtimeActive) return;
  await flushRealtimeChunks(true);
  await teardownRealtimeStream();
  stopRecordingBtn.classList.add("hidden");
  startRecordingBtn.classList.remove("hidden");
  setRealtimeStatus("最終チャンクを解析中...", "busy");
});

clearRealtimeBtn?.addEventListener("click", () => {
  realtimeSegmentsLog = [];
  preciseSegmentsLog = [];
  realtimeTranscript.textContent = "";
  if (preciseTranscript) {
    preciseTranscript.textContent = "";
  }
});

// Socket.IOイベント
socket.on("connect", () => {
  toast("WebSocket接続完了", "info");
});

socket.on("status", (data) => {
  toast(data.message, "info");
});

socket.on("transcribe_segment", (data) => {
  if (
    data.sessionId &&
    activeRealtimeSessionId &&
    data.sessionId !== activeRealtimeSessionId
  ) {
    return;
  }

  if (data.type === "info") {
    const probability = (data.language_probability * 100).toFixed(1);
    toast(`言語: ${data.language} / ${probability}%`);
    return;
  }
});

socket.on("transcribe_snapshot", (data = {}) => {
  if (
    data.sessionId &&
    activeRealtimeSessionId &&
    data.sessionId !== activeRealtimeSessionId
  ) {
    return;
  }

  const segments = data.segments || [];
  realtimeSegmentsLog = segments.map((segment) => ({
    start: segment.start,
    end: segment.end,
    text: segment.text,
  }));
  renderSegments(realtimeTranscript, realtimeSegmentsLog);
});

socket.on("transcribe_complete", (data = {}) => {
  if (
    data.sessionId &&
    activeRealtimeSessionId &&
    data.sessionId !== activeRealtimeSessionId
  ) {
    return;
  }
  setRealtimeStatus("解析完了", "idle");
});

socket.on("transcribe_precise", (data = {}) => {
  if (
    data.sessionId &&
    activeRealtimeSessionId &&
    data.sessionId !== activeRealtimeSessionId
  ) {
    return;
  }
  const segments = data.segments || [];
  if (!preciseTranscript || !segments.length) return;

  if (data.isFull) {
    preciseSegmentsLog = segments.map((segment) => ({
      start: segment.start,
      end: segment.end,
      text: segment.text,
    }));
  } else {
    const windowStart = data.windowStart ?? segments[0].start;
    const windowEnd = data.windowEnd ?? segments[segments.length - 1].end;
    preciseSegmentsLog = preciseSegmentsLog.filter(
      (segment) =>
        segment.end < windowStart - 0.05 || segment.start > windowEnd + 0.05
    );
    segments.forEach((segment) => {
      preciseSegmentsLog.push({
        start: segment.start,
        end: segment.end,
        text: segment.text,
      });
    });
    preciseSegmentsLog.sort((a, b) => a.start - b.start);
  }

  renderSegments(preciseTranscript, preciseSegmentsLog);
});

socket.on("error", (data = {}) => {
  toast(data.message || "サーバエラーが発生しました", "error");
  setRealtimeStatus("エラーが発生しました", "error");
  stopRecordingBtn.classList.add("hidden");
  startRecordingBtn.classList.remove("hidden");
});

// ファイルアップロード
fileUpload?.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    currentFile = file;
    fileName.textContent = `選択: ${file.name} (${formatFileSize(file.size)})`;
    uploadBtn.disabled = false;
    fileResult.classList.add("hidden");
    fileError.classList.add("hidden");
  }
});

uploadBtn?.addEventListener("click", async () => {
  if (!currentFile) return;

  const formData = new FormData();
  formData.append("file", currentFile);
  formData.append("language", fileLanguage.value);
  formData.append("beam_size", fileBeamSize.value);

  uploadBtn.disabled = true;
  fileProgress.classList.remove("hidden");
  fileResult.classList.add("hidden");
  fileError.classList.add("hidden");

  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "エラーが発生しました");
    }

    lastResult = data;
    displayResult(data);
  } catch (error) {
    fileError.classList.remove("hidden");
    fileErrorMessage.textContent = error.message;
  } finally {
    fileProgress.classList.add("hidden");
    uploadBtn.disabled = false;
  }
});

function displayResult(data) {
  fileResult.classList.remove("hidden");
  const duration = formatTime(data.duration);
  const processingTime = data.processing_time.toFixed(2);
  const rtf = (data.processing_time / data.duration).toFixed(2);
  resultInfo.textContent = `時間: ${duration} | 処理時間: ${processingTime}秒 | RTF: ${rtf}x | 言語: ${
    data.language
  } (${(data.language_probability * 100).toFixed(1)}%)`;

  let transcriptHTML = "";
  data.segments.forEach((segment) => {
    const time = `<span class="text-gray-500 text-xs">[${formatTime(
      segment.start
    )} - ${formatTime(segment.end)}]</span>`;
    transcriptHTML += `<div class="mb-3"><div>${time}</div><div class="mt-1">${escapeHtml(
      segment.text
    )}</div></div>`;
  });
  fileTranscript.innerHTML = transcriptHTML;
}

copyBtn?.addEventListener("click", () => {
  if (!lastResult) return;
  const text = lastResult.segments.map((s) => s.text).join("\n");
  navigator.clipboard.writeText(text).then(() => {
    const originalText = copyBtn.textContent;
    copyBtn.textContent = "コピーしました！";
    setTimeout(() => {
      copyBtn.textContent = originalText;
    }, 2000);
  });
});

downloadBtn?.addEventListener("click", () => {
  if (!lastResult) return;
  const dataStr = JSON.stringify(lastResult, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `transcript_${new Date()
    .toISOString()
    .replace(/:/g, "-")}.json`;
  link.click();
  URL.revokeObjectURL(url);
});

function handleWorkletChunk(event) {
  if (!isRealtimeActive) return;
  const { buffer, sampleRate } = event.data || {};
  if (!buffer) return;
  const floatChunk = new Float32Array(buffer);
  enqueueFloat32Samples(floatChunk, sampleRate || PCM_TARGET_SAMPLE_RATE);
}

function enqueueFloat32Samples(floatChunk, sourceRate) {
  if (!floatChunk?.length) return;
  pendingSampleRate = sourceRate || pendingSampleRate || PCM_TARGET_SAMPLE_RATE;
  pendingSamples.push({ data: floatChunk, offset: 0 });
  pendingSampleCount += floatChunk.length;
  const threshold = Math.max(
    1024,
    Math.floor((pendingSampleRate * STREAM_CHUNK_INTERVAL_MS) / 1000)
  );

  while (pendingSampleCount >= threshold) {
    const samples = extractSamples(threshold);
    pushFloat32Samples(samples, false);
  }
}

function extractSamples(sampleCount) {
  const output = new Float32Array(sampleCount);
  let written = 0;

  while (written < sampleCount && pendingSamples.length) {
    const head = pendingSamples[0];
    const available = head.data.length - head.offset;
    const take = Math.min(sampleCount - written, available);
    output.set(head.data.subarray(head.offset, head.offset + take), written);
    head.offset += take;
    written += take;
    pendingSampleCount -= take;

    if (head.offset >= head.data.length) {
      pendingSamples.shift();
    }
  }

  return output;
}

function pushFloat32Samples(floatSamples, isFinalChunk) {
  if (!floatSamples.length || !pendingSampleRate) return;
  const downsampled = downsampleBuffer(
    floatSamples,
    pendingSampleRate,
    PCM_TARGET_SAMPLE_RATE
  );
  if (!downsampled.length) return;
  const pcmBuffer = float32ToPCM16(downsampled);
  sendRealtimeChunk(pcmBuffer, realtimeChunkIndex++, isFinalChunk);
}

function downsampleBuffer(input, sourceRate, targetRate) {
  if (!input.length || sourceRate === targetRate) {
    return input;
  }

  const ratio = sourceRate / targetRate;
  const newLength = Math.max(1, Math.floor(input.length / ratio));
  const result = new Float32Array(newLength);

  for (let i = 0; i < newLength; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    const span = Math.max(1, end - start);
    for (let j = start; j < end; j += 1) {
      sum += input[j];
    }
    result[i] = sum / span;
  }

  return result;
}

function float32ToPCM16(floatArray) {
  const buffer = new ArrayBuffer(floatArray.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < floatArray.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, floatArray[i]));
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

function arrayBufferToBase64(buffer) {
  if (!buffer || buffer.byteLength === 0) return "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function resetRealtimePipeline() {
  pendingSamples = [];
  pendingSampleCount = 0;
  pendingSampleRate = 0;
  realtimeChunkIndex = 0;
}

async function flushRealtimeChunks(markFinal = false) {
  if (pendingSampleCount > 0 && pendingSampleRate) {
    const remaining = extractSamples(pendingSampleCount);
    pushFloat32Samples(remaining, markFinal);
  } else if (markFinal && realtimeChunkIndex === 0) {
    // 送信データが無かった場合でもセッション終了を通知
    sendRealtimeChunk(null, realtimeChunkIndex++, true);
  } else if (markFinal) {
    sendRealtimeChunk(null, realtimeChunkIndex++, true);
  }
}

async function teardownRealtimeStream() {
  isRealtimeActive = false;
  if (workletNode) {
    try {
      workletNode.port.onmessage = null;
      workletNode.disconnect();
    } catch (error) {
      console.warn("Failed to disconnect worklet", error);
    }
    workletNode = null;
  }
  if (workletSilenceNode) {
    try {
      workletSilenceNode.disconnect();
    } catch (error) {
      console.warn("Failed to disconnect gain node", error);
    }
    workletSilenceNode = null;
  }
  if (audioContext) {
    try {
      await audioContext.close();
    } catch (error) {
      console.warn("Failed to close audio context", error);
    }
    audioContext = null;
  }
  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
  }
  resetRealtimePipeline();
}

function sendRealtimeChunk(pcmBuffer, chunkNumber, isFinalChunk) {
  if (!activeRealtimeSessionId) return;
  try {
    console.log("sendRealtimeChunk", {
      sessionId: activeRealtimeSessionId,
      chunk: chunkNumber,
      isFinal: isFinalChunk,
      bytes: pcmBuffer ? pcmBuffer.byteLength : 0,
    });
    const base64Audio = pcmBuffer ? arrayBufferToBase64(pcmBuffer) : null;
    socket.emit("realtime_chunk", {
      sessionId: activeRealtimeSessionId,
      audio: base64Audio,
      language: realtimeLanguage.value,
      sampleRate: PCM_TARGET_SAMPLE_RATE,
      chunkIndex: chunkNumber,
      isFinal: isFinalChunk,
    });
    setRealtimeStatus(
      isFinalChunk ? "最終チャンク送信済み" : "送信中...",
      isFinalChunk ? "busy" : "live"
    );
  } catch (error) {
    toast(`音声送信に失敗しました: ${error.message}`, "error");
  }
}

function renderSegments(target, segments) {
  if (!target) return;
  const content = segments
    .map((segment) => {
      return `[${formatTime(segment.start)} - ${formatTime(segment.end)}]\n${
        segment.text
      }`;
    })
    .join("\n\n");
  target.textContent = content;
  target.scrollTop = target.scrollHeight;
}

function setRealtimeStatus(text, state = "idle") {
  realtimeStatus.classList.remove("hidden");
  realtimeStatusText.textContent = text;
  const colors = {
    idle: "bg-slate-500",
    live: "bg-emerald-400",
    busy: "bg-sky-400",
    error: "bg-rose-400",
  };
  const indicator = realtimeStatus.querySelector("[data-indicator]");
  if (indicator) {
    indicator.className = `h-3 w-3 rounded-full ${
      colors[state] || colors.idle
    }`;
  }
}

function toast(message, variant = "info") {
  if (!realtimeDeviceHint) return;
  realtimeDeviceHint.textContent = message;
  if (variant === "error") {
    realtimeDeviceHint.classList.add("text-rose-300");
  } else {
    realtimeDeviceHint.classList.remove("text-rose-300");
  }
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.000";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 1000);
  return `${mins.toString().padStart(2, "0")}:${secs
    .toString()
    .padStart(2, "0")}.${ms.toString().padStart(3, "0")}`;
}

function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
