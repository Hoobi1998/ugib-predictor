const radios = document.querySelectorAll('input[name="source"]');
const sampleSelect = document.querySelector('select[name="sample_index"]');
const fileInput = document.querySelector('input[name="ecg_file"]');
const form = document.querySelector("form");
const submitButton = document.querySelector(".primary-button");

function syncInputs() {
  const source = document.querySelector('input[name="source"]:checked')?.value;
  if (sampleSelect) sampleSelect.disabled = source !== "sample" || sampleSelect.options.length === 0;
  if (fileInput) fileInput.disabled = source !== "upload";
}

for (const radio of radios) {
  radio.addEventListener("change", syncInputs);
}
syncInputs();

if (form && submitButton) {
  form.addEventListener("submit", () => {
    submitButton.disabled = true;
    submitButton.textContent = submitButton.dataset.loadingText || "Loading...";
  });
}

const canvas = document.getElementById("signalChart");
if (canvas) {
  const signal = JSON.parse(canvas.dataset.signal || "[]");
  const context = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  context.scale(ratio, ratio);

  const width = rect.width;
  const height = rect.height;
  const pad = 22;

  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#d7dfdc";
  context.lineWidth = 1;

  for (let i = 0; i < 5; i += 1) {
    const y = pad + ((height - pad * 2) * i) / 4;
    context.beginPath();
    context.moveTo(pad, y);
    context.lineTo(width - pad, y);
    context.stroke();
  }

  if (signal.length > 1) {
    const min = Math.min(...signal);
    const max = Math.max(...signal);
    const span = max - min || 1;

    context.strokeStyle = "#0f766e";
    context.lineWidth = 2;
    context.beginPath();

    signal.forEach((value, index) => {
      const x = pad + ((width - pad * 2) * index) / (signal.length - 1);
      const y = height - pad - ((value - min) / span) * (height - pad * 2);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });

    context.stroke();
  }
}
