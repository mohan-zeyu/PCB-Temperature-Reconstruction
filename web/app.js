/*
 * Thermal Studio frontend.
 *
 * Server mode is the real product: boards come from /api/boards, inference is
 * a POST of the selected CSV to /api/infer, and training is a background job
 * polled through /api/jobs/<id>.  Demo mode is a read-only fallback for a plain
 * static server: it can display the bundled predictions but says so, instead of
 * pretending to train or infer.
 */
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const DEFAULT_STAGES = [
  { key: 'read', label: '读取上传数据' },
  { key: 'window', label: '定位板体' },
  { key: 'rectify', label: '配准到统一网格' },
  { key: 'mask', label: '识别不可信区域' },
  { key: 'pair', label: '按热状态配对' },
  { key: 'train', label: '训练校正模型' },
  { key: 'evaluate', label: '评估训练误差' },
  { key: 'save', label: '保存板型' }
];

let mode = 'server';
let serverBoards = [];
let activeBoardId = null;
let inferenceFile = null;
let parsedInput = null;
let resultFrame = null;
let resultView = null;
let resultRawView = null;
let demoMatch = null;
let demoReady = false;
let trainingToken = 0;
let trainingStartedAt = 0;
let pollTimer = null;
let archive = {};
let demoBoards = [];
const previewCache = {};

const heatStops = [
  [0.00, [9, 21, 38]], [0.16, [44, 26, 78]], [0.34, [139, 31, 89]],
  [0.53, [218, 66, 52]], [0.73, [249, 147, 59]], [0.89, [252, 211, 100]], [1.00, [252, 248, 187]]
];

function colorFor(value) {
  const t = Math.max(0, Math.min(1, (value - 25) / 65));
  let right = heatStops.findIndex(stop => stop[0] >= t);
  if (right <= 0) right = 1;
  const [a, ac] = heatStops[right - 1];
  const [b, bc] = heatStops[right];
  const amount = (t - a) / (b - a);
  return [0, 1, 2].map(index => Math.round(ac[index] + (bc[index] - ac[index]) * amount));
}

function unpackMask(encoded, count) {
  if (!encoded) return Array(count).fill(true);
  const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
  return Array.from({ length: count }, (_, index) => !!(bytes[index >> 3] & (128 >> (index & 7))));
}

function decodeField(encoded) {
  const count = encoded.width * encoded.height;
  return {
    width: encoded.width,
    height: encoded.height,
    ambient: encoded.ambient_c,
    values: encoded.values,
    display: encoded.display,
    boardMask: unpackMask(encoded.board_mask, count)
  };
}

function drawMap(canvas, values, view, options = {}) {
  const filled = options.filled === true;
  const width = view.width, height = view.height;
  const small = document.createElement('canvas');
  small.width = width;
  small.height = height;
  const context = small.getContext('2d');
  const image = context.createImageData(width, height);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const source = row * width + col;
      // Source orientation: row 0 is the top of the camera frame.  The old
      // canvas mirrored both axes, which put the data bottom (the port edge on
      // case10) at the display top and made the board look upside down.
      const x = col;
      const y = row;
      const destination = (y * width + x) * 4;
      const value = values[source];
      const invalid = value === -32768 || !Number.isFinite(value);
      if (invalid || (!filled && !view.boardMask[source])) {
        image.data[destination] = 16;
        image.data[destination + 1] = 28;
        image.data[destination + 2] = 34;
      } else {
        const color = colorFor(value / 10 + view.ambient);
        image.data[destination] = color[0];
        image.data[destination + 1] = color[1];
        image.data[destination + 2] = color[2];
      }
      image.data[destination + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  canvas.width = width * 3;
  canvas.height = height * 3;
  const target = canvas.getContext('2d');
  target.imageSmoothingEnabled = true;
  target.drawImage(small, 0, 0, canvas.width, canvas.height);
  const frameElement = canvas.closest('.canvas-frame');
  if (frameElement) frameElement.style.aspectRatio = `${width} / ${height}`;
}

function clearCanvas(canvas, width = 103, height = 157) {
  canvas.width = width * 3;
  canvas.height = height * 3;
  const context = canvas.getContext('2d');
  context.fillStyle = '#101d23';
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function fieldRange(values, view) {
  let low = Infinity, high = -Infinity;
  values.forEach((value, index) => {
    if (!view.boardMask[index] || value === -32768) return;
    const absolute = value / 10 + view.ambient;
    low = Math.min(low, absolute);
    high = Math.max(high, absolute);
  });
  if (!Number.isFinite(low)) return '—';
  return `${low.toFixed(1)}–${high.toFixed(1)} °C`;
}

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */
async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) throw new Error((payload && payload.error) || `请求失败（HTTP ${response.status}）`);
  return payload;
}

function postForm(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', url);
    request.upload.onprogress = event => {
      if (onProgress && event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () => {
      let payload = null;
      try { payload = JSON.parse(request.responseText || '{}'); } catch { payload = null; }
      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error((payload && payload.error) || `请求失败（HTTP ${request.status}）`));
    };
    request.onerror = () => reject(new Error('无法连接后端服务'));
    request.send(formData);
  });
}

/* ------------------------------------------------------------------ */
/* CSV                                                                 */
/* ------------------------------------------------------------------ */
function parseHmCsv(buffer, name) {
  let decoded;
  try { decoded = new TextDecoder('utf-8', { fatal: true }).decode(buffer); }
  catch { decoded = new TextDecoder('gb18030').decode(buffer); }
  const lines = decoded.replace(/^\uFEFF/, '').split(/\r?\n/);
  let header = -1, width = 0;
  for (let index = 0; index < lines.length; index++) {
    const cells = lines[index].split(',');
    if (cells.length >= 4 && cells.slice(1, 4).map(cell => cell.trim()).join(',') === '0,1,2') {
      while (cells.length && !cells[cells.length - 1].trim()) cells.pop();
      width = cells.length - 1;
      header = index;
      break;
    }
  }
  if (header < 0 || width < 4) throw new Error('未找到温度矩阵');
  const values = [];
  let height = 0, low = Infinity, high = -Infinity;
  for (let row = 0; row < lines.length - header - 1; row++) {
    const cells = (lines[header + row + 1] || '').split(',');
    while (cells.length && !cells[cells.length - 1].trim()) cells.pop();
    if (cells.length < width + 1) break;
    if (Number(cells[0]) !== row) throw new Error('温度矩阵行号不连续');
    for (let col = 0; col < width; col++) {
      const value = Number(cells[col + 1]);
      if (!Number.isFinite(value)) throw new Error('温度矩阵包含无效数值');
      low = Math.min(low, value);
      high = Math.max(high, value);
      values.push(Math.round(value * 10));
    }
    height++;
  }
  if (height < 2) throw new Error('温度矩阵尺寸不正确');
  return {
    name, width, height, values, ambient: 0,
    boardMask: Array(values.length).fill(true),
    range: `${low.toFixed(1)}–${high.toFixed(1)} °C`
  };
}

async function fileDigest(buffer) {
  const bytes = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(bytes)].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

/* ------------------------------------------------------------------ */
/* boards                                                              */
/* ------------------------------------------------------------------ */
function activeBoard() {
  return (mode === 'server' ? serverBoards : demoBoards).find(board => board.id === activeBoardId) || null;
}

function boardMetric(board) {
  const metrics = board && board.metrics;
  if (!metrics) return null;
  return metrics.final || metrics.best || null;
}

function updateBoardPicker() {
  const select = $('#board-select');
  select.replaceChildren();
  const boards = mode === 'server' ? serverBoards : demoBoards;
  boards.forEach(board => {
    const option = document.createElement('option');
    option.value = board.id;
    const grid = board.grid_hw ? `${board.grid_hw[1]}×${board.grid_hw[0]}` : '';
    option.textContent = grid ? `${board.name} · ${grid}` : board.name;
    select.append(option);
  });
  if (activeBoardId) select.value = activeBoardId;
  $('#topbar-count').textContent = `${boards.length} 个板型可用`;
  $('#library-count').textContent = `${boards.length} 个板型`;
}

function updateBoardDetails() {
  const board = activeBoard();
  if (!board) {
    $('#board-details').textContent = '还没有可用板型。请先在“训练新板”中导入数据。';
    return;
  }
  const metric = boardMetric(board);
  const mae = metric && Number.isFinite(metric.board_mae_k) ? `${metric.board_mae_k.toFixed(3)} K` : '—';
  const grid = board.grid_hw ? `${board.grid_hw[1]} × ${board.grid_hw[0]} px` : '';
  const pairs = board.pair_count ? `${board.pair_count} 组配对` : '';
  $('#board-details').innerHTML = `<span>${pairs || '已训练板型'}</span><strong>${grid}</strong><small>训练集板体误差 ${mae}${board.builtin ? ' · 预置' : ''}</small>`;
}

/* ------------------------------------------------------------------ */
/* inference                                                           */
/* ------------------------------------------------------------------ */
function resetInferenceInput() {
  inferenceFile = null;
  parsedInput = null;
  resultFrame = null;
  resultView = null;
  resultRawView = null;
  demoMatch = null;
  demoReady = false;
  const input = $('#inference-file');
  if (input) input.value = '';
  $('#inference-file-name').textContent = '支持 HIKMICRO 单帧温度 CSV 或 JPEG';
  $('#inference-error').classList.add('hidden');
  $('#run-inference').innerHTML = '开始重建 <span>↗</span>';
  updateInference();
}

async function selectInferenceFile(file) {
  inferenceFile = file || null;
  parsedInput = null;
  resultFrame = null;
  resultView = null;
  resultRawView = null;
  demoMatch = null;
  demoReady = false;
  $('#inference-error').classList.add('hidden');
  if (!file) {
    $('#inference-file-name').textContent = '支持 HIKMICRO 单帧温度 CSV 或 JPEG';
    updateInference();
    return;
  }
  const isJpeg = /\.(jpe?g)$/i.test(file.name);
  $('#inference-file-name').textContent = isJpeg ? '正在读取 JPEG…' : '正在读取 CSV…';
  try {
    const buffer = await file.arrayBuffer();
    if (inferenceFile !== file) return;
    if (isJpeg) {
      // The 16-bit matrix inside a radiometric JPEG is decoded on the server
      // (tdp/io/hm_jpeg.py).  The browser keeps no local preview for it; the
      // field appears when /api/infer returns.
      parsedInput = null;
      if (mode === 'demo') {
        $('#inference-file-name').textContent = `${file.name} · 演示模式不支持 JPEG`;
        showInferenceError('演示模式没有模型可运行。请运行 python web/server.py，再上传任意 CSV 或 JPEG。');
      } else {
        $('#inference-file-name').textContent = `${file.name} · HIKMICRO JPEG，服务端解码`;
      }
    } else {
      parsedInput = parseHmCsv(buffer, file.name);
      $('#inference-file-name').textContent = `${file.name} · ${parsedInput.width} × ${parsedInput.height} px`;
      if (mode === 'demo') {
        const digest = await fileDigest(buffer);
        const found = Object.values(archive).find(data => data.frames.some(frame => frame.source_sha256 === digest));
        if (found) {
          const frame = found.frames.find(item => item.source_sha256 === digest);
          demoMatch = { data: found, frame };
          $('#inference-file-name').textContent = `${file.name} · 演示数据 · ${found.width} × ${found.height} px`;
        } else {
          $('#inference-file-name').textContent = `${file.name} · 演示模式：该文件没有预存结果`;
          showInferenceError('演示模式没有模型可运行。请运行 python web/server.py，再上传任意 CSV 或 JPEG。');
        }
      }
    }
  } catch (error) {
    parsedInput = null;
    $('#inference-file-name').textContent = `${file.name} · ${error.message}`;
  }
  updateInference();
}

async function runInference() {
  if (mode === 'demo') {
    if (!demoMatch) {
      showInferenceError('演示模式只能显示预存帧。请运行 python web/server.py 后再上传任意 CSV。');
      return;
    }
    demoReady = true;
    updateInference();
    return;
  }
  if (!inferenceFile) return;
  const board = activeBoard();
  if (!board) { showInferenceError('请先选择或训练一个板型。'); return; }
  const button = $('#run-inference');
  button.disabled = true;
  button.textContent = '正在重建…';
  $('#result-state').textContent = '处理中';
  $('#pred-overlay').classList.remove('hidden');
  $('#inference-error').classList.add('hidden');
  try {
    const form = new FormData();
    form.append('board_id', board.id);
    form.append('file', inferenceFile, inferenceFile.name);
    const payload = await postForm('/api/infer', form);
    resultFrame = payload.frame;
    resultView = decodeField(resultFrame.predicted);
    renderResult();
    $('#result-state').textContent = '重构完成';
    $('#result-state').classList.add('finished');
  } catch (error) {
    showInferenceError(error.message);
    $('#result-state').textContent = '重建失败';
    $('#result-state').classList.remove('finished');
  } finally {
    button.innerHTML = '开始重建 <span>↗</span>';
    updateInference();
  }
}

function showInferenceError(message) {
  const element = $('#inference-error');
  element.textContent = message;
  element.classList.remove('hidden');
}

function renderResult() {
  const raw = decodeField(resultFrame.raw);
  const predicted = decodeField(resultFrame.predicted);
  resultRawView = raw;
  resultView = predicted;
  drawMap($('#raw-canvas'), raw.display, raw, { filled: true });
  drawMap($('#pred-canvas'), predicted.display, predicted, { filled: true });
  $('#raw-range').textContent = fieldRange(raw.display, raw);
  $('#pred-range').textContent = fieldRange(predicted.display, predicted);
  const warnings = resultFrame.warnings || [];
  const container = $('#frame-warnings');
  if (warnings.length) {
    container.replaceChildren(...warnings.map(text => {
      const item = document.createElement('p');
      item.textContent = text;
      return item;
    }));
    container.classList.remove('hidden');
  } else {
    container.classList.add('hidden');
  }
  const note = $('#signature-note');
  if (resultFrame.signature_score !== null && resultFrame.signature_score !== undefined) {
    note.textContent = `结构匹配度 ${Number(resultFrame.signature_score).toFixed(2)}（1.00 表示与训练板型结构一致；这是匹配度，不是概率）`;
    note.classList.remove('hidden');
  } else {
    note.classList.add('hidden');
  }
  $('#download-csv').disabled = false;
  $('#download-png').disabled = false;
}

function updateInference() {
  updateBoardDetails();
  const hasResult = !!resultFrame;
  if (!hasResult) $('#signature-note').classList.add('hidden');
  if (hasResult) {
    renderResult();
  } else if (mode === 'demo' && demoReady && demoMatch) {
    const { data, frame } = demoMatch;
    const raw = {
      width: data.width, height: data.height, ambient: frame.ambient_c,
      values: frame.raw, display: frame.raw_display, boardMask: data.boardMask
    };
    const predicted = {
      width: data.width, height: data.height, ambient: frame.ambient_c,
      values: frame.predicted, display: frame.predicted_display, boardMask: data.boardMask
    };
    drawMap($('#raw-canvas'), raw.display, raw, { filled: true });
    drawMap($('#pred-canvas'), predicted.display, predicted, { filled: true });
    $('#raw-range').textContent = fieldRange(raw.display, raw);
    $('#pred-range').textContent = fieldRange(predicted.display, predicted);
    $('#result-state').textContent = '重构完成';
    $('#download-csv').disabled = false;
    $('#download-png').disabled = false;
  } else if (parsedInput) {
    drawMap($('#raw-canvas'), parsedInput.values, parsedInput, { filled: true });
    $('#raw-range').textContent = parsedInput.range;
    clearCanvas($('#pred-canvas'), parsedInput.width, parsedInput.height);
    $('#pred-range').textContent = '—';
    $('#result-state').textContent = mode === 'demo' && !demoMatch ? '演示模式' : '待重建';
    $('#download-csv').disabled = true;
    $('#download-png').disabled = true;
  } else {
    clearCanvas($('#raw-canvas'));
    clearCanvas($('#pred-canvas'));
    $('#raw-range').textContent = '—';
    $('#pred-range').textContent = '—';
    $('#result-state').textContent = '待处理';
    $('#download-csv').disabled = true;
    $('#download-png').disabled = true;
    $('#signature-note').classList.add('hidden');
    $('#frame-warnings').classList.add('hidden');
  }
  $('#raw-overlay').classList.toggle('hidden', !!parsedInput || hasResult || (mode === 'demo' && demoReady && !!demoMatch));
  $('#pred-overlay').classList.toggle('hidden', hasResult || (mode === 'demo' && demoReady && !!demoMatch));
  const ready = mode === 'server' ? !!(inferenceFile && activeBoard()) : !!(demoMatch && parsedInput);
  $('#run-inference').disabled = hasResult ? false : !ready;
  $('#run-inference').innerHTML = '开始重建 <span>↗</span>';
  $('#pixel-readout').textContent = (resultFrame || parsedInput) ? '移动光标查看温度' : '将鼠标移至图像查看';
  $('#result-state').classList.toggle('finished', hasResult || (mode === 'demo' && demoReady && !!demoMatch));
}

function inspectPixel(event, kind) {
  const canvas = event.currentTarget;
  let view, values;
  if (resultFrame) {
    view = kind === 'predicted' ? resultView : resultRawView;
    if (!view) return;
    values = view.values;
  } else if (mode === 'demo' && demoReady && demoMatch) {
    const { data, frame } = demoMatch;
    view = { width: data.width, height: data.height, ambient: frame.ambient_c, boardMask: data.boardMask };
    values = kind === 'predicted' ? frame.predicted : frame.raw;
  } else if (parsedInput) {
    view = parsedInput;
    values = parsedInput.values;
  } else {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const shownWidth = canvas.width * ratio, shownHeight = canvas.height * ratio;
  const x = Math.floor((event.clientX - rect.left - (rect.width - shownWidth) / 2) / ratio / 3);
  const y = Math.floor((event.clientY - rect.top - (rect.height - shownHeight) / 2) / ratio / 3);
  const row = y, col = x;
  if (row < 0 || row >= view.height || col < 0 || col >= view.width) return;
  const position = row * view.width + col;
  if (!view.boardMask[position]) return;
  if (kind === 'predicted' && !resultFrame && !(mode === 'demo' && demoReady)) return;
  const value = values[position];
  if (value === -32768 || !Number.isFinite(value)) return;
  $('#pixel-readout').textContent = `第 ${y + 1} 行 · 第 ${x + 1} 列 · ${(value / 10 + view.ambient).toFixed(1)} °C`;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}

function downloadCsv() {
  const board = activeBoard();
  let view, values, stem;
  if (resultFrame) {
    view = resultView || decodeField(resultFrame.predicted);
    values = view.display;
    stem = `${board.id}-${(resultFrame.source_name || 'frame').replace(/\.(csv|jpe?g)$/i, '')}-temperature-c.csv`;
  } else if (mode === 'demo' && demoMatch) {
    const { data, frame } = demoMatch;
    view = { width: data.width, height: data.height, ambient: frame.ambient_c, boardMask: data.boardMask };
    values = frame.predicted_display;
    stem = `${data.id}-${frame.index}-temperature-c.csv`;
  } else return;
  const lines = [];
  for (let row = 0; row < view.height; row++) {
    const cells = [];
    for (let col = 0; col < view.width; col++) {
      const index = row * view.width + col;
      const value = values[index];
      cells.push(value === -32768 || !Number.isFinite(value) ? '' : (value / 10 + view.ambient).toFixed(1));
    }
    lines.push(cells.join(','));
  }
  downloadBlob(new Blob(['\uFEFF', lines.join('\n')], { type: 'text/csv;charset=utf-8' }), stem);
}

function downloadPng() {
  const board = activeBoard();
  const canvas = $('#pred-canvas');
  let caption;
  if (resultFrame) caption = resultFrame.source_name || '上传帧';
  else if (mode === 'demo' && demoMatch) caption = `第 ${demoMatch.frame.index} 帧`;
  else return;
  const output = document.createElement('canvas');
  output.width = 760; output.height = 1040;
  const context = output.getContext('2d');
  context.fillStyle = '#101c22'; context.fillRect(0, 0, output.width, output.height);
  context.fillStyle = '#d4f778'; context.font = '600 18px sans-serif'; context.fillText('THERMAL STUDIO', 44, 50);
  context.fillStyle = '#f2f6ef'; context.font = '600 31px sans-serif'; context.fillText(board ? board.name : 'Board', 44, 98);
  context.fillStyle = '#94aba0'; context.font = '16px sans-serif'; context.fillText(`重建表面温度（°C） · ${caption}`, 44, 130);
  const imageRatio = canvas.width / canvas.height;
  const imageHeight = Math.min(825, 672 / imageRatio);
  const imageWidth = imageHeight * imageRatio;
  context.drawImage(canvas, 44 + (672 - imageWidth) / 2, 160 + (825 - imageHeight) / 2, imageWidth, imageHeight);
  const gradient = context.createLinearGradient(44, 0, 716, 0);
  heatStops.forEach(([stop, color]) => gradient.addColorStop(stop, `rgb(${color.join(',')})`));
  context.fillStyle = gradient; context.fillRect(44, 1000, 672, 9);
  context.fillStyle = '#99aaa1'; context.font = '12px sans-serif'; context.fillText('25 °C', 44, 1028); context.fillText('90 °C', 674, 1028);
  output.toBlob(blob => { if (blob) downloadBlob(blob, `${board ? board.id : 'board'}-heatmap.png`); });
}

/* ------------------------------------------------------------------ */
/* training                                                            */
/* ------------------------------------------------------------------ */
function selectedFiles(id) {
  const input = $(id);
  return input ? [...input.files] : [];
}

function checkTrainingReady() {
  const name = $('#board-name').value.trim();
  const bare = selectedFiles('#bare-files');
  const painted = selectedFiles('#painted-files');
  const server = mode === 'server';
  const ready = server && !!name && bare.length >= 4 && painted.length >= 4;
  $('#start-training').disabled = !ready || !!trainingStartedAt;
  const line = $('#training-validation');
  line.classList.toggle('valid', ready);
  if (!server) {
    line.textContent = '演示模式没有后端，无法训练。请运行 python web/server.py。';
  } else if (!name) {
    line.textContent = '请填写板型名称，并选择两组温度数据。';
  } else if (bare.length < 4 || painted.length < 4) {
    line.textContent = `每组至少需要 4 帧；当前裸板 ${bare.length} 帧、黑化板 ${painted.length} 帧。`;
  } else {
    line.textContent = `已选择裸板 ${bare.length} 帧、黑化板 ${painted.length} 帧。`;
  }
}

function setUploadLabel(input, labelId, boxId) {
  const count = input.files.length;
  $(labelId).textContent = count ? `已选择 ${count} 个文件` : '选择 CSV / JPEG 文件';
  $(boxId).classList.toggle('filled', count > 0);
  checkTrainingReady();
}

function setProgress(percent) {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  $('#pipeline-ring').style.setProperty('--progress', `${value}%`);
  $('#pipeline-ring span').textContent = `${Math.round(value)}%`;
  if (trainingStartedAt) {
    const seconds = Math.floor((Date.now() - trainingStartedAt) / 1000);
    $('#pipeline-clock').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  }
}

function renderStages(stages, activeKey, done = false) {
  const container = $('#pipeline-stages');
  container.replaceChildren();
  container.style.gridTemplateColumns = `repeat(${stages.length}, minmax(0, 1fr))`;
  const activeIndex = stages.findIndex(stage => stage.key === activeKey);
  stages.forEach((stage, index) => {
    const element = document.createElement('div');
    element.className = `stage ${done || (activeIndex >= 0 && index < activeIndex) ? 'done' : index === activeIndex ? 'active' : ''}`;
    element.innerHTML = `<div class="stage-line"></div><div class="stage-label">${stage.label}</div>`;
    container.append(element);
  });
}

function renderJob(job) {
  const stages = job.stages && job.stages.length ? job.stages : DEFAULT_STAGES;
  const activeKey = job.status === 'done' ? null : job.stage;
  const activeIndex = stages.findIndex(stage => stage.key === activeKey);
  renderStages(stages, activeKey, job.status === 'done');
  setProgress(job.status === 'done' ? 100 : job.progress);
  const current = stages[activeIndex] || stages[stages.length - 1];
  $('#pipeline-title').textContent = job.status === 'done' ? '板型已就绪' : job.status === 'error' ? '训练失败' : current.label;
  $('#pipeline-subtitle').textContent = job.detail || '';
  renderJobLog(job, stages, activeIndex);
  if (job.history && job.history.length) {
    $('#training-chart').classList.remove('hidden');
    const totalEpochs = Number($('#epochs').value) || (job.board && job.board.training && job.board.training.epochs) || job.history[job.history.length - 1].epoch;
    drawChart(job.history, totalEpochs);
  }
}

function renderJobLog(job, stages, activeIndex) {
  const container = $('#pipeline-log');
  container.replaceChildren();
  const reached = activeIndex < 0 ? stages.length : activeIndex + 1;
  const lastByStage = {};
  (job.logs || []).forEach(entry => { lastByStage[entry.stage] = entry.message; });
  for (let index = 0; index < reached; index++) {
    const stage = stages[index];
    const message = lastByStage[stage.key];
    if (!message && index !== activeIndex) continue;
    const entry = document.createElement('section');
    entry.className = 'log-entry';
    const head = document.createElement('div');
    head.className = 'log-entry-head';
    const title = document.createElement('strong');
    title.textContent = `${String(index + 1).padStart(2, '0')}  ${stage.label}`;
    const state = document.createElement('span');
    state.className = `log-state ${index === activeIndex && job.status === 'running' ? 'running' : job.status === 'error' && index === activeIndex ? 'error' : 'done'}`;
    state.textContent = index === activeIndex && job.status === 'running' ? '进行中' : job.status === 'error' && index === activeIndex ? '失败' : '完成';
    head.append(title, state);
    const body = document.createElement('div');
    body.className = 'log-entry-body';
    const summary = document.createElement('p');
    summary.textContent = message || stage.label;
    body.append(summary);
    entry.append(head, body);
    container.append(entry);
  }
  if (job.status === 'error' && job.error) {
    const entry = document.createElement('section');
    entry.className = 'log-entry is-active';
    const head = document.createElement('div');
    head.className = 'log-entry-head';
    head.innerHTML = '<strong>错误</strong><span class="log-state error">失败</span>';
    const body = document.createElement('div');
    body.className = 'log-entry-body';
    const summary = document.createElement('p');
    summary.textContent = job.error;
    body.append(summary);
    entry.append(head, body);
    container.append(entry);
  }
  if (!container.children.length) {
    const empty = document.createElement('p');
    empty.className = 'log-empty';
    empty.textContent = '尚无处理记录';
    container.append(empty);
  }
}

function drawChart(history, totalEpochs) {
  const svg = $('#loss-chart');
  const points = (history || []).filter(point => Number.isFinite(point.board_mae_k));
  if (!points.length) {
    svg.innerHTML = '';
    $('#chart-value').textContent = '— K';
    return;
  }
  const total = Math.max(1, Number(totalEpochs) || points[points.length - 1].epoch || 1);
  const maxValue = Math.max(0.25, ...points.map(point => point.board_mae_k)) * 1.15;
  const x = epoch => 46 + (Math.min(epoch, total) / total) * 454;
  const y = value => 116 - (Math.min(value, maxValue) / maxValue) * 96;
  const grid = [0, 0.25, 0.5, 0.75, 1].map(fraction => {
    const value = maxValue * fraction;
    const lineY = y(value);
    return `<line x1="46" x2="506" y1="${lineY}" y2="${lineY}" stroke="#e7eee5" stroke-width="1"/><text x="0" y="${lineY + 3}" fill="#a8b8a9" font-size="9">${value.toFixed(2)}</text>`;
  }).join('');
  const path = points.map((point, index) => `${index ? 'L' : 'M'} ${x(point.epoch).toFixed(1)} ${y(point.board_mae_k).toFixed(1)}`).join(' ');
  const area = points.length ? `${path} L ${x(points[points.length - 1].epoch).toFixed(1)} 116 L ${x(points[0].epoch).toFixed(1)} 116 Z` : '';
  const last = points[points.length - 1];
  const dot = `<circle cx="${x(last.epoch).toFixed(1)}" cy="${y(last.board_mae_k).toFixed(1)}" r="4" fill="#587f5b" stroke="#fff" stroke-width="2"/>`;
  svg.innerHTML = `${grid}<path d="${area}" fill="#e7f3dc" opacity="0.8"/><path d="${path}" fill="none" stroke="#6e9e6c" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>${dot}`;
  $('#chart-value').textContent = `${last.board_mae_k.toFixed(3)} K`;
  $('#chart-min').textContent = '0 轮';
  $('#chart-mid').textContent = `${Math.round(total / 2)} 轮`;
  $('#chart-max').textContent = `${Math.round(total)} 轮`;
}

async function startTraining() {
  if (mode !== 'server') {
    showTrainingError('演示模式没有后端，无法训练。');
    return;
  }
  const name = $('#board-name').value.trim();
  const bare = selectedFiles('#bare-files');
  const painted = selectedFiles('#painted-files');
  if (!name || bare.length < 4 || painted.length < 4) return;
  const token = ++trainingToken;
  trainingStartedAt = Date.now();
  $('#training-error').classList.add('hidden');
  $('#training-complete').classList.add('hidden');
  $('#training-chart').classList.add('hidden');
  $('#pipeline-log').replaceChildren();
  $('#pipeline-title').textContent = '正在上传数据';
  $('#pipeline-subtitle').textContent = '文件会发送到本机后端，不上传网络。';
  setProgress(0);
  renderStages(DEFAULT_STAGES, DEFAULT_STAGES[0].key);
  $('#start-training').textContent = '正在训练…';
  $('#start-training').disabled = true;
  drawChart([], 0);
  const form = new FormData();
  form.append('name', name);
  form.append('epochs', $('#epochs').value);
  const size = $('#board-size').value.trim();
  if (size) form.append('board_size_mm', size);
  bare.forEach(file => form.append('bare', file, file.name));
  painted.forEach(file => form.append('painted', file, file.name));
  let job;
  try {
    const payload = await postForm('/api/boards', form, fraction => {
      setProgress(fraction * 4);
      $('#pipeline-subtitle').textContent = `正在上传 ${Math.round(fraction * 100)}%`;
    });
    job = payload.job;
  } catch (error) {
    showTrainingError(error.message);
    trainingStartedAt = 0;
    $('#start-training').innerHTML = '开始训练 <span>↗</span>';
    checkTrainingReady();
    return;
  }
  if (token !== trainingToken) return;
  renderJob(job);
  pollJob(job.id, token);
}

async function pollJob(jobId, token) {
  if (token !== trainingToken) return;
  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch (error) {
    showTrainingError(`读取训练进度失败：${error.message}`);
    return;
  }
  if (token !== trainingToken) return;
  renderJob(job);
  if (job.status === 'done') {
    trainingStartedAt = 0;
    await refreshBoards(job.board ? job.board.id : null);
    $('#training-complete').classList.remove('hidden');
    $('#complete-copy').textContent = `${job.board ? job.board.name : '新板型'} 已可用于温度重建。上报的是训练集误差。`;
    $('#start-training').innerHTML = '再次训练 <span>↗</span>';
    checkTrainingReady();
    $('#training-chart').classList.remove('hidden');
    return;
  }
  if (job.status === 'error') {
    trainingStartedAt = 0;
    showTrainingError(job.error || '训练失败');
    $('#start-training').innerHTML = '重新训练 <span>↗</span>';
    checkTrainingReady();
    return;
  }
  pollTimer = setTimeout(() => pollJob(jobId, token), 700);
}

function showTrainingError(message) {
  const element = $('#training-error');
  element.textContent = message;
  element.classList.remove('hidden');
}

/* ------------------------------------------------------------------ */
/* library                                                             */
/* ------------------------------------------------------------------ */
async function updateLibrary() {
  const grid = $('#library-grid');
  grid.replaceChildren();
  const boards = mode === 'server' ? serverBoards : demoBoards;
  $('#library-empty').classList.toggle('hidden', boards.length > 0);
  for (const board of boards) {
    const card = document.createElement('article');
    card.className = 'library-card';
    const preview = document.createElement('div');
    preview.className = 'library-preview';
    const canvas = document.createElement('canvas');
    preview.append(canvas);
    const body = document.createElement('div');
    body.className = 'library-body';
    const top = document.createElement('div');
    top.className = 'library-body-top';
    const titleBlock = document.createElement('div');
    const title = document.createElement('h2');
    title.textContent = board.name;
    const subtitle = document.createElement('p');
    subtitle.textContent = board.builtin ? '预置板型' : '已训练板型';
    titleBlock.append(title, subtitle);
    const tag = document.createElement('span');
    tag.className = 'library-tag';
    tag.textContent = '● 可用';
    top.append(titleBlock, tag);
    const meta = document.createElement('div');
    meta.className = 'board-meta';
    const gridText = board.grid_hw ? `${board.grid_hw[1]} × ${board.grid_hw[0]} px` : '';
    meta.innerHTML = `<span>${board.pair_count || 0} 组训练数据</span><span>${gridText}</span>`;
    const metricPills = document.createElement('div');
    metricPills.className = 'library-metrics';
    const metric = boardMetric(board);
    if (metric && Number.isFinite(metric.board_mae_k)) {
      const pill = document.createElement('span');
      pill.className = 'metric-pill';
      pill.textContent = `训练集板体 MAE ${metric.board_mae_k.toFixed(3)} K`;
      metricPills.append(pill);
    }
    if (metric && Number.isFinite(metric.masked_mae_k)) {
      const pill = document.createElement('span');
      pill.className = 'metric-pill';
      pill.textContent = `不可信区 MAE ${metric.masked_mae_k.toFixed(3)} K`;
      metricPills.append(pill);
    }
    const actions = document.createElement('div');
    actions.className = 'card-actions';
    const enter = document.createElement('button');
    enter.type = 'button';
    enter.textContent = '进入重构 ↗';
    enter.addEventListener('click', () => {
      activeBoardId = board.id;
      resetInferenceInput();
      updateBoardPicker();
      navigate('inference');
    });
    actions.append(enter);
    if (mode === 'server' && !board.builtin) {
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'card-delete';
      remove.textContent = '删除板型';
      remove.addEventListener('click', () => deleteBoard(board));
      actions.append(remove);
    }
    body.append(top, meta, metricPills, actions);
    card.append(preview, body);
    grid.append(card);
    drawPreview(canvas, board);
  }
}

async function drawPreview(canvas, board) {
  try {
    let encoded;
    if (mode === 'server') {
      if (!previewCache[board.id]) previewCache[board.id] = await api(`/api/boards/${board.id}/preview`);
      encoded = previewCache[board.id];
    } else {
      const data = archive[board.base || board.id];
      if (!data) return;
      const frame = data.frames[Math.min(4, data.frames.length - 1)];
      drawMap(canvas, frame.predicted_display, {
        width: data.width, height: data.height, ambient: frame.ambient_c,
        values: frame.predicted, display: frame.predicted_display, boardMask: data.boardMask
      }, { filled: true });
      return;
    }
    const view = decodeField(encoded.predicted);
    drawMap(canvas, view.display, view, { filled: true });
  } catch (error) {
    /* a missing preview must not break the library */
  }
}

async function deleteBoard(board) {
  if (!confirm(`删除板型「${board.name}」？模型文件会一并删除。`)) return;
  try {
    await api(`/api/boards/${board.id}`, { method: 'DELETE' });
    await refreshBoards(activeBoardId === board.id ? null : activeBoardId);
  } catch (error) {
    showTrainingError(error.message);
  }
}

async function refreshBoards(preferredId = null) {
  if (mode !== 'server') return;
  const previousId = activeBoardId;
  try {
    const payload = await api('/api/boards');
    serverBoards = payload.boards || [];
  } catch (error) {
    serverBoards = [];
  }
  if (preferredId && serverBoards.some(board => board.id === preferredId)) activeBoardId = preferredId;
  else if (!serverBoards.some(board => board.id === activeBoardId)) activeBoardId = serverBoards[0] ? serverBoards[0].id : null;
  Object.keys(previewCache).forEach(key => { if (!serverBoards.some(board => board.id === key)) delete previewCache[key]; });
  if (activeBoardId !== previousId) resetInferenceInput();
  updateBoardPicker();
  updateInference();
  updateLibrary();
}

async function resetPresentation() {
  trainingToken++;
  if (pollTimer) clearTimeout(pollTimer);
  trainingStartedAt = 0;
  if (mode === 'server') {
    const removable = serverBoards.filter(board => !board.builtin);
    if (removable.length && !confirm(`清除 ${removable.length} 个新增板型？内置板型会保留。`)) return;
    for (const board of removable) {
      try { await api(`/api/boards/${board.id}`, { method: 'DELETE' }); } catch { /* keep going */ }
    }
    await refreshBoards(null);
  }
  $('#board-name').value = '';
  $('#board-size').value = '';
  $('#bare-files').value = '';
  $('#painted-files').value = '';
  $('#bare-files-label').textContent = '选择 CSV 文件';
  $('#painted-files-label').textContent = '选择 CSV 文件';
  $('#bare-box').classList.remove('filled');
  $('#painted-box').classList.remove('filled');
  $('#training-complete').classList.add('hidden');
  $('#training-chart').classList.add('hidden');
  $('#training-error').classList.add('hidden');
  $('#pipeline-log').replaceChildren();
  const empty = document.createElement('p');
  empty.className = 'log-empty';
  empty.textContent = '尚无处理记录';
  $('#pipeline-log').append(empty);
  $('#pipeline-title').textContent = '等待导入数据';
  $('#pipeline-subtitle').textContent = '训练步骤与结果将在下方显示。';
  $('#pipeline-clock').textContent = '00:00';
  setProgress(0);
  $('#start-training').innerHTML = '开始训练 <span>↗</span>';
  renderStages(DEFAULT_STAGES, null);
  drawChart([], 0);
  checkTrainingReady();
  resetInferenceInput();
  updateBoardPicker();
  updateLibrary();
  navigate('inference');
}

/* ------------------------------------------------------------------ */
/* navigation and events                                               */
/* ------------------------------------------------------------------ */
function navigate(view) {
  $$('.view').forEach(element => element.classList.toggle('active', element.id === `${view}-view`));
  $$('.nav-item').forEach(button => button.classList.toggle('active', button.dataset.view === view));
  $('#breadcrumb-current').textContent = ({ inference: '温度重建', training: '训练新板', library: '板型库' })[view];
  if (view === 'library') updateLibrary();
  if (view === 'training') checkTrainingReady();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function connectEvents() {
  $$('.nav-item').forEach(button => button.addEventListener('click', () => navigate(button.dataset.view)));
  $('#board-select').addEventListener('change', event => { activeBoardId = event.target.value; resetInferenceInput(); });
  $('#run-inference').addEventListener('click', runInference);
  $('#download-csv').addEventListener('click', downloadCsv);
  $('#download-png').addEventListener('click', downloadPng);
  $('#raw-canvas').addEventListener('mousemove', event => inspectPixel(event, 'raw'));
  $('#pred-canvas').addEventListener('mousemove', event => inspectPixel(event, 'predicted'));
  $('#inference-file').addEventListener('change', event => selectInferenceFile(event.target.files[0] || null));
  const drop = $('#inference-drop');
  ['dragenter', 'dragover'].forEach(type => drop.addEventListener(type, event => { event.preventDefault(); drop.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(type => drop.addEventListener(type, event => { event.preventDefault(); drop.classList.remove('dragging'); }));
  drop.addEventListener('drop', event => {
    const file = [...event.dataTransfer.files].find(item => /\.(csv|jpe?g)$/i.test(item.name));
    if (file) selectInferenceFile(file);
  });
  $('#board-name').addEventListener('input', checkTrainingReady);
  $('#bare-files').addEventListener('change', event => setUploadLabel(event.target, '#bare-files-label', '#bare-box'));
  $('#painted-files').addEventListener('change', event => setUploadLabel(event.target, '#painted-files-label', '#painted-box'));
  $('#start-training').addEventListener('click', startTraining);
  $('#open-trained-board').addEventListener('click', () => { resetInferenceInput(); navigate('inference'); });
  $('#library-add').addEventListener('click', () => navigate('training'));
  $('#reset-demo').addEventListener('click', resetPresentation);
}

/* ------------------------------------------------------------------ */
/* boot                                                                */
/* ------------------------------------------------------------------ */
function setupModeBanner() {
  const banner = $('#mode-banner');
  banner.classList.remove('hidden');
  if (mode === 'server') {
    banner.innerHTML = '<strong>后端已连接。</strong> 任意 HIKMICRO CSV 会在服务端完成真实推理；新增板型会真正训练模型。';
    banner.classList.remove('demo');
    $('#backend-state').textContent = '后端已连接';
    $('#topbar-state').textContent = '模型服务运行中';
  } else {
    banner.innerHTML = '<strong>演示模式：未连接后端。</strong> 页面只能显示内置样例。运行 <code>python web/server.py</code> 后刷新，即可上传任意 CSV 并训练新板型。';
    banner.classList.add('demo');
    $('#backend-state').textContent = '演示模式（无后端）';
    $('#topbar-state').textContent = '只读演示';
  }
}

async function loadDemo() {
  mode = 'demo';
  const responses = await Promise.all(['raspberry-pi-4b', 'orange-pi-5-pro'].map(id => fetch(`assets/${id}.json`)));
  if (responses.some(response => !response.ok)) throw new Error('无法加载演示数据');
  const payload = await Promise.all(responses.map(response => response.json()));
  payload.forEach(item => {
    item.boardMask = unpackMask(item.mask, item.width * item.height);
    archive[item.id] = item;
  });
  demoBoards = payload.map(item => {
    const last = item.training && item.training.history ? item.training.history[item.training.history.length - 1] : null;
    return {
      id: item.id, name: item.name, base: item.id, builtin: true,
      grid_hw: [item.height, item.width], pair_count: item.pair_count,
      metrics: last ? { final: { board_mae_k: last.board_mae_k } } : {}
    };
  });
  activeBoardId = demoBoards[0] ? demoBoards[0].id : null;
}

async function init() {
  try {
    await api('/api/health');
    mode = 'server';
    const payload = await api('/api/boards');
    serverBoards = payload.boards || [];
    activeBoardId = serverBoards[0] ? serverBoards[0].id : null;
  } catch {
    await loadDemo();
  }
  setupModeBanner();
  connectEvents();
  updateBoardPicker();
  updateInference();
  updateLibrary();
  renderStages(DEFAULT_STAGES, null);
  drawChart([], 0);
  checkTrainingReady();
}

init().catch(error => {
  mode = 'demo';
  setupModeBanner();
  $('#result-state').textContent = '初始化失败';
  $('#inference-file-name').textContent = error.message;
  $('#mode-banner').textContent = `无法加载页面数据：${error.message}`;
});
