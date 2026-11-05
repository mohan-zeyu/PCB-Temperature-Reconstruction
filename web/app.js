const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const STORAGE_KEY = 'thermal-studio-boards-v2';
const archive = {};
const stages = [
  { label: '文件检查', title: '检查采集文件', subtitle: '读取裸板与黑化板采集数据', detail: '核对半负载与满负载下的裸板、黑化板记录。', metrics: [['4', '采集序列'], ['2', '负载工况']], duration: 1400 },
  { label: '板体提取', title: '提取板体区域', subtitle: '统一各帧板体位置与尺寸', detail: '识别板体边界，将温度帧配准至统一网格。', metrics: [['164 × 103', '配准网格'], ['16,892', '板体像素']], duration: 2400 },
  { label: '时序对齐', title: '对齐采集序列', subtitle: '按负载开始后的时间匹配热状态', detail: '对黑化板记录进行时间插值；后段采用稳态均值。', metrics: [['314', '插值配对'], ['164', '稳态目标']], duration: 3600 },
  { label: '数据配对', title: '建立训练数据', subtitle: '生成裸板与黑化板温度对', detail: '温度场按各帧环境温度归一，建立模型训练样本。', metrics: [['480', '训练配对'], ['2', '负载工况']], duration: 1900 },
  { label: '模型训练', title: '训练温度重构模型', subtitle: '显示误差随轮次变化', detail: '训练 300 轮，逐步更新温度重构参数。', metrics: [['300', '训练轮次'], ['0.170 K', '最佳板体平均误差']], duration: 8800 },
  { label: '模型入库', title: '模型入库', subtitle: '将新板型加入板型列表', detail: '该板型现可用于温度重构。', metrics: [['1', '新增板型'], ['可用', '模型状态']], duration: 1000 }
];

let boards = loadBoards();
let activeBoardId = 'raspberry-pi-4b';
let inferenceReady = false;
let inferenceFile = null;
let parsedInput = null;
let matchedFrame = null;
let trainingSourceReady = false;
let trainingRun = 0;
let trainingStartedAt = 0;

function loadBoards() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return [{ id: 'raspberry-pi-4b', name: 'Raspberry Pi 4B', base: 'raspberry-pi-4b', builtIn: true }, ...saved.filter(item => item && typeof item.name === 'string' && item.base === 'orange-pi-5-pro')];
  } catch {
    return [{ id: 'raspberry-pi-4b', name: 'Raspberry Pi 4B', base: 'raspberry-pi-4b', builtIn: true }];
  }
}

function saveBoards() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(boards.filter(board => !board.builtIn))); } catch { /* private browsing */ }
}

function currentBoard() { return boards.find(board => board.id === activeBoardId) || boards[0]; }
function currentArchive() { return archive[currentBoard().base]; }
function unpackMask(encoded, count) {
  const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
  return Array.from({ length: count }, (_, index) => !!(bytes[index >> 3] & (128 >> (index & 7))));
}

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
  return [0, 1, 2].map(i => Math.round(ac[i] + (bc[i] - ac[i]) * amount));
}

function drawMap(canvas, values, data, ambientC, filledValues = null) {
  const field = filledValues || values;
  const outputWidth = data.width;
  const outputHeight = data.height;
  const small = document.createElement('canvas');
  small.width = outputWidth;
  small.height = outputHeight;
  const context = small.getContext('2d');
  const image = context.createImageData(outputWidth, outputHeight);
  for (let row = 0; row < data.height; row++) {
    for (let col = 0; col < data.width; col++) {
      const source = row * data.width + col;
      const x = data.width - 1 - col;
      const y = data.height - 1 - row;
      const destination = (y * outputWidth + x) * 4;
      const color = filledValues || data.boardMask[source] ? colorFor(field[source] / 10 + ambientC) : [16, 28, 34];
      image.data[destination] = color[0];
      image.data[destination + 1] = color[1];
      image.data[destination + 2] = color[2];
      image.data[destination + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  canvas.width = outputWidth * 3;
  canvas.height = outputHeight * 3;
  const target = canvas.getContext('2d');
  target.imageSmoothingEnabled = true;
  target.drawImage(small, 0, 0, canvas.width, canvas.height);
}

function clearCanvas(canvas) {
  const data = currentArchive();
  canvas.width = data.width * 3;
  canvas.height = data.height * 3;
}

function fieldRange(values, data, ambientC) {
  let low = Infinity, high = -Infinity;
  values.forEach((value, index) => { if (data.boardMask[index]) { low = Math.min(low, value); high = Math.max(high, value); } });
  return `${(low / 10 + ambientC).toFixed(1)}–${(high / 10 + ambientC).toFixed(1)} °C`;
}

function updateBoardPicker() {
  const select = $('#board-select');
  select.replaceChildren();
  boards.forEach(board => {
    const option = document.createElement('option');
    option.value = board.id;
    option.textContent = board.name;
    select.append(option);
  });
  select.value = activeBoardId;
  $('#topbar-count').textContent = `已就绪 ${boards.length} 种板型`;
  $('#library-count').textContent = `${boards.length} 种板型`;
}

function updateInference() {
  const data = currentArchive();
  $('#board-details').innerHTML = `<span>${data.pair_count} 组训练数据</span><strong>${data.width} × ${data.height} px</strong>`;
  if (matchedFrame) {
    drawMap($('#raw-canvas'), matchedFrame.raw, data, matchedFrame.ambient_c, matchedFrame.raw_display);
    $('#raw-range').textContent = fieldRange(matchedFrame.raw, data, matchedFrame.ambient_c);
  } else if (parsedInput) {
    drawMap($('#raw-canvas'), parsedInput.values, parsedInput, 0);
    $('#raw-range').textContent = parsedInput.range;
  } else { clearCanvas($('#raw-canvas')); $('#raw-range').textContent = '—'; }
  if (inferenceReady && matchedFrame) {
    drawMap($('#pred-canvas'), matchedFrame.predicted, data, matchedFrame.ambient_c, matchedFrame.predicted_display);
    $('#pred-range').textContent = fieldRange(matchedFrame.predicted, data, matchedFrame.ambient_c);
  } else { clearCanvas($('#pred-canvas')); $('#pred-range').textContent = '—'; }
  $('#raw-overlay')?.classList.toggle('hidden', !!parsedInput || !!matchedFrame);
  $('#pred-overlay').classList.toggle('hidden', inferenceReady && !!matchedFrame);
  $('#result-state').textContent = inferenceReady ? '重构完成' : matchedFrame ? '待重构' : '待导入';
  $('#result-state').classList.toggle('finished', inferenceReady);
  $('#run-inference').disabled = !matchedFrame;
  if (!inferenceReady && $('#run-inference').textContent.includes('正在')) $('#run-inference').innerHTML = '开始重建 <span>↗</span>';
  $('#download-csv').disabled = !inferenceReady;
  $('#download-png').disabled = !inferenceReady;
  $('#pixel-readout').textContent = matchedFrame || parsedInput ? '移动光标查看温度' : '请选择裸板 CSV 文件';
}

function navigate(view) {
  $$('.view').forEach(element => element.classList.toggle('active', element.id === `${view}-view`));
  $$('.nav-item').forEach(button => button.classList.toggle('active', button.dataset.view === view));
  $('#breadcrumb-current').textContent = ({ inference: '温度重构', training: '新增板型', library: '板型库' })[view];
  if (view === 'library') updateLibrary();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function parseHmCsv(buffer, name) {
  let decoded;
  try { decoded = new TextDecoder('utf-8', { fatal: true }).decode(buffer); }
  catch { decoded = new TextDecoder('gb18030').decode(buffer); }
  const lines = decoded.replace(/^\uFEFF/, '').split(/\r?\n/);
  const header = lines.findIndex(line => {
    const cells = line.split(',');
    return cells.length >= 193 && cells.slice(1, 4).map(cell => cell.trim()).join(',') === '0,1,2';
  });
  if (header < 0) throw new Error('未找到温度矩阵');
  const width = 192, height = 256, values = [];
  let low = Infinity, high = -Infinity;
  for (let row = 0; row < height; row++) {
    const cells = (lines[header + row + 1] || '').split(',');
    if (Number(cells[0]) !== row || cells.length < width + 1) throw new Error('温度矩阵尺寸不正确');
    for (let col = 0; col < width; col++) {
      const value = Number(cells[col + 1]);
      if (!Number.isFinite(value)) throw new Error('温度矩阵包含无效数值');
      low = Math.min(low, value); high = Math.max(high, value);
      values.push(Math.round(value * 10));
    }
  }
  return { name, width, height, values, boardMask: Array(values.length).fill(true), range: `${low.toFixed(1)}–${high.toFixed(1)} °C` };
}
async function selectInferenceFile(file) {
  inferenceFile = file || null; parsedInput = null; matchedFrame = null; inferenceReady = false;
  if (!file) { $('#inference-file-name').textContent = '请选择 HIKMICRO CSV 文件'; updateInference(); return; }
  $('#inference-file-name').textContent = '正在读取 CSV…';
  updateInference();
  $('#result-state').textContent = '预处理中';
  $('#raw-overlay b').textContent = '正在读取 CSV';
  try {
    const bytes = await file.arrayBuffer();
    const parsed = parseHmCsv(bytes, file.name);
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(byte => byte.toString(16).padStart(2, '0')).join('');
    if (inferenceFile !== file) return;
    $('#inference-file-name').textContent = '正在提取板体…';
    $('#raw-overlay b').textContent = '正在提取板体';
    await delay(650);
    if (inferenceFile !== file) return;
    parsedInput = parsed;
    const found = currentArchive().frames.find(frame => frame.source_sha256 === digest);
    if (found) {
      matchedFrame = found;
      $('#inference-file-name').textContent = `${file.name} · ${parsed.width} × ${parsed.height} → ${currentArchive().width} × ${currentArchive().height} px`;
    } else {
      const other = Object.values(archive).find(data => data.frames.some(frame => frame.source_sha256 === digest));
      $('#inference-file-name').textContent = other ? `${file.name} · 请切换至 ${other.name}` : `${file.name} · 数据已解析，当前模型无对应结果`;
    }
  } catch (error) {
    $('#inference-file-name').textContent = `${file.name} · ${error.message}`;
  }
  $('#raw-overlay b').textContent = '请选择 CSV 文件';
  updateInference();
}

function runInference() {
  if (!matchedFrame) return;
  const selectedFile = inferenceFile, selectedFrame = matchedFrame, selectedBoard = activeBoardId;
  const button = $('#run-inference');
  button.disabled = true;
  button.textContent = '正在重构…';
  $('#result-state').textContent = '处理中';
  setTimeout(() => {
    if (inferenceFile !== selectedFile || matchedFrame !== selectedFrame || activeBoardId !== selectedBoard) return;
    inferenceReady = true;
    button.innerHTML = '开始重建 <span>↗</span>';
    updateInference();
  }, 850);
}

function inspectPixel(event, kind) {
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const shownWidth = canvas.width * ratio, shownHeight = canvas.height * ratio;
  const x = Math.floor((event.clientX - rect.left - (rect.width - shownWidth) / 2) / ratio / 3);
  const y = Math.floor((event.clientY - rect.top - (rect.height - shownHeight) / 2) / ratio / 3);
  const data = kind === 'predicted' || matchedFrame ? currentArchive() : parsedInput;
  if (!data) return;
  const row = data.height - 1 - y, col = data.width - 1 - x;
  if (row < 0 || row >= data.height || col < 0 || col >= data.width) return;
  const position = row * data.width + col;
  if (!data.boardMask[position] || (kind === 'predicted' && !inferenceReady)) return;
  const value = kind === 'predicted' ? matchedFrame.predicted[position] / 10 + matchedFrame.ambient_c : matchedFrame ? matchedFrame.raw[position] / 10 + matchedFrame.ambient_c : parsedInput.values[position] / 10;
  $('#pixel-readout').textContent = `第 ${y + 1} 行 · 第 ${x + 1} 列 · ${value.toFixed(1)} °C`;
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
  if (!inferenceReady || !matchedFrame) return;
  const data = currentArchive(), frame = matchedFrame;
  const lines = [];
  // Export the same corner-filled temperature field shown in the heatmap.
  // Rows follow the portrait display orientation.
  for (let row = data.height - 1; row >= 0; row--) {
    const cells = [];
    for (let col = data.width - 1; col >= 0; col--) {
      const index = row * data.width + col;
      cells.push((frame.predicted_display[index] / 10 + frame.ambient_c).toFixed(1));
    }
    lines.push(cells.join(','));
  }
  downloadBlob(new Blob(['\uFEFF', lines.join('\n')], { type: 'text/csv;charset=utf-8' }), `${currentBoard().base}-${frame.index}-temperature-c.csv`);
}

function downloadPng() {
  if (!inferenceReady || !matchedFrame) return;
  const frame = matchedFrame, data = currentArchive();
  const boardCanvas = $('#pred-canvas');
  const output = document.createElement('canvas');
  output.width = 760; output.height = 1040;
  const context = output.getContext('2d');
  context.fillStyle = '#101c22'; context.fillRect(0, 0, output.width, output.height);
  context.fillStyle = '#d4f778'; context.font = '600 18px sans-serif'; context.fillText('THERMAL STUDIO', 44, 50);
  context.fillStyle = '#f2f6ef'; context.font = '600 31px sans-serif'; context.fillText(currentBoard().name, 44, 98);
  context.fillStyle = '#94aba0'; context.font = '16px sans-serif'; context.fillText(`重构表面温度（°C） · 第 ${frame.index} 帧`, 44, 130);
  const imageRatio = boardCanvas.width / boardCanvas.height;
  const imageHeight = Math.min(825, 672 / imageRatio);
  const imageWidth = imageHeight * imageRatio;
  context.drawImage(boardCanvas, 44 + (672 - imageWidth) / 2, 160 + (825 - imageHeight) / 2, imageWidth, imageHeight);
  const gradient = context.createLinearGradient(44, 0, 716, 0);
  heatStops.forEach(([stop, color]) => gradient.addColorStop(stop, `rgb(${color.join(',')})`));
  context.fillStyle = gradient; context.fillRect(44, 1000, 672, 9);
  context.fillStyle = '#99aaa1'; context.font = '12px sans-serif'; context.fillText('25 °C', 44, 1028); context.fillText('90 °C', 674, 1028);
  output.toBlob(blob => { if (blob) downloadBlob(blob, `${data.id}-${frame.index}-heatmap.png`); });
}

function updateLibrary() {
  const grid = $('#library-grid');
  grid.replaceChildren();
  boards.forEach(board => {
    const data = archive[board.base];
    const card = document.createElement('article');
    card.className = 'library-card';
    const preview = document.createElement('div'); preview.className = 'library-preview';
    const canvas = document.createElement('canvas');
    preview.append(canvas);
    const body = document.createElement('div'); body.className = 'library-body';
    const top = document.createElement('div'); top.className = 'library-body-top';
    const titleBlock = document.createElement('div');
    const title = document.createElement('h2'); title.textContent = board.name;
    const subtitle = document.createElement('p'); subtitle.textContent = board.builtIn ? '预置板型' : '已训练板型';
    titleBlock.append(title, subtitle);
    const tag = document.createElement('span'); tag.className = 'library-tag'; tag.textContent = '● 可用';
    top.append(titleBlock, tag);
    const meta = document.createElement('div'); meta.className = 'board-meta';
    meta.innerHTML = `<span>${data.pair_count} 组训练数据</span><span>${data.width} × ${data.height} px</span>`;
    const button = document.createElement('button'); button.type = 'button'; button.textContent = '进入重构 ↗';
    button.addEventListener('click', () => { activeBoardId = board.id; resetInferenceInput(); updateBoardPicker(); navigate('inference'); });
    body.append(top, meta, button);
    card.append(preview, body);
    grid.append(card);
    drawMap(canvas, data.frames[4].predicted, data, data.frames[4].ambient_c, data.frames[4].predicted_display);
  });
}

function resetInferenceInput() {
  inferenceFile = null; parsedInput = null; matchedFrame = null; inferenceReady = false;
  $('#inference-file').value = '';
  $('#inference-file-name').textContent = '请选择 HIKMICRO CSV 文件';
  $('#raw-overlay b').textContent = '请选择 CSV 文件';
  $('#run-inference').innerHTML = '开始重构 <span>↗</span>';
  updateInference();
}

function resetPresentation() {
  // Stop any running playback before clearing its UI and local board entries.
  trainingRun++;
  trainingStartedAt = 0;
  trainingSourceReady = false;
  boards = boards.filter(board => board.builtIn);
  saveBoards();
  activeBoardId = 'raspberry-pi-4b';
  framePosition = 4;

  $('#board-name').value = 'Orange Pi 5 Pro';
  $('#bare-files').value = '';
  $('#painted-files').value = '';
  $('#bare-files-label').textContent = '选择 CSV 文件';
  $('#painted-files-label').textContent = '选择 CSV 文件';
  $('#bare-box').classList.remove('filled');
  $('#painted-box').classList.remove('filled');
  $('#training-complete').classList.add('hidden');
  const visual = $('#training-visual'), chart = $('#training-chart');
  visual.classList.add('hidden');
  chart.classList.add('hidden');
  $('.pipeline').insertBefore(visual, $('#training-complete'));
  $('.pipeline').insertBefore(chart, $('#training-complete'));
  const empty = document.createElement('p');
  empty.className = 'log-empty';
  empty.textContent = '尚无处理记录';
  $('#pipeline-log').replaceChildren(empty);
  $('#pipeline-title').textContent = '等待导入数据';
  $('#pipeline-subtitle').textContent = '训练步骤与结果将在下方显示。';
  $('#pipeline-clock').textContent = '00:00';
  $('#pipeline-ring').style.setProperty('--progress', '0%');
  $('#pipeline-ring span').textContent = '0%';
  $('#start-training').innerHTML = '开始训练 <span>↗</span>';
  renderStages(-1);
  drawChart(0);
  checkTrainingReady();

  resetInferenceInput();
  updateBoardPicker();
  updateLibrary();
  navigate('inference');
}

function checkTrainingReady() {
  const name = $('#board-name').value.trim();
  const ready = trainingSourceReady || ($('#bare-files').files.length && $('#painted-files').files.length);
  $('#start-training').disabled = !name || !ready || !!trainingStartedAt;
  const line = $('#training-validation');
  line.classList.toggle('valid', !!ready);
  line.textContent = ready ? (trainingSourceReady ? '已载入半负载与满负载采集数据。' : '已选择采集文件。') : '请选择裸板与黑化板采集文件。';
}

function setUploadLabel(input, labelId, boxId) {
  const count = input.files.length;
  $(labelId).textContent = count ? `已选择 ${count} 个 CSV 文件` : '请选择 CSV 文件';
  $(boxId).classList.toggle('filled', count > 0);
  trainingSourceReady = false;
  checkTrainingReady();
}

function setProgress(percent) {
  $('#pipeline-ring').style.setProperty('--progress', `${percent}%`);
  $('#pipeline-ring span').textContent = `${percent}%`;
  const seconds = Math.floor((Date.now() - trainingStartedAt) / 1000);
  $('#pipeline-clock').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

function renderStages(activeIndex, complete = false) {
  $('#pipeline-stages').replaceChildren();
  stages.forEach((stage, index) => {
    const element = document.createElement('div');
    element.className = `stage ${complete || index < activeIndex ? 'done' : index === activeIndex ? 'active' : ''}`;
    element.innerHTML = `<div class="stage-line"></div><div class="stage-label">${stage.label}</div>`;
    $('#pipeline-stages').append(element);
  });
}

function createLogEntry(index) {
  const stage = stages[index], entry = document.createElement('section');
  entry.className = 'log-entry is-active';
  const head = document.createElement('div'); head.className = 'log-entry-head';
  const title = document.createElement('strong'); title.textContent = `${String(index + 1).padStart(2, '0')}  ${stage.title}`;
  const time = document.createElement('span'); time.className = 'log-entry-time'; time.textContent = '处理中';
  head.append(title, time);
  const body = document.createElement('div'); body.className = 'log-entry-body';
  const summary = document.createElement('p'); summary.textContent = stage.detail; body.append(summary);
  entry.append(head, body); $('#pipeline-log').append(entry);
  entry.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  return { entry, time, body };
}
function completeLogEntry(log, index) {
  const stage = stages[index], metrics = document.createElement('div'); metrics.className = 'log-metrics';
  stage.metrics.forEach(([value, label]) => {
    const metric = document.createElement('div'); metric.className = 'log-metric';
    const number = document.createElement('strong'); number.textContent = value;
    const caption = document.createElement('span'); caption.textContent = label;
    metric.append(number, caption); metrics.append(metric);
  });
  log.body.append(metrics);
  if (index === 1) {
    const preview = document.createElement('canvas'); preview.className = 'log-map'; preview.setAttribute('aria-label', '提取后的板体温度图');
    log.body.append(preview);
    const data = archive['orange-pi-5-pro'];
    drawMap(preview, data.paired_preview.bare, data, 25);
  }
  if (index === 3) {
    const visual = $('#training-visual'); visual.classList.remove('hidden'); log.body.append(visual);
  }
  if (index === 4) {
    const chart = $('#training-chart'); chart.classList.remove('hidden'); log.body.append(chart);
  }
  log.entry.classList.remove('is-active');
  log.time.textContent = `完成 · ${(stage.duration / 1000).toFixed(1)} 秒`;
}

function drawChart(visibleCount = 0) {
  const history = archive['orange-pi-5-pro'].training.history;
  const svg = $('#loss-chart');
  const points = history.slice(0, visibleCount);
  const x = epoch => 26 + epoch / 300 * 480;
  const y = value => 116 - (Math.min(value, 1.35) - 0.12) / (1.35 - 0.12) * 99;
  const grid = [0.2, 0.5, 0.8, 1.1].map(value => `<line x1="26" x2="506" y1="${y(value)}" y2="${y(value)}" stroke="#e7eee5" stroke-width="1"/><text x="0" y="${y(value) + 3}" fill="#a8b8a9" font-size="9">${value.toFixed(1)}</text>`).join('');
  const path = points.map((point, index) => `${index ? 'L' : 'M'} ${x(point.epoch).toFixed(1)} ${y(point.board_mae_k).toFixed(1)}`).join(' ');
  const area = points.length ? `${path} L ${x(points.at(-1).epoch).toFixed(1)} 116 L ${x(points[0].epoch).toFixed(1)} 116 Z` : '';
  const dot = points.length ? `<circle cx="${x(points.at(-1).epoch)}" cy="${y(points.at(-1).board_mae_k)}" r="4" fill="#587f5b" stroke="#fff" stroke-width="2"/>` : '';
  svg.innerHTML = `${grid}<path d="${area}" fill="#e7f3dc" opacity="0.8"/><path d="${path}" fill="none" stroke="#6e9e6c" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>${dot}`;
  $('#chart-value').textContent = points.length ? `${points.at(-1).board_mae_k.toFixed(3)} K` : '— K';
}

function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
async function startTraining() {
  const name = $('#board-name').value.trim();
  if (!name || $('#start-training').disabled) return;
  const run = ++trainingRun;
  trainingStartedAt = Date.now();
  checkTrainingReady();
  $('#training-complete').classList.add('hidden');
  const visual = $('#training-visual'), chart = $('#training-chart');
  visual.classList.add('hidden'); chart.classList.add('hidden');
  $('.pipeline').append(visual, chart);
  $('#pipeline-log').replaceChildren();
  $('#start-training').textContent = '正在训练…';
  drawChart(0);
  const total = stages.reduce((sum, stage) => sum + stage.duration, 0);
  let elapsed = 0;
  for (let index = 0; index < stages.length; index++) {
    if (run !== trainingRun) return;
    const stage = stages[index], log = createLogEntry(index);
    $('#pipeline-title').textContent = stage.title;
    $('#pipeline-subtitle').textContent = stage.subtitle;
    renderStages(index);
    if (index === 4) { const chart = $('#training-chart'); chart.classList.remove('hidden'); log.body.append(chart); }
    const start = Date.now();
    while (Date.now() - start < stage.duration) {
      if (run !== trainingRun) return;
      const fraction = Math.min(1, (Date.now() - start) / stage.duration);
      setProgress(Math.round((elapsed + fraction * stage.duration) / total * 100));
      if (index === 4) drawChart(Math.max(1, Math.round(fraction * archive['orange-pi-5-pro'].training.history.length)));
      await delay(110);
    }
    elapsed += stage.duration;
    if (index === 4) drawChart(archive['orange-pi-5-pro'].training.history.length);
    completeLogEntry(log, index);
  }
  if (run !== trainingRun) return;
  const id = `board-${Date.now()}`;
  boards.push({ id, name, base: 'orange-pi-5-pro', builtIn: false });
  saveBoards();
  activeBoardId = id;
  resetInferenceInput();
  updateBoardPicker();
  renderStages(6, true);
  setProgress(100);
  $('#pipeline-title').textContent = `${name} 已就绪`;
  $('#pipeline-subtitle').textContent = '新板型已加入板型库。';
  $('#complete-copy').textContent = `${name} 已可用于温度重构。`;
  $('#training-complete').classList.remove('hidden');
  $('#start-training').innerHTML = '再次训练 <span>↗</span>';
  trainingStartedAt = 0;
  checkTrainingReady();
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
    const file = [...event.dataTransfer.files].find(item => item.name.toLowerCase().endsWith('.csv'));
    if (file) selectInferenceFile(file);
  });
  $('#board-name').addEventListener('input', checkTrainingReady);
  $('#bare-files').addEventListener('change', event => setUploadLabel(event.target, '#bare-files-label', '#bare-box'));
  $('#painted-files').addEventListener('change', event => setUploadLabel(event.target, '#painted-files-label', '#painted-box'));
  $('#use-archive').addEventListener('click', () => {
    trainingSourceReady = true;
    $('#bare-files-label').textContent = '已载入裸板采集序列';
    $('#painted-files-label').textContent = '已载入黑化板采集序列';
    $('#bare-box').classList.add('filled'); $('#painted-box').classList.add('filled');
    checkTrainingReady();
  });
  $('#start-training').addEventListener('click', startTraining);
  $('#open-trained-board').addEventListener('click', () => { resetInferenceInput(); navigate('inference'); });
  $('#library-add').addEventListener('click', () => navigate('training'));
  $('#reset-demo').addEventListener('click', resetPresentation);
}

async function init() {
  try {
    const responses = await Promise.all(['raspberry-pi-4b', 'orange-pi-5-pro'].map(id => fetch(`assets/${id}.json`)));
    if (responses.some(response => !response.ok)) throw new Error('无法加载温度数据');
    const data = await Promise.all(responses.map(response => response.json()));
    data.forEach(item => {
      item.boardMask = unpackMask(item.mask, item.width * item.height);
      archive[item.id] = item;
    });
    connectEvents();
    updateBoardPicker(); updateInference(); updateLibrary(); renderStages(-1); drawChart(0); checkTrainingReady();
    const orange = archive['orange-pi-5-pro'];
    drawMap($('#pair-bare'), orange.paired_preview.bare, orange, 25);
    drawMap($('#pair-painted'), orange.paired_preview.painted, orange, 25);
  } catch (error) {
    $('#result-state').textContent = '数据加载失败';
    $('#inference-file-name').textContent = '请从项目目录启动本地服务器后刷新页面。';
    console.error(error);
  }
}

init();
