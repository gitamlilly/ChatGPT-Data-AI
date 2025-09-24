// app.js — Full finished version
// Data Collection & Analysis AI — Expanded & completed
// Comments explain the reasoning behind important sections and functions.

/*
  High-level:
  - Everything runs client-side so you can run locally in VS Code or via Live Server.
  - Uses Chart.js for charts (CDN in index.html).
  - Uses PapaParse if available for CSV parsing; otherwise falls back to a robust in-file parser.
  - Includes preprocessing, visualization, models, cross-validation, model export, and project save/load.
*/

// ------------------
// DOM references
// ------------------
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const pasteArea = document.getElementById('pasteArea');
const parseBtn = document.getElementById('parseBtn');
const sampleBtn = document.getElementById('sampleBtn');
const clearBtn = document.getElementById('clearBtn');
const autoCleanBtn = document.getElementById('autoCleanBtn');
const scaleBtn = document.getElementById('scaleBtn');
const encodeBtn = document.getElementById('encodeBtn');
const targetSelect = document.getElementById('targetSelect');
const featuresSelect = document.getElementById('featuresSelect');
const trainLRBtn = document.getElementById('trainLRBtn');
const trainLogisticBtn = document.getElementById('trainLogisticBtn');
const kmeansBtn = document.getElementById('kmeansBtn');
const exportCsvBtn = document.getElementById('exportCsvBtn');
const exportModelBtn = document.getElementById('exportModelBtn');
const histColSelect = document.getElementById('histColSelect');
const histChartCanvas = document.getElementById('histChart');
const boxChartCanvas = document.getElementById('boxChart');
const xSelect = document.getElementById('xSelect');
const ySelect = document.getElementById('ySelect');
const scatterChartCanvas = document.getElementById('scatterChart');
const corrCanvas = document.getElementById('corrCanvas');
const pcaChartCanvas = document.getElementById('pcaChart');
const dataTableDiv = document.getElementById('dataTable');
const summaryPre = document.getElementById('summaryPre');
const suggestionsList = document.getElementById('suggestionsList');
const modelPre = document.getElementById('modelPre');

// ------------------
// App state
// ------------------
let rawData = [];       // parsed raw rows (objects)
let workingData = [];   // transformed rows (objects)
let columns = [];       // ordered column names
let colTypes = {};      // inferred types
let models = {};        // store trained models for export
let chartInstances = {}; // Chart.js instances keyed by id

// ------------------
// Utilities
// ------------------
function isMissing(v) {
  return v === null || v === undefined || String(v).trim() === '' ||
         ['na','n/a','null','undefined'].includes(String(v).toLowerCase());
}
function toNumberIfPossible(v) {
  const n = parseFloat(String(v).replace(/[^0-9eE+.-]/g, ''));
  return Number.isFinite(n) ? n : NaN;
}
function unique(arr) {
  return Array.from(new Set(arr));
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c]));
}

// ------------------
// CSV parsing
// - prefer Papa.parse if loaded; fallback to a robust in-file parser that handles quotes/newlines
// ------------------
function parseCSVtext(text) {
  if (!text) return [];
  if (window.Papa && typeof window.Papa.parse === 'function') {
    // Use PapaParse for robust parsing when available
    const parsed = Papa.parse(text.trim(), { header: true, skipEmptyLines: true });
    return parsed.data;
  }
  // Fallback CSV parser (handles quoted fields, commas/newlines inside quotes)
  const rows = [];
  let cur = '';
  let row = [];
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const nch = text[i+1];
    if (ch === '"') {
      if (inQuotes && nch === '"') { cur += '"'; i++; } else { inQuotes = !inQuotes; }
    } else if (ch === ',' && !inQuotes) {
      row.push(cur);
      cur = '';
    } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && nch === '\n') { /* we'll handle on next loop */ }
      row.push(cur);
      // avoid pushing fully empty rows
      if (!(row.length === 1 && row[0] === '')) rows.push(row);
      row = [];
      cur = '';
    } else {
      cur += ch;
    }
  }
  if (cur !== '' || row.length > 0) {
    row.push(cur);
    if (!(row.length === 1 && row[0] === '')) rows.push(row);
  }

  if (rows.length === 0) return [];
  const headers = rows[0].map(h => h.trim());
  const data = [];
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const obj = {};
    for (let j = 0; j < headers.length; j++) {
      obj[headers[j]] = r[j] !== undefined ? r[j].trim() : '';
    }
    data.push(obj);
  }
  return data;
}

// ------------------
// Type inference & summarization
// ------------------
function inferColumnTypes(sampleData, threshold=0.8) {
  if (!sampleData || sampleData.length === 0) return {};
  const cols = Object.keys(sampleData[0]);
  const types = {};
  for (const col of cols) {
    const vals = sampleData.map(r => r[col]);
    let numericCount = 0, count = 0;
    for (const v of vals) {
      if (isMissing(v)) continue;
      count++;
      if (!Number.isNaN(toNumberIfPossible(v))) numericCount++;
    }
    types[col] = (count === 0) ? 'categorical' : (numericCount / count >= threshold ? 'numeric' : 'categorical');
  }
  return types;
}

function summarizeColumn(values, type) {
  const cleaned = values.filter(v => !isMissing(v));
  if (type === 'numeric') {
    const nums = cleaned.map(v=>toNumberIfPossible(v)).filter(n=>!Number.isNaN(n));
    if (nums.length === 0) return {count:0};
    nums.sort((a,b)=>a-b);
    const sum = nums.reduce((a,b)=>a+b,0);
    const mean = sum / nums.length;
    const median = nums.length % 2 === 1 ? nums[(nums.length-1)/2] : (nums[nums.length/2-1] + nums[nums.length/2])/2;
    const sq = nums.map(n=>Math.pow(n-mean,2));
    const variance = sq.reduce((a,b)=>a+b,0)/(nums.length-1 || 1);
    const std = Math.sqrt(variance);
    const q1 = nums[Math.floor((nums.length-1)/4)];
    const q3 = nums[Math.ceil((nums.length-1)*3/4)];
    return {count: nums.length, mean, median, std, min: nums[0], max: nums[nums.length-1], q1, q3};
  } else {
    const freq = {};
    for (const v of cleaned) freq[v] = (freq[v] || 0) + 1;
    const entries = Object.entries(freq).sort((a,b)=>b[1]-a[1]);
    return {count: cleaned.length, unique: Object.keys(freq).length, top: entries[0] ? {value: entries[0][0], count: entries[0][1]} : null};
  }
}

// ------------------
// Small matrix utilities (for linear algebra used in regression & PCA)
// - matMul (matrix multiplication), transpose, matInverse (Gauss-Jordan). These are simple and intended for small matrices.
// ------------------
function transpose(A) {
  return A[0].map((_, c) => A.map(r => r[c]));
}
function matMul(A, B) { // A(mxp) * B(p x n) = m x n
  const m = A.length, p = A[0].length, n = B[0].length;
  const out = Array.from({length: m}, () => Array(n).fill(0));
  for (let i=0;i<m;i++){
    for (let k=0;k<p;k++){
      for (let j=0;j<n;j++){
        out[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return out;
}
function matInverse(A) {
  const n = A.length;
  // build augmented
  const M = A.map((r,i) => r.concat(Array.from({length:n}, (_,j) => i===j?1:0)));
  for (let i=0;i<n;i++) {
    // pivot
    let pivot = i;
    for (let j=i;j<n;j++) if (Math.abs(M[j][i]) > Math.abs(M[pivot][i])) pivot = j;
    if (Math.abs(M[pivot][i]) < 1e-12) throw new Error('Matrix singular or nearly singular');
    if (pivot !== i) { const tmp = M[i]; M[i] = M[pivot]; M[pivot] = tmp; }
    const div = M[i][i];
    for (let k=0;k<2*n;k++) M[i][k] /= div;
    for (let j=0;j<n;j++) if (j !== i) {
      const factor = M[j][i];
      for (let k=0;k<2*n;k++) M[j][k] -= factor * M[i][k];
    }
  }
  return M.map(r => r.slice(n));
}

// ------------------
// UI helpers
// ------------------
function fillSelect(sel, options, includeBlank=false) {
  sel.innerHTML = (includeBlank? '<option value="">— select —</option>' : '') +
                  options.map(o => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join('');
}
function fillMultiSelect(sel, options) {
  sel.innerHTML = options.map(o => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join('');
}
function renderTable(data) {
  if (!data || data.length === 0) { dataTableDiv.innerHTML = '<div class="note">No data to display</div>'; return; }
  const cols = Object.keys(data[0]);
  const thead = `<thead><tr>${cols.map(c=>`<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>`;
  const rows = data.map(r => `<tr>${cols.map(c=>`<td>${escapeHtml(String(r[c]===undefined? '': r[c]))}</td>`).join('')}</tr>`).join('');
  dataTableDiv.innerHTML = `<table class="table">${thead}<tbody>${rows}</tbody></table>`;
}

// ------------------
// Charts: histogram+box, scatter with optional regression overlay, correlation heatmap, PCA scatter
// ------------------
function destroyChart(id) {
  if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

function drawHistogramAndBox(column) {
  if (!column) return;
  const values = workingData.map(r => r[column]).filter(v => !isMissing(v)).map(v => toNumberIfPossible(v)).filter(n => !Number.isNaN(n));
  destroyChart('hist'); destroyChart('box');
  if (values.length === 0) return;
  // histogram
  const bins = 12;
  const min = Math.min(...values), max = Math.max(...values);
  const width = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  const labels = [];
  for (let i=0;i<bins;i++) labels.push(`${(min + i*width).toFixed(2)} - ${(min + (i+1)*width).toFixed(2)}`);
  for (const v of values) {
    const idx = Math.min(bins-1, Math.floor((v - min)/width));
    counts[idx]++;
  }
  chartInstances['hist'] = new Chart(histChartCanvas.getContext('2d'), {
    type: 'bar',
    data: { labels, datasets: [{ label: column, data: counts }] },
    options: { responsive: true, maintainAspectRatio: false }
  });

  // boxplot-like: we draw a simple IQR bar; Chart.js doesn't provide native boxplot without plugins
  const nums = values.slice().sort((a,b)=>a-b);
  const q1 = nums[Math.floor((nums.length-1)/4)];
  const median = nums.length % 2 ? nums[(nums.length-1)/2] : (nums[nums.length/2-1]+nums[nums.length/2])/2;
  const q3 = nums[Math.ceil((nums.length-1)*3/4)];
  const iqr = q3 - q1;
  destroyChart('box');
  chartInstances['box'] = new Chart(boxChartCanvas.getContext('2d'), {
    type: 'bar',
    data: { labels: [column], datasets: [{ label: 'IQR', data: [q3 - q1] }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { display: false } }, responsive: true, maintainAspectRatio: false }
  });
}

function drawScatterWithRegression(xCol, yCol, regression=null) {
  if (!xCol || !yCol) return;
  destroyChart('scatter');
  const pts = workingData.map(r => ({ x: toNumberIfPossible(r[xCol]), y: toNumberIfPossible(r[yCol]) }))
                         .filter(p => !Number.isNaN(p.x) && !Number.isNaN(p.y));
  chartInstances['scatter'] = new Chart(scatterChartCanvas.getContext('2d'), {
    type: 'scatter',
    data: { datasets: [{ label: `${yCol} vs ${xCol}`, data: pts, pointRadius: 4 }] },
    options: { scales: { x: { title: { display: true, text: xCol } }, y: { title: { display: true, text: yCol } } }, plugins: { legend: { display: false } }, responsive: true, maintainAspectRatio: false }
  });

  if (regression && pts.length > 1) {
    // draw line using chart's pixel transforms
    const chart = chartInstances['scatter'];
    const xScale = chart.scales.x, yScale = chart.scales.y;
    const xs = pts.map(p => p.x);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    let y1, y2;
    if (typeof regression.predictOne === 'function') {
      y1 = regression.predictOne([xmin]);
      y2 = regression.predictOne([xmax]);
    } else if (Array.isArray(regression.coefficients) && typeof regression.intercept !== 'undefined') {
      // simple single-feature regression fallback
      const a = regression.coefficients[0];
      const b = regression.intercept;
      y1 = b + a * xmin; y2 = b + a * xmax;
    } else {
      return;
    }
    const ctx = scatterChartCanvas.getContext('2d');
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 2;
    ctx.moveTo(xScale.getPixelForValue(xmin), yScale.getPixelForValue(y1));
    ctx.lineTo(xScale.getPixelForValue(xmax), yScale.getPixelForValue(y2));
    ctx.stroke();
    ctx.restore();
  }
}

function drawCorrelationHeatmap() {
  destroyChart('corr');
  const numericCols = columns.filter(c => colTypes[c] === 'numeric');
  if (numericCols.length === 0) return;
  const n = numericCols.length;
  const matrix = Array.from({length: n}, () => Array(n).fill(0));
  for (let i=0;i<n;i++) {
    for (let j=0;j<n;j++) {
      const a = workingData.map(r => toNumberIfPossible(r[numericCols[i]]));
      const b = workingData.map(r => toNumberIfPossible(r[numericCols[j]]));
      const paired = a.map((v,k)=>({a:v, b:b[k]})).filter(p => !Number.isNaN(p.a) && !Number.isNaN(p.b));
      if (paired.length < 2) matrix[i][j] = 0;
      else {
        const meanA = paired.reduce((s,p)=>s+p.a,0)/paired.length;
        const meanB = paired.reduce((s,p)=>s+p.b,0)/paired.length;
        const num = paired.reduce((s,p)=>s + (p.a - meanA)*(p.b - meanB), 0);
        const denA = Math.sqrt(paired.reduce((s,p)=>s + Math.pow(p.a - meanA,2),0));
        const denB = Math.sqrt(paired.reduce((s,p)=>s + Math.pow(p.b - meanB,2),0));
        matrix[i][j] = num / (denA * denB || 1);
      }
    }
  }

  // draw onto canvas
  const canvas = corrCanvas;
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth, h = canvas.clientHeight;
  // scale for high DPI
  canvas.width = w * 2; canvas.height = h * 2; ctx.scale(2,2);
  const cellW = w / n, cellH = h / n;
  // background
  ctx.clearRect(0,0,w,h);
  for (let i=0;i<n;i++) {
    for (let j=0;j<n;j++) {
      const v = matrix[i][j]; // -1..1
      // map to color
      const r = Math.round((v + 1) / 2 * 255);
      const b = Math.round((1 - (v + 1) / 2) * 255);
      const g = Math.round((1 - Math.abs(v)) * 255);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j*cellW, i*cellH, cellW, cellH);
      // text
      ctx.fillStyle = 'black';
      ctx.font = '10px sans-serif';
      const txt = v.toFixed(2);
      ctx.fillText(txt, j*cellW + 4, i*cellH + 12);
    }
  }
  // labels on left just to help readability
  ctx.fillStyle = 'white';
  ctx.font = '12px sans-serif';
  for (let i=0;i<n;i++) ctx.fillText(numericCols[i], 4, i*cellH + cellH/2);
}

//
// PCA: compute covariance, power iterate for eigenvectors (small-scale, educational approach).
//
function computePCA(dataMatrix, k=2) {
  const n = dataMatrix.length;
  if (n === 0) return null;
  const p = dataMatrix[0].length;
  // covariance
  const C = Array.from({length:p}, () => Array(p).fill(0));
  for (let i=0;i<p;i++) {
    for (let j=0;j<p;j++) {
      let s = 0;
      for (let r=0;r<n;r++) s += dataMatrix[r][i] * dataMatrix[r][j];
      C[i][j] = s / (n - 1 || 1);
    }
  }
  function powerIteration(A, iter=500) {
    let v = Array.from({length: A.length}, () => Math.random());
    // normalize
    let norm = Math.sqrt(v.reduce((s,x)=>s + x*x, 0));
    v = v.map(x => x / (norm || 1));
    for (let it=0; it<iter; it++) {
      const Av = A.map(row => row.reduce((s,val,j) => s + val * v[j], 0));
      const anorm = Math.sqrt(Av.reduce((s,x)=>s + x*x, 0));
      v = Av.map(x => x / (anorm || 1));
    }
    // eigenvalue approx
    const Av = A.map(row => row.reduce((s,val,j) => s + val * v[j], 0));
    const lambda = v.reduce((s,vi,i) => s + vi * Av[i], 0);
    return {vector: v, value: lambda};
  }
  const comps = [];
  const Acopy = C.map(r => r.slice());
  for (let comp=0; comp<k; comp++) {
    const {vector} = powerIteration(Acopy, 500);
    // minor deflation so next iteration finds different vector
    for (let i=0;i<p;i++) for (let j=0;j<p;j++) Acopy[i][j] -= vector[i] * vector[j] * 1e-6;
    comps.push(vector);
  }
  // projection: n x k
  const projection = dataMatrix.map(row => comps.map(c => row.reduce((s,v,i) => s + v * row[i], 0)));
  return {components: comps, projection};
}

function drawPCA() {
  const numericCols = columns.filter(c => colTypes[c] === 'numeric');
  if (numericCols.length < 2) return;
  const matrix = workingData.map(r => numericCols.map(c => toNumberIfPossible(r[c]))).map(row => row.map(v => Number.isNaN(v) ? 0 : v));
  const p = numericCols.length, n = matrix.length;
  const means = Array(p).fill(0);
  if (n === 0) return;
  for (let j=0;j<p;j++) means[j] = matrix.reduce((s,row)=>s + row[j], 0) / n;
  const centered = matrix.map(row => row.map((v,j) => v - means[j]));
  const pca = computePCA(centered, 2);
  if (!pca) return;
  destroyChart('pca');
  const pts = pca.projection.map(v => ({ x: v[0], y: v[1] }));
  chartInstances['pca'] = new Chart(pcaChartCanvas.getContext('2d'), {
    type: 'scatter',
    data: { datasets: [{ label: 'PCA (2D)', data: pts, pointRadius: 4 }] },
    options: { plugins: { legend: { display: false } }, responsive: true, maintainAspectRatio: false }
  });
}

// ------------------
// Preprocessing functions
// ------------------
function autoClean() {
  if (!workingData || workingData.length === 0) return;
  const cols = Object.keys(workingData[0]);
  for (const col of cols) {
    if (colTypes[col] === 'numeric') {
      const nums = workingData.map(r => toNumberIfPossible(r[col])).filter(n => !Number.isNaN(n));
      const mean = nums.reduce((a,b)=>a+b,0) / (nums.length || 1);
      for (const r of workingData) {
        if (isMissing(r[col]) || Number.isNaN(toNumberIfPossible(r[col]))) r[col] = mean;
      }
    } else {
      for (const r of workingData) {
        if (r[col] === undefined || r[col] === null) r[col] = '';
        r[col] = String(r[col]).trim();
      }
    }
  }
  postProcessState();
}

function scaleNumericColumns() {
  const numericCols = columns.filter(c => colTypes[c] === 'numeric');
  for (const col of numericCols) {
    const vals = workingData.map(r => toNumberIfPossible(r[col])).map(n => Number.isNaN(n) ? 0 : n);
    const mean = vals.reduce((a,b)=>a+b,0) / vals.length;
    const std = Math.sqrt(vals.map(v => Math.pow(v-mean,2)).reduce((a,b)=>a+b,0) / (vals.length - 1 || 1));
    for (const r of workingData) r[col] = (toNumberIfPossible(r[col]) - mean) / (std || 1);
  }
  postProcessState();
}

function oneHotEncodeCategorical() {
  const catCols = columns.filter(c => colTypes[c] === 'categorical');
  for (const col of catCols) {
    const vals = unique(workingData.map(r => r[col]));
    if (vals.length > 50) { console.warn(`Skipping one-hot for ${col} — high cardinality (${vals.length})`); continue; }
    for (const v of vals) {
      const key = `${col}__${v}`;
      for (const r of workingData) r[key] = (r[col] === v) ? 1 : 0;
    }
    // drop original
    for (const r of workingData) delete r[col];
  }
  postProcessState();
}

// ------------------
// Modeling
//  - Linear regression closed form
//  - Expanded linear regression with train/test, CV, Ridge
//  - Logistic regression (GD)
//  - k-means
// ------------------
function linearRegressionClosedForm(featureCols, targetCol) {
  const X = [], y = [];
  for (const r of workingData) {
    const row = [1]; let ok = true;
    for (const c of featureCols) {
      const v = toNumberIfPossible(r[c]);
      if (Number.isNaN(v)) { ok = false; break; }
      row.push(v);
    }
    const tv = toNumberIfPossible(r[targetCol]);
    if (Number.isNaN(tv)) ok = false;
    if (ok) { X.push(row); y.push([tv]); }
  }
  if (X.length === 0) throw new Error('No valid rows for training');
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const XtXinv = matInverse(XtX);
  const XtY = matMul(Xt, y);
  const betaMat = matMul(XtXinv, XtY);
  const coeffs = betaMat.map(r => r[0]);
  const model = { type: 'linear', intercept: coeffs[0], coefficients: coeffs.slice(1), featureCols };

  // metrics
  const preds = X.map(row => coeffs.reduce((s,b,i) => s + b * row[i], 0));
  const yFlat = y.map(r => r[0]);
  const ssRes = preds.map((p,i)=>Math.pow(p - yFlat[i],2)).reduce((a,b)=>a+b,0);
  const meanY = yFlat.reduce((a,b)=>a+b,0) / yFlat.length;
  const ssTot = yFlat.map(v=>Math.pow(v-meanY,2)).reduce((a,b)=>a+b,0) || 1;
  const r2 = 1 - ssRes / ssTot;
  const rmse = Math.sqrt(ssRes / yFlat.length);
  return {...model, r2, rmse};
}

// Expanded linear regression: CV, Ridge, train/test
function fitLinearRegressionExpanded(featureCols, targetCol, options = { regularization: 0, cvFolds: 0, testFraction: 0.2 }) {
  // gather valid rows
  const rows = [];
  for (const r of workingData) {
    const x = [];
    let ok = true;
    for (const c of featureCols) {
      const v = toNumberIfPossible(r[c]);
      if (Number.isNaN(v)) { ok = false; break; }
      x.push(v);
    }
    const yv = toNumberIfPossible(r[targetCol]);
    if (Number.isNaN(yv)) ok = false;
    if (ok) rows.push({ x, y: yv });
  }
  if (rows.length === 0) throw new Error('No valid rows for regression training');

  function splitRows(rows, testFrac) {
    const shuffled = rows.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const cut = Math.floor(shuffled.length * (1 - testFrac));
    return { train: shuffled.slice(0, cut), test: shuffled.slice(cut) };
  }

  function fitClosedForm(rRows, lambda = 0) {
    const X = rRows.map(r => [1, ...r.x]);
    const y = rRows.map(r => [r.y]);
    const Xt = transpose(X);
    const XtX = matMul(Xt, X);
    if (lambda && lambda > 0) {
      for (let i=1;i<XtX.length;i++) XtX[i][i] += lambda;
    }
    const XtXinv = matInverse(XtX);
    const XtY = matMul(Xt, y);
    const beta = matMul(XtXinv, XtY).map(r => r[0]);
    return beta;
  }

  function metricsFromPreds(yTrue, yPred) {
    const n = yTrue.length;
    const ssRes = yTrue.reduce((s,v,i) => s + Math.pow(v - yPred[i], 2), 0);
    const mean = yTrue.reduce((s,v) => s + v, 0) / n;
    const ssTot = yTrue.reduce((s,v) => s + Math.pow(v - mean, 2), 0) || 1;
    const r2 = 1 - ssRes / ssTot;
    const rmse = Math.sqrt(ssRes / n);
    return { r2, rmse };
  }

  const results = {};
  if (options.cvFolds && options.cvFolds > 1) {
    const k = options.cvFolds;
    const foldSize = Math.floor(rows.length / k);
    const allCoefs = [];
    const allMetrics = [];
    for (let i=0;i<k;i++) {
      const testStart = i * foldSize;
      const testEnd = (i+1) * foldSize;
      const test = rows.slice(testStart, testEnd);
      const train = rows.slice(0, testStart).concat(rows.slice(testEnd));
      if (train.length === 0 || test.length === 0) continue;
      const beta = fitClosedForm(train, options.regularization || 0);
      const yPred = test.map(r => beta.reduce((s,b,idx) => s + b * (idx===0 ? 1 : r.x[idx-1]), 0));
      const yTrue = test.map(r => r.y);
      allCoefs.push(beta);
      allMetrics.push(metricsFromPreds(yTrue, yPred));
    }
    // aggregate coefficients (mean) and metrics
    const meanCoefs = allCoefs[0].map((_,i) => allCoefs.reduce((s,c) => s + c[i], 0) / allCoefs.length);
    const meanR2 = allMetrics.reduce((s,m) => s + m.r2, 0) / allMetrics.length;
    const meanRMSE = allMetrics.reduce((s,m) => s + m.rmse, 0) / allMetrics.length;
    results.coefficients = meanCoefs.slice(1);
    results.intercept = meanCoefs[0];
    results.r2 = meanR2;
    results.rmse = meanRMSE;
    results.cv = true;
    results.folds = allMetrics.length;
    return results;
  } else {
    const split = splitRows(rows, options.testFraction || 0.2);
    const beta = fitClosedForm(split.train, options.regularization || 0);
    const predict = (beta, xRow) => beta.reduce((s,b,i) => s + b * (i===0 ? 1 : xRow[i-1]), 0);
    const yPredTrain = split.train.map(r => predict(beta, r.x));
    const yTrain = split.train.map(r => r.y);
    const yPredTest = split.test.map(r => predict(beta, r.x));
    const yTest = split.test.map(r => r.y);
    results.intercept = beta[0];
    results.coefficients = beta.slice(1);
    results.trainMetrics = metricsFromPreds(yTrain, yPredTrain);
    results.testMetrics = metricsFromPreds(yTest, yPredTest);
    results.nTrain = split.train.length;
    results.nTest = split.test.length;
    return results;
  }
}

// Logistic regression (binary) using gradient descent
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function trainLogisticRegression(featureCols, targetCol, opts = { lr: 0.1, epochs: 1000, lambda: 0 }) {
  const X = [];
  const y = [];
  // attempt to coerce or map categorical binary targets
  const targetVals = unique(workingData.map(r => r[targetCol]).filter(v => !isMissing(v)));
  let mapTo01 = null;
  if (targetVals.length === 2) {
    mapTo01 = {};
    mapTo01[targetVals[0]] = 0; mapTo01[targetVals[1]] = 1;
  }

  for (const r of workingData) {
    const row = [1]; let ok = true;
    for (const c of featureCols) {
      const v = toNumberIfPossible(r[c]);
      if (Number.isNaN(v)) { ok = false; break; }
      row.push(v);
    }
    if (!ok) continue;
    const tv = r[targetCol];
    if (isMissing(tv)) continue;
    let yv = null;
    if (mapTo01) yv = mapTo01[tv];
    else if (tv === 0 || tv === '0') yv = 0;
    else if (tv === 1 || tv === '1') yv = 1;
    else continue;
    X.push(row); y.push(yv);
  }
  if (X.length === 0) throw new Error('No valid rows for logistic training');

  const m = X.length;
  const n = X[0].length;
  let theta = Array(n).fill(0);

  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    const grads = Array(n).fill(0);
    let loss = 0;
    for (let i=0;i<m;i++) {
      const xi = X[i];
      const z = xi.reduce((s,v,j) => s + v * theta[j], 0);
      const h = sigmoid(z);
      const err = h - y[i];
      for (let j=0;j<n;j++) grads[j] += err * xi[j];
      loss += - (y[i] * Math.log(h + 1e-12) + (1 - y[i]) * Math.log(1 - h + 1e-12));
    }
    for (let j=0;j<n;j++) theta[j] -= (opts.lr / m) * (grads[j] + opts.lambda * theta[j]);
    // optionally monitor loss; avoid spamming console
    if (epoch % Math.max(1, Math.floor(opts.epochs/5)) === 0) {
      //console.log(`logistic epoch ${epoch} loss ${(loss/m).toFixed(4)}`);
    }
  }
  // evaluate
  let correct = 0;
  for (let i=0;i<m;i++) {
    const z = X[i].reduce((s,v,j) => s + v * theta[j], 0);
    const p = sigmoid(z);
    const pred = p >= 0.5 ? 1 : 0;
    if (pred === y[i]) correct++;
  }
  const acc = correct / m;
  const model = { type: 'logistic', theta, featureCols, accuracy: acc };
  return model;
}

// k-means
function kMeans(k = 3, featureCols = [], maxIter = 100) {
  const X = workingData.map(r => featureCols.map(c => toNumberIfPossible(r[c]))).filter(row => row.every(v => !Number.isNaN(v)));
  if (X.length === 0) throw new Error('No valid numeric rows for k-means');
  // init centroids
  const centroids = [];
  const used = new Set();
  while (centroids.length < k) {
    const idx = Math.floor(Math.random() * X.length);
    if (!used.has(idx)) { used.add(idx); centroids.push(X[idx].slice()); }
  }
  let changed = true, iter = 0;
  let assignments = new Array(X.length).fill(-1);
  while (changed && iter < maxIter) {
    iter++; changed = false;
    for (let i=0;i<X.length;i++) {
      let best = -1, bestd = Infinity;
      for (let c=0;c<k;c++) {
        const d = euclidean(X[i], centroids[c]);
        if (d < bestd) { bestd = d; best = c; }
      }
      if (assignments[i] !== best) { changed = true; assignments[i] = best; }
    }
    for (let c=0;c<k;c++) {
      const members = X.filter((_,i) => assignments[i] === c);
      if (members.length > 0) {
        const mean = members[0].map(()=>0);
        for (const m of members) for (let j=0;j<m.length;j++) mean[j] += m[j];
        for (let j=0;j<mean.length;j++) mean[j] /= members.length;
        centroids[c] = mean;
      }
    }
  }
  return { centroids, assignments };
}
function euclidean(a,b) { let s=0; for (let i=0;i<a.length;i++) s += Math.pow(a[i] - b[i], 2); return Math.sqrt(s); }

// ------------------
// Suggestions (rule-based EDA hints)
// ------------------
function generateSuggestions() {
  const suggestions = [];
  if (!workingData || workingData.length === 0) return suggestions;
  const n = workingData.length;
  for (const col of columns) {
    const stats = summarizeColumn(workingData.map(r=>r[col]), colTypes[col]);
    if (colTypes[col] === 'numeric') {
      if (stats.count < n * 0.95) suggestions.push(`Column '${col}' has ${n - stats.count} missing values — consider imputation.`);
      if (stats.std && Math.abs(stats.std) > Math.abs(stats.mean) * 2) suggestions.push(`Column '${col}' has high variance relative to mean — consider scaling.`);
    } else {
      if (stats.unique > Math.min(50, n/2)) suggestions.push(`Column '${col}' is high-cardinality categorical (unique=${stats.unique}) — one-hot may be expensive.`);
    }
  }
  const numericCols = columns.filter(c => colTypes[c] === 'numeric');
  if (numericCols.length >= 1) suggestions.push('Try PCA for dimensionality reduction or k-means for clustering.');
  else suggestions.push('No numeric columns detected for regression — consider numeric features or encode categoricals.');
  return suggestions;
}

// ------------------
// Export functions
// ------------------
function exportCSV() {
  if (!workingData || workingData.length === 0) return;
  const cols = Object.keys(workingData[0]);
  const rows = [cols.join(',')];
  for (const r of workingData) rows.push(cols.map(c => `"${String(r[c]).replace(/"/g, '""')}"`).join(','));
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'export.csv'; a.click(); URL.revokeObjectURL(url);
}
function exportModel() {
  const blob = new Blob([JSON.stringify(models, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'models.json'; a.click(); URL.revokeObjectURL(url);
}

// ------------------
// State update and UI refresh
// ------------------
function postProcessState() {
  if (!workingData || workingData.length === 0) {
    columns = []; colTypes = {};
    summaryPre.textContent = '(no data)';
    suggestionsList.innerHTML = '<li>No data</li>';
    renderTable([]);
    // destroy charts
    Object.keys(chartInstances).forEach(k => destroyChart(k));
    return;
  }
  columns = Object.keys(workingData[0]);
  colTypes = inferColumnTypes(workingData);
  fillSelect(targetSelect, columns, true);
  fillMultiSelect(featuresSelect, columns);
  fillSelect(histColSelect, columns);
  fillSelect(xSelect, columns);
  fillSelect(ySelect, columns);

  const lines = [`Rows: ${workingData.length}`];
  for (const col of columns) {
    lines.push(`${col} (${colTypes[col]}): ${JSON.stringify(summarizeColumn(workingData.map(r => r[col]), colTypes[col]))}`);
  }
  summaryPre.textContent = lines.join('\n');

  const sugg = generateSuggestions();
  suggestionsList.innerHTML = (sugg.length ? sugg.map(t => `<li>${escapeHtml(t)}</li>`).join('') : '<li>No suggestions</li>');

  // visuals
  const histDefault = columns.find(c => colTypes[c] === 'numeric') || columns[0];
  drawHistogramAndBox(histColSelect.value || histDefault);
  const x0 = columns[0], y0 = columns[1] || columns[0];
  drawScatterWithRegression(xSelect.value || x0, ySelect.value || y0);
  drawCorrelationHeatmap();
  drawPCA();
  renderTable(workingData.slice(0, 200));
}

// ------------------
// Event wiring
// ------------------
fileInput.addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = ev => { rawData = parseCSVtext(ev.target.result); workingData = rawData.map(r => ({...r})); postProcessState(); };
  reader.readAsText(f);
});

// drag & drop
['dragenter','dragover'].forEach(ev => dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.style.borderColor = '#7dd3fc'; }));
['dragleave','drop'].forEach(ev => dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.style.borderColor = ''; }));
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = ev => { rawData = parseCSVtext(ev.target.result); workingData = rawData.map(r => ({...r})); postProcessState(); };
  reader.readAsText(f);
});

parseBtn.addEventListener('click', () => {
  const txt = pasteArea.value;
  if (!txt.trim()) return alert('Paste CSV text first');
  rawData = parseCSVtext(txt);
  workingData = rawData.map(r => ({...r}));
  postProcessState();
});

sampleBtn.addEventListener('click', () => {
  const sample = `name,age,years_experience,salary,department,senior
Alice,34,10,85000,Engineering,yes
Bob,28,4,56000,Product,no
Carol,45,20,130000,Engineering,yes
Dan,22,1,42000,Support,no
Eve,39,12,98000,Product,yes
Frank,29,3,48000,Support,no
Grace,31,6,72000,Engineering,no`;
  rawData = parseCSVtext(sample);
  workingData = rawData.map(r => ({...r}));
  pasteArea.value = sample;
  postProcessState();
});

clearBtn.addEventListener('click', () => { rawData = []; workingData = []; postProcessState(); pasteArea.value = ''; });

autoCleanBtn.addEventListener('click', () => { autoClean(); alert('Auto-clean applied (impute numeric with mean; trim strings)'); });
scaleBtn.addEventListener('click', () => { scaleNumericColumns(); alert('Numeric columns z-scored'); });
encodeBtn.addEventListener('click', () => { oneHotEncodeCategorical(); alert('One-hot encoding applied (skips high-cardinality)'); });

histColSelect.addEventListener('change', () => drawHistogramAndBox(histColSelect.value));
xSelect.addEventListener('change', () => drawScatterWithRegression(xSelect.value, ySelect.value));
ySelect.addEventListener('change', () => drawScatterWithRegression(xSelect.value, ySelect.value));

trainLRBtn.addEventListener('click', () => {
  const target = targetSelect.value; const features = Array.from(featuresSelect.selectedOptions).map(o => o.value);
  if (!target) return alert('Choose a target');
  if (!features.length) return alert('Choose features');
  try {
    const model = linearRegressionClosedForm(features, target);
    models['linear_' + Date.now()] = model;
    modelPre.textContent = JSON.stringify(model, null, 2);
    postProcessState();
  } catch (err) {
    modelPre.textContent = 'Training failed: ' + err.message;
  }
});

trainLogisticBtn.addEventListener('click', () => {
  const target = targetSelect.value; const features = Array.from(featuresSelect.selectedOptions).map(o => o.value);
  if (!target) return alert('Choose a target');
  if (!features.length) return alert('Choose features');
  try {
    const model = trainLogisticRegression(features, target, { lr: 0.5, epochs: 300 });
    models['logistic_' + Date.now()] = model;
    modelPre.textContent = JSON.stringify(model, null, 2);
  } catch (err) {
    modelPre.textContent = 'Training failed: ' + err.message;
  }
});

kmeansBtn.addEventListener('click', () => {
  const k = parseInt(prompt('Number of clusters k', '3') || '3');
  const features = Array.from(featuresSelect.selectedOptions).map(o => o.value);
  if (!features.length) return alert('Choose numeric features for k-means');
  try {
    const out = kMeans(k, features, 100);
    models['kmeans_' + Date.now()] = out;
    modelPre.textContent = JSON.stringify(out, null, 2);
    postProcessState();
  } catch (err) {
    modelPre.textContent = 'k-means failed: ' + err.message;
  }
});

exportCsvBtn.addEventListener('click', () => exportCSV());
exportModelBtn.addEventListener('click', () => exportModel());

// ------------------
// Additional UI: Advanced Regression controls (added dynamically so HTML unchanged)
// - Lambda (Ridge), CV folds, test fraction, and feature importance chart
// ------------------
(function addAdvancedControls(){
  const panel = document.createElement('div');
  panel.style.marginTop = '8px';
  panel.innerHTML = `
    <h4>Advanced Regression</h4>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <label>Regularization (Ridge λ): <input id="ridgeLambda" type="number" value="0" step="0.1" min="0" style="width:100px"></label>
      <label>CV Folds: <input id="cvFolds" type="number" value="0" step="1" min="0" style="width:80px"></label>
      <label>Test fraction: <input id="testFrac" type="number" value="0.2" step="0.05" min="0" max="0.9" style="width:80px"></label>
      <button id="runExpandedLR">Run Expanded LR</button>
    </div>
    <canvas id="featureImportance" height="160" style="margin-top:8px"></canvas>
  `;
  const left = document.getElementById('left-panel');
  left.appendChild(panel);
  document.getElementById('runExpandedLR').addEventListener('click', () => {
    const features = Array.from(featuresSelect.selectedOptions).map(o => o.value);
    const target = targetSelect.value;
    if (!target) return alert('Choose a target');
    if (features.length === 0) return alert('Choose features');
    const lambda = parseFloat(document.getElementById('ridgeLambda').value) || 0;
    const cv = parseInt(document.getElementById('cvFolds').value) || 0;
    const testFrac = parseFloat(document.getElementById('testFrac').value) || 0.2;
    try {
      const res = fitLinearRegressionExpanded(features, target, { regularization: lambda, cvFolds: cv, testFraction: testFrac });
      models['linear_exp_' + Date.now()] = res;
      modelPre.textContent = JSON.stringify(res, null, 2);
      // feature importance: absolute coefficients
      const labels = features;
      const values = (res.coefficients || []).map(Math.abs);
      destroyChart('featImp');
      const ctx = document.getElementById('featureImportance').getContext('2d');
      chartInstances['featImp'] = new Chart(ctx, { type: 'bar', data: { labels, datasets: [{ label: '|coef|', data: values }] }, options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false } });
    } catch (err) {
      alert('LR failed: ' + err.message);
    }
  });
})();

// ------------------
// Project save/load/download
// ------------------
(function addSaveLoad() {
  const panel = document.createElement('div');
  panel.style.marginTop = '8px';
  panel.innerHTML = `
    <h4>Project</h4>
    <div style="display:flex;gap:8px;flex-wrap:wrap;"><button id="saveProject">Save Project</button><button id="loadProject">Load Project</button><button id="downloadProject">Download Project</button></div>
  `;
  document.getElementById('left-panel').appendChild(panel);
  document.getElementById('saveProject').addEventListener('click', () => {
    const project = { rawData, workingData, columns, colTypes, models };
    localStorage.setItem('dcai_project', JSON.stringify(project));
    alert('Project saved to localStorage');
  });
  document.getElementById('loadProject').addEventListener('click', () => {
    const blob = localStorage.getItem('dcai_project');
    if (!blob) return alert('No project saved');
    const obj = JSON.parse(blob);
    rawData = obj.rawData || []; workingData = obj.workingData || []; columns = obj.columns || []; colTypes = obj.colTypes || {}; models = obj.models || {};
    postProcessState();
    alert('Project loaded');
  });
  document.getElementById('downloadProject').addEventListener('click', () => {
    const project = { rawData, workingData, columns, colTypes, models };
    const blob = new Blob([JSON.stringify(project, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'dcai_project.json'; a.click(); URL.revokeObjectURL(url);
  });
})();

// ------------------
// Initial call
// ------------------
postProcessState();

// End of app.js
