// app.js
// This file contains the application logic for the Data Collection & Analysis AI.

/*
  HIGH-LEVEL THOUGHTS (developer commentary):

  Goal: Provide a small, self-contained web app that demonstrates a full data pipeline
  in the browser: ingestion, parsing, cleaning, analysis, visualization, and a tiny
  "AI" layer that generates automated suggestions and can train a simple regression model.

  Design choices:
  - Keep everything client-side so it's runnable from VS Code without servers.
  - Use Chart.js (CDN) for charts because it is reliable and concise.
  - Implement a small math utility (matrix ops + least-squares) rather than pulling a heavy ML library,
    to show the underlying mechanics and avoid extra dependencies.
  - Provide clear comments that explain WHY each step exists (not just what it does).
*/

// -----------------------------
// DOM elements (grabbed once)
// -----------------------------
const fileInput = document.getElementById('fileInput');
const pasteArea = document.getElementById('pasteArea');
const parseBtn = document.getElementById('parseBtn');
const sampleBtn = document.getElementById('sampleBtn');
const autoCleanBtn = document.getElementById('autoCleanBtn');
const exportBtn = document.getElementById('exportBtn');
const targetSelect = document.getElementById('targetSelect');
const featuresSelect = document.getElementById('featuresSelect');
const trainBtn = document.getElementById('trainBtn');
const autoAnalyzeBtn = document.getElementById('autoAnalyzeBtn');
const histColSelect = document.getElementById('histColSelect');
const histChartCanvas = document.getElementById('histChart');
const xSelect = document.getElementById('xSelect');
const ySelect = document.getElementById('ySelect');
const scatterChartCanvas = document.getElementById('scatterChart');
const dataTableDiv = document.getElementById('dataTable');
const summaryPre = document.getElementById('summaryPre');
const suggestionsList = document.getElementById('suggestionsList');
const modelPre = document.getElementById('modelPre');

// Application state (keeps things simple and centralized)
let rawData = [];       // array of objects representing the original parsed dataset
let workingData = [];   // array of objects after cleaning/transform steps
let columns = [];       // ordered list of column names
let colTypes = {};      // inferred types per column: 'numeric'|'categorical'
let histChart = null;   // Chart.js instance for histogram
let scatterChart = null;// Chart.js instance for scatter


// -----------------------------
// Utility functions
// -----------------------------

// Thought: many small helpers make code easier to reason about; keep them pure where possible.

function isMissing(val){
  // treat empty strings and common placeholders as missing
  return val === null || val === undefined || (String(val).trim() === '') || ['na','n/a','null','undefined'].includes(String(val).toLowerCase());
}

function toNumberIfPossible(v){
  // Try to parse a value to number; if impossible return NaN
  const n = parseFloat(String(v).replace(/[^0-9eE+.-]/g,''));
  return Number.isFinite(n) ? n : NaN;
}

function unique(arr){
  return Array.from(new Set(arr));
}

// -----------------------------
// CSV parsing (simple but robust enough for demo)
// -----------------------------

// THOUGHT: For production-grade CSV parsing you'd use PapaParse. Here we implement
// a compact parser that handles quoted fields and commas/newlines inside quotes.

function parseCSV(text){
  const rows = [];
  let cur = '';
  let row = [];
  let inQuotes = false;
  for (let i = 0; i < text.length; i++){
    const ch = text[i];
    const nch = text[i+1];
    if (ch === '"'){
      if (inQuotes && nch === '"'){
        // escaped quote ""
        cur += '"';
        i++; // skip next
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes){
      row.push(cur);
      cur = '';
    } else if ((ch === '\n' || ch === '\r') && !inQuotes){
      // handle CRLF or LF -- only push a new row when there is content in this row (avoids blank lines)
      if (ch === '\r' && nch === '\n') { /* skip: we'll get the \n next iteration */ }
      row.push(cur);
      // sometimes last row may be empty
      if (!(row.length === 1 && row[0] === '')) rows.push(row);
      row = [];
      cur = '';
    } else {
      cur += ch;
    }
  }
  // last cell
  if (cur !== '' || row.length > 0){
    row.push(cur);
    if (!(row.length === 1 && row[0] === '')) rows.push(row);
  }

  if (rows.length === 0) return [];
  // first row = headers
  const headers = rows[0].map(h => h.trim());
  const data = [];
  for (let i = 1; i < rows.length; i++){
    const r = rows[i];
    const obj = {};
    for (let j = 0; j < headers.length; j++){
      obj[headers[j]] = r[j] !== undefined ? r[j].trim() : '';
    }
    data.push(obj);
  }
  return data;
}

// -----------------------------
// Column type inference
// -----------------------------

function inferColumnTypes(sampleData, threshold=0.8){
  // For each column, compute fraction of values that parse to numbers. If above threshold -> numeric.
  const cols = Object.keys(sampleData[0] || {});
  const types = {};
  for (const col of cols){
    const vals = sampleData.map(r => r[col]);
    let numericCount = 0, count = 0;
    for (const v of vals){
      if (isMissing(v)) continue;
      count++;
      if (!Number.isNaN(toNumberIfPossible(v))) numericCount++;
    }
    types[col] = (count === 0) ? 'categorical' : (numericCount / count >= threshold ? 'numeric' : 'categorical');
  }
  return types;
}

// -----------------------------
// Basic statistics
// -----------------------------

function summarizeColumn(values, type){
  // Thought: keep stats lightweight yet useful for quick EDA.
  const cleaned = values.filter(v => !isMissing(v));
  if (type === 'numeric'){
    const nums = cleaned.map(v => toNumberIfPossible(v)).filter(n => !Number.isNaN(n));
    if (nums.length === 0) return {count:0};
    nums.sort((a,b)=>a-b);
    const sum = nums.reduce((a,b)=>a+b,0);
    const mean = sum / nums.length;
    const median = nums.length % 2 === 1 ? nums[(nums.length-1)/2] : (nums[nums.length/2-1]+nums[nums.length/2])/2;
    const sq = nums.map(n=>Math.pow(n-mean,2));
    const variance = sq.reduce((a,b)=>a+b,0)/(nums.length-1 || 1);
    const std = Math.sqrt(variance);
    return {count:nums.length, mean, median, std, min:nums[0], max:nums[nums.length-1]};
  } else {
    // categorical: return top frequency
    const freq = {};
    for (const v of cleaned){
      freq[v] = (freq[v] || 0) + 1;
    }
    const entries = Object.entries(freq).sort((a,b)=>b[1]-a[1]);
    return {count: cleaned.length, unique: Object.keys(freq).length, top: entries[0] ? {value:entries[0][0],count:entries[0][1]} : null};
  }
}

// -----------------------------
// Visualization helpers (Chart.js wrappers)
// -----------------------------

function drawHistogram(column){
  if (!column) return;
  const values = workingData.map(r => r[column]).filter(v => !isMissing(v)).map(v => toNumberIfPossible(v)).filter(n=>!Number.isNaN(n));
  if (values.length === 0){
    if (histChart) histChart.destroy();
    histChart = null;
    return;
  }

  // Build bins (10 bins by default)
  const bins = 10;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min) / bins || 1;
  const labels = [];
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < bins; i++) labels.push(`${(min + i*width).toFixed(2)} - ${(min + (i+1)*width).toFixed(2)}`);
  for (const v of values){
    const idx = Math.min(bins-1, Math.floor((v - min)/width));
    counts[idx]++;
  }

  if (histChart) histChart.destroy();
  histChart = new Chart(histChartCanvas.getContext('2d'), {
    type: 'bar',
    data: {labels, datasets: [{label: column, data: counts, backgroundColor: 'rgba(0,0,0,0.5)'}]},
    options: {responsive:true, maintainAspectRatio:false}
  });
}

function drawScatter(xCol, yCol){
  if (!xCol || !yCol) return;
  const pts = workingData.map(r => ({x: toNumberIfPossible(r[xCol]), y: toNumberIfPossible(r[yCol])})).filter(p => !Number.isNaN(p.x) && !Number.isNaN(p.y));
  if (scatterChart) scatterChart.destroy();
  scatterChart = new Chart(scatterChartCanvas.getContext('2d'), {
    type: 'scatter',
    data: {datasets:[{label:`${yCol} vs ${xCol}`, data: pts, showLine:false}]},
    options: {scales:{x:{title:{display:true,text:xCol}}, y:{title:{display:true,text:yCol}}}, responsive:true, maintainAspectRatio:false}
  });
}

// -----------------------------
// Table rendering
// -----------------------------

function renderTable(data){
  if (!data || data.length === 0){
    dataTableDiv.innerHTML = '<div class="note">No rows to display.</div>';
    return;
  }
  const cols = Object.keys(data[0]);
  const thead = `<thead><tr>${cols.map(c=>`<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>`;
  const rows = data.map(r => `<tr>${cols.map(c=>`<td>${escapeHtml(String(r[c]===undefined? '': r[c]))}</td>`).join('')}</tr>`).join('');
  dataTableDiv.innerHTML = `<table class="table">${thead}<tbody>${rows}</tbody></table>`;
}

function escapeHtml(s){
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c]));
}

// -----------------------------
// Cleaning and transformation
// -----------------------------

function autoClean(){
  // Thought: A conservative auto-clean — trim strings, impute simple numeric missing values with column mean.
  if (!workingData || workingData.length === 0) return;
  const cols = Object.keys(workingData[0]);
  // Impute numeric columns with mean
  for (const col of cols){
    if (colTypes[col] === 'numeric'){
      const nums = workingData.map(r => toNumberIfPossible(r[col])).filter(n=>!Number.isNaN(n));
      const mean = nums.reduce((a,b)=>a+b,0) / (nums.length || 1);
      for (const r of workingData){
        if (isMissing(r[col]) || Number.isNaN(toNumberIfPossible(r[col]))) r[col] = mean;
      }
    } else {
      // Trim strings
      for (const r of workingData){
        if (r[col] !== undefined && r[col] !== null) r[col] = String(r[col]).trim();
      }
    }
  }
  postProcessState();
}

// -----------------------------
// Simple linear regression using normal equation (supports multiple features)
// -----------------------------

// THOUGHT: Implementing least-squares ourselves demonstrates how coefficients are computed
// and keeps the app minimal. For large or production models, use numeric libraries or TensorFlow.js.

function transpose(mat){
  return mat[0].map((_,c) => mat.map(r => r[c]));
}
function matMul(A,B){
  const m=A.length, n=B[0].length, p=A[0].length; // A(mxp) * B(pxn)
  const out = Array.from({length:m}, ()=>Array(n).fill(0));
  for (let i=0;i<m;i++) for (let k=0;k<p;k++) for (let j=0;j<n;j++) out[i][j] += A[i][k]*B[k][j];
  return out;
}

function matInverse(A){
  // Gauss-Jordan inversion — acceptable for small matrices produced by feature counts.
  const n = A.length;
  const M = A.map((r,i)=>r.concat(Array.from({length:n},(_,j)=>i===j?1:0)));
  // forward elimination
  for (let i=0;i<n;i++){
    // find pivot
    let pivot = i;
    for (let j=i;j<n;j++) if (Math.abs(M[j][i]) > Math.abs(M[pivot][i])) pivot = j;
    if (Math.abs(M[pivot][i]) < 1e-12) throw new Error('Matrix singular or nearly singular');
    // swap
    if (pivot !== i){ const tmp = M[i]; M[i] = M[pivot]; M[pivot] = tmp; }
    // normalize row
    const div = M[i][i];
    for (let k=0;k<2*n;k++) M[i][k] /= div;
    // eliminate others
    for (let j=0;j<n;j++) if (j!==i){
      const factor = M[j][i];
      for (let k=0;k<2*n;k++) M[j][k] -= factor * M[i][k];
    }
  }
  // extract inverse
  return M.map(r => r.slice(n));
}

function fitLinearRegression(featureCols, targetCol){
  // Build X matrix (n x (p+1)) with bias column and y vector
  const X = workingData.map(r => [1, ...featureCols.map(c => toNumberIfPossible(r[c]))]);
  const y = workingData.map(r => [toNumberIfPossible(r[targetCol])]);
  // If any value is NaN we'll filter rows — thought: easier than trying to impute here.
  const good = X.map((row,i) => ({row, y:y[i]})).filter(obj => obj.row.slice(1).every(v=>!Number.isNaN(v)) && !Number.isNaN(obj.y[0]));
  if (good.length === 0) throw new Error('No valid rows for training');
  const Xgood = good.map(g=>g.row);
  const ygood = good.map(g=>g.y);

  // Normal equation: beta = (X^T X)^-1 X^T y
  const Xt = transpose(Xgood);
  const XtX = matMul(Xt, Xgood);
  const XtXinv = matInverse(XtX);
  const XtY = matMul(Xt, ygood);
  const beta = matMul(XtXinv, XtY); // (p+1) x 1

  // Predictions & metrics
  const yPred = Xgood.map(row => {
    return beta.reduce((acc, b, i) => acc + b[0]*row[i], 0);
  });
  const yTrue = ygood.map(v => v[0]);
  const ssRes = yTrue.map((t,i)=>Math.pow(t - yPred[i],2)).reduce((a,b)=>a+b,0);
  const mean = yTrue.reduce((a,b)=>a+b,0)/yTrue.length;
  const ssTot = yTrue.map(t=>Math.pow(t-mean,2)).reduce((a,b)=>a+b,0) || 1;
  const r2 = 1 - ssRes/ssTot;
  const rmse = Math.sqrt(ssRes / yTrue.length);

  return {coefficients: beta.map(b=>b[0]), r2, rmse, featureCols, intercept: beta[0][0]};
}

// -----------------------------
// AI suggestions (rule-based)
// -----------------------------

function generateSuggestions(){
  const suggestions = [];
  if (!workingData || workingData.length === 0) return suggestions;
  const nRows = workingData.length;
  const stats = {};
  for (const col of columns){
    const vals = workingData.map(r=>r[col]);
    stats[col] = summarizeColumn(vals, colTypes[col]);
    if (colTypes[col] === 'numeric'){
      // detect outliers via simple z-score rule
      const nums = workingData.map(r=>toNumberIfPossible(r[col])).filter(n=>!Number.isNaN(n));
      if (nums.length > 1){
        const mean = nums.reduce((a,b)=>a+b,0)/nums.length;
        const std = Math.sqrt(nums.map(v=>Math.pow(v-mean,2)).reduce((a,b)=>a+b,0)/(nums.length-1||1));
        const outliers = nums.filter(v => Math.abs((v-mean)/(std||1)) > 3);
        if (outliers.length > 0) suggestions.push(`Column '${col}' has ${outliers.length} extreme outlier(s) — consider capping or investigating.`);
      }
      if (stats[col].count < nRows * 0.95) suggestions.push(`Column '${col}' has missing or non-numeric entries — ${nRows - stats[col].count} missing/invalid values.`);
    } else {
      if ((stats[col].unique || 0) > Math.min(50, nRows/2)) suggestions.push(`Column '${col}' is high-cardinality categorical (unique=${stats[col].unique}) — consider hashing or embedding.`);
    }
  }

  // correlation hint: compute simple Pearson for numeric pairs and warn if very high collinearity
  const numericCols = columns.filter(c=>colTypes[c]==='numeric');
  const correlations = [];
  for (let i=0;i<numericCols.length;i++) for (let j=i+1;j<numericCols.length;j++){
    const a = numericCols[i], b = numericCols[j];
    const valsA = workingData.map(r=>toNumberIfPossible(r[a]));
    const valsB = workingData.map(r=>toNumberIfPossible(r[b]));
    const paired = valsA.map((v,k)=>({a:v,b:valsB[k]})).filter(p=>!Number.isNaN(p.a)&&!Number.isNaN(p.b));
    if (paired.length < 3) continue;
    const meanA = paired.reduce((s,p)=>s+p.a,0)/paired.length;
    const meanB = paired.reduce((s,p)=>s+p.b,0)/paired.length;
    const num = paired.reduce((s,p)=>s+(p.a-meanA)*(p.b-meanB),0);
    const denA = Math.sqrt(paired.reduce((s,p)=>s+Math.pow(p.a-meanA,2),0));
    const denB = Math.sqrt(paired.reduce((s,p)=>s+Math.pow(p.b-meanB,2),0));
    const corr = num / (denA*denB || 1);
    if (Math.abs(corr) > 0.9) suggestions.push(`Columns '${a}' and '${b}' are highly correlated (r=${corr.toFixed(2)}) — consider removing one or using PCA.`);
  }

  // Modeling hint
  const potentialTargets = numericCols;
  if (potentialTargets.length > 0) suggestions.push(`You can train a regression model on numeric targets — pick a target column and features.`);
  else suggestions.push(`No numeric columns detected for regression. If classification is desired, ensure a categorical target exists.`);

  return suggestions;
}

// -----------------------------
// Helpers to update UI from state
// -----------------------------

function postProcessState(){
  // Recompute derived state after updates
  if (!workingData || workingData.length === 0){
    summaryPre.textContent = '(no data)';
    suggestionsList.innerHTML = '<li>No suggestions — no data.</li>';
    renderTable([]);
    return;
  }
  columns = Object.keys(workingData[0]);
  colTypes = inferColumnTypes(workingData);

  // populate selectors
  fillSelectOptions(targetSelect, [''].concat(columns));
  fillMultiSelectOptions(featuresSelect, columns);
  fillSelectOptions(histColSelect, columns);
  fillSelectOptions(xSelect, columns);
  fillSelectOptions(ySelect, columns);

  // show summary
  const lines = [];
  lines.push(`Rows: ${workingData.length}`);
  for (const col of columns){
    const s = summarizeColumn(workingData.map(r=>r[col]), colTypes[col]);
    lines.push(`${col} (${colTypes[col]}): ${JSON.stringify(s)}`);
  }
  summaryPre.textContent = lines.join('\n');

  // show suggestions
  const sugg = generateSuggestions();
  if (sugg.length === 0) suggestionsList.innerHTML = '<li>No issues detected — dataset looks clean.</li>';
  else suggestionsList.innerHTML = sugg.map(s=>`<li>${escapeHtml(s)}</li>`).join('');

  // redraw visual widgets
  drawHistogram(histColSelect.value || columns.find(c=>colTypes[c]==='numeric'));
  drawScatter(xSelect.value || columns[0], ySelect.value || columns[1] || columns[0]);

  renderTable(workingData.slice(0,200)); // show only first 200 rows for performance
}

function fillSelectOptions(sel, options){
  sel.innerHTML = options.map(o => `<option value="${escapeHtml(String(o))}">${escapeHtml(String(o))}</option>`).join('');
}
function fillMultiSelectOptions(sel, options){
  sel.innerHTML = options.map(o => `<option value="${escapeHtml(String(o))}">${escapeHtml(String(o))}</option>`).join('');
}

// -----------------------------
// Export CSV
// -----------------------------

function exportCSV(){
  if (!workingData || workingData.length === 0) return;
  const cols = Object.keys(workingData[0]);
  const rows = [cols.join(',')];
  for (const r of workingData) rows.push(cols.map(c=>`"${String(r[c]).replace(/"/g,'""')}"`).join(','));
  const blob = new Blob([rows.join('\n')], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'export.csv'; a.click();
  URL.revokeObjectURL(url);
}

// -----------------------------
// Event handlers wiring
// -----------------------------

fileInput.addEventListener('change', e => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = ev => {
    rawData = parseCSV(ev.target.result);
    // Duplicate rawData to workingData so user can keep original and transform copy
    workingData = rawData.map(r => ({...r}));
    postProcessState();
  };
  reader.readAsText(f);
});

parseBtn.addEventListener('click', ()=>{
  const txt = pasteArea.value;
  rawData = parseCSV(txt);
  workingData = rawData.map(r => ({...r}));
  postProcessState();
});

sampleBtn.addEventListener('click', ()=>{
  // small sample dataset with numeric and categorical columns
  const sample = `name,age,years_experience,salary,department\nAlice,34,10,85000,Engineering\nBob,28,4,56000,Product\nCarol,45,20,130000,Engineering\nDan,22,1,42000,Support\nEve,39,12,98000,Product\nFrank,NaN,3,48000,Support\nGrace,31,6,,Engineering`;
  rawData = parseCSV(sample);
  workingData = rawData.map(r=>({...r}));
  postProcessState();
  pasteArea.value = sample;
});

autoCleanBtn.addEventListener('click', ()=>{
  autoClean();
  alert('Auto-clean finished (numeric missing values imputed with mean; strings trimmed)');
});

exportBtn.addEventListener('click', ()=>{
  exportCSV();
});

histColSelect.addEventListener('change', ()=> drawHistogram(histColSelect.value));
xSelect.addEventListener('change', ()=> drawScatter(xSelect.value, ySelect.value));
ySelect.addEventListener('change', ()=> drawScatter(xSelect.value, ySelect.value));

trainBtn.addEventListener('click', ()=>{
  const target = targetSelect.value;
  const features = Array.from(featuresSelect.selectedOptions).map(o=>o.value);
  if (!target) return alert('Select a target column');
  if (!features || features.length === 0) return alert('Select at least one feature');
  try{
    const model = fitLinearRegression(features, target);
    modelPre.textContent = `Intercept: ${model.intercept.toFixed(4)}\nCoefficients:\n${model.featureCols.map((c,i)=>` ${c}: ${model.coefficients[i+1].toFixed(4)}`).join('\n')}\nR2: ${model.r2.toFixed(4)}\nRMSE: ${model.rmse.toFixed(4)}`;
  } catch (err){
    modelPre.textContent = 'Model training failed: ' + err.message;
  }
});

autoAnalyzeBtn.addEventListener('click', ()=>{
  const sugg = generateSuggestions();
  alert('Auto Analyze complete — see the Suggestions panel for details.');
  postProcessState();
});

// Initialize tiny reactive wiring: when columns change, re-run state handlers

// -----------------------------
// Initial note: nothing loaded
// -----------------------------
postProcessState();

// End of app.js
