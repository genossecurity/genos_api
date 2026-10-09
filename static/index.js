'use strict';
(function () {
  const cmdInput = document.getElementById('cmdInput');
  const scanBtn = document.getElementById('scanBtn');
  const scanStatus = document.getElementById('scanStatus');
  const results = document.getElementById('results');
  const STAGE_MITRE_URLS = {
    'Execution': 'https://attack.mitre.org/tactics/TA0002/',
    'Persistence': 'https://attack.mitre.org/tactics/TA0003/',
    'Privilege Escalation': 'https://attack.mitre.org/tactics/TA0004/',
    'Defense Evasion': 'https://attack.mitre.org/tactics/TA0005/',
    'Credential Access': 'https://attack.mitre.org/tactics/TA0006/',
    'Discovery / Recon': 'https://attack.mitre.org/tactics/TA0007/',
    'Lateral Movement': 'https://attack.mitre.org/tactics/TA0008/',
    'Collection / Staging': 'https://attack.mitre.org/tactics/TA0009/',
    'C2 / Remote Access': 'https://attack.mitre.org/tactics/TA0011/',
    'Payload Retrieval': 'https://attack.mitre.org/techniques/T1105/',
    'Exfiltration': 'https://attack.mitre.org/tactics/TA0010/',
    'Impact': 'https://attack.mitre.org/tactics/TA0040/',
  };

  const themeButtons = document.querySelectorAll('[data-theme-btn]');
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('genos-theme', theme);
    themeButtons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.themeBtn === theme)));
  }
  themeButtons.forEach(b => b.addEventListener('click', () => applyTheme(b.dataset.themeBtn)));
  applyTheme(document.documentElement.getAttribute('data-theme') || 'dark');

  // Maps evidence / triggered features / fired rules into the 11 MITRE ATT&CK
  // tactic families so every family shows only its own relevant indicators.
  const FAMILY_KEYWORDS = {
    'Execution': ['runs_interpreter', 'executes_inline_code', 'inline_code', 'shell_spawn', 'eval_exec', 'interpreter', 'run_script_interpreter', 'execute_inline_code'],
    'Persistence': ['persistence', 'scheduled_task', 'registry_autorun', 'schtasks', 'cron', 'rc.local', 'service', 'autorun', 'establish_persistence'],
    'Privilege Escalation': ['privilege_escalation', 'privesc', 'setuid', 'sudoers', 'attempt_privilege_escalation'],
    'Defense Evasion': ['obfuscation', 'encoded_payload', 'base64_or_encoded_exec', 'defense_impairment', 'signed_proxy', 'lolbin', 'encode_payload', 'obfuscate_command', 'disable_defenses', 'use_signed_proxy'],
    'Credential Access': ['credential', 'reads_credential_store', 'shadow', 'mimikatz', 'secretsdump', 'access_credential_store', 'credential_dumping'],
    'Discovery': ['enumeration_recon', 'enumerates_identity', 'enumerates_network_config', 'network_enum', 'process_enum', 'enumerate_identity', 'enumerate_network', 'enumerate_environment'],
    'Lateral Movement': ['remote_execution_or_session', 'tunneling', 'remote_session', 'establish_tunnel'],
    'Command-and-Control / Payload Retrieval': ['downloads_remote_resource', 'download', 'pipe_to_shell', 'reverse_shell', 'download_remote_resource', 'reverse_shell_pattern'],
    'Exfiltration': ['exfil_data_movement', 'transfers_file_to_remote', 'transfer_file_remote', 'exfiltrate_data', 'remote_transfer'],
    'Impact': ['destructive_write', 'deletes_shadow_copies', 'delete_shadow_copies', 'archive_or_bulk_copy', 'archive_or_stage_data'],
  };

  function matchesFamily(familyName, key) {
    const kws = FAMILY_KEYWORDS[familyName] || [];
    const lowered = String(key).toLowerCase();
    return kws.some(kw => lowered.includes(kw));
  }

  function chip(text, mono) {
    const span = document.createElement('span');
    span.className = 'ioc-chip' + (mono ? ' mono' : '');
    span.textContent = text;
    return span;
  }

  function buildIocGroup(label, items, mono) {
    if (!items || !items.length) return null;
    const wrap = document.createElement('div');
    wrap.className = 'ioc-group';
    const l = document.createElement('div');
    l.className = 'ioc-group-label';
    l.textContent = label;
    wrap.appendChild(l);
    const chips = document.createElement('div');
    chips.className = 'ioc-chips';
    items.forEach(item => chips.appendChild(chip(item, mono)));
    wrap.appendChild(chips);
    return wrap;
  }

  // Artifact types each family owns; anything not claimed by a rendered family goes to "Other IOCs".
  const FAMILY_ARTIFACTS = {
    'Execution': ['files'],
    'Persistence': ['registry', 'files'],
    'Privilege Escalation': ['files'],
    'Defense Evasion': ['files', 'registry'],
    'Credential Access': ['files'],
    'Discovery': ['network', 'ports'],
    'Lateral Movement': ['network', 'ports'],
    'Command-and-Control / Payload Retrieval': ['network', 'ports', 'files'],
    'Exfiltration': ['network', 'ports', 'files'],
    'Impact': ['files', 'registry'],
  };

  function artifactGroup(type, evidence) {
    if (type === 'network') return buildIocGroup('Network', [...(evidence.urls || []), ...(evidence.domains || []), ...(evidence.ips || [])], true);
    if (type === 'ports') return buildIocGroup('Ports', evidence.ports || [], true);
    if (type === 'registry') return buildIocGroup('Registry', evidence.registry_paths || [], true);
    if (type === 'files') return buildIocGroup('Files', evidence.file_paths || [], true);
    return null;
  }

  function collectFamilyIndicators(familyName, data, claimed) {
    const evidence = data.evidence || {};
    const triggered = data.triggered_features || [];
    const firedRules = evidence.fired_rules || [];
    const semanticFeatures = evidence.semantic_features || [];

    const flags = new Set();
    triggered.filter(f => matchesFamily(familyName, f)).forEach(f => flags.add(f.replace(/^has_/, '').replace(/_/g, ' ')));
    firedRules.filter(r => matchesFamily(familyName, r)).forEach(r => flags.add(r));
    semanticFeatures.filter(s => matchesFamily(familyName, s)).forEach(s => flags.add(s.replace(/_/g, ' ')));

    const groups = [];
    const flagGroup = buildIocGroup('Indicators', Array.from(flags));
    if (flagGroup) groups.push(flagGroup);

    (FAMILY_ARTIFACTS[familyName] || []).forEach(type => {
      const g = artifactGroup(type, evidence);
      if (g) { groups.push(g); claimed.add(type); }
    });
    return groups;
  }

  function section(title) {
    const sec = document.createElement('div');
    sec.className = 'section';
    const t = document.createElement('div');
    t.className = 'section-title';
    t.textContent = title;
    const body = document.createElement('div');
    body.className = 'section-body';
    sec.appendChild(t);
    sec.appendChild(body);
    return { sec, body };
  }

  function renderProfile(data) {
    const ev = data.evidence || {};
    const rows = [
      ['OS / Platform', ev.platform],
      ['Binary', ev.executable],
      ['Subcommand', ev.subcommand],
      ['Interpreter', ev.interpreter],
      ['Execution style', ev.execution_style],
      ['Artifact type', ev.primary_artifact_type],
      ['Pipe', ev.has_pipe ? 'yes' : null],
      ['Chain', ev.has_chain ? 'yes' : null],
      ['Redirect', ev.has_redirect ? 'yes' : null],
      ['Obfuscated', ev.uses_obfuscation ? 'yes' : null],
    ].filter(r => r[1]);
    if (!rows.length) return;
    const { sec, body } = section('Command profile');
    const grid = document.createElement('div');
    grid.className = 'kv-grid';
    rows.forEach(([k, v]) => {
      const kv = document.createElement('div');
      kv.className = 'kv';
      kv.innerHTML = '<div class="k"></div><div class="v"></div>';
      kv.querySelector('.k').textContent = k;
      kv.querySelector('.v').textContent = v;
      grid.appendChild(kv);
    });
    body.appendChild(grid);
    results.appendChild(sec);
  }

  function renderBinaries(data) {
    const bins = (data.evidence || {}).binaries || [];
    if (!bins.length) return;
    const { sec, body } = section(`Binary inventory (${bins.length})`);
    const table = document.createElement('table');
    table.className = 'inv';
    table.innerHTML = '<thead><tr><th>Binary</th><th>Roles</th><th>Flags</th></tr></thead><tbody></tbody>';
    const tbody = table.querySelector('tbody');
    bins.forEach(b => {
      const tr = document.createElement('tr');
      const tdName = document.createElement('td');
      tdName.className = 'mono';
      tdName.textContent = b.binary;
      const tdRoles = document.createElement('td');
      (b.roles || []).forEach(r => {
        const span = document.createElement('span');
        span.className = 'role ' + r;
        span.textContent = r;
        tdRoles.appendChild(span);
      });
      const tdFlags = document.createElement('td');
      tdFlags.className = 'mono';
      tdFlags.textContent = (b.flags || []).join(' ') || '—';
      tr.append(tdName, tdRoles, tdFlags);
      tbody.appendChild(tr);
    });
    body.appendChild(table);
    results.appendChild(sec);
  }

  function renderFlags(data) {
    const ev = data.evidence || {};
    const groups = [
      buildIocGroup('High-signal flags', ev.high_signal_flags || [], true),
      buildIocGroup('Routing features', (data.triggered_features || []).map(f => f.replace(/^has_/, '').replace(/_/g, ' '))),
      buildIocGroup('Fired rules', ev.fired_rules || []),
      buildIocGroup('Obfuscation markers', ev.obfuscation_markers || []),
    ].filter(Boolean);
    if (!groups.length) return;
    const { sec, body } = section('Flags');
    groups.forEach(g => body.appendChild(g));
    results.appendChild(sec);
  }

  function renderBenign(data) {
    const card = document.createElement('div');
    card.className = 'verdict-card';

    const left = document.createElement('div');
    left.className = 'verdict-left';
    const pill = document.createElement('span');
    pill.className = 'verdict-pill benign';
    pill.textContent = 'Benign';
    left.appendChild(pill);
    const conf = document.createElement('span');
    conf.className = 'verdict-conf';
    conf.textContent = `${Number(data.label_confidence).toFixed(1)}%`;
    left.appendChild(conf);
    card.appendChild(left);

    const meta = document.createElement('div');
    meta.className = 'verdict-meta';
    if (typeof data.elapsed_ms === 'number') {
      const t = document.createElement('div');
      t.textContent = `${data.elapsed_ms} ms`;
      meta.appendChild(t);
    }
    card.appendChild(meta);
    results.appendChild(card);
  }

  function renderSuspicious(data) {
    const label = data.label || 'Suspicious';
    const pillClass = label.toLowerCase().includes('context') ? 'context_dependent'
      : label.toLowerCase().includes('malicious') ? 'malicious' : 'suspicious';

    const card = document.createElement('div');
    card.className = 'verdict-card verdict-summary';

    const left = document.createElement('div');
    left.className = 'verdict-stage-section';
    const stageLabel = document.createElement('span');
    stageLabel.className = 'verdict-stage-label';
    stageLabel.textContent = 'MITRE Stage:';
    left.appendChild(stageLabel);
    if (data.attack_stage) {
      const stage = document.createElement('a');
      stage.className = 'verdict-stage';
      stage.textContent = data.attack_stage;
      const stageUrl = STAGE_MITRE_URLS[data.attack_stage];
      stage.href = stageUrl || 'https://attack.mitre.org/tactics/';
      stage.target = '_blank';
      stage.rel = 'noopener noreferrer';
      stage.title = stageUrl ? `View ${data.attack_stage} in MITRE ATT&CK` : 'Explore MITRE ATT&CK tactics';
      left.appendChild(stage);
    }
    card.appendChild(left);

    const right = document.createElement('div');
    right.className = 'verdict-classification-section';
    const pill = document.createElement('span');
    pill.className = 'verdict-pill ' + pillClass;
    pill.textContent = label.replace(/_/g, ' ');
    right.appendChild(pill);
    const conf = document.createElement('span');
    conf.className = 'verdict-conf';
    conf.textContent = `${Number(data.label_confidence).toFixed(1)}%`;
    right.appendChild(conf);
    if (typeof data.elapsed_ms === 'number') {
      const t = document.createElement('span');
      t.className = 'verdict-time';
      t.textContent = `${data.elapsed_ms} ms`;
      right.appendChild(t);
    }
    card.appendChild(right);
    results.appendChild(card);

    if (data.deobfuscated_cmd || data.decoded_payload) {
      const box = document.createElement('div');
      box.className = 'decoded-box';
      const l = document.createElement('div');
      l.className = 'label';
      l.textContent = 'Deobfuscated Command';
      box.appendChild(l);
      const code = document.createElement('code');
      code.textContent = data.deobfuscated_cmd || data.decoded_payload;
      box.appendChild(code);
      results.appendChild(box);
    }

    renderProfile(data);
    renderBinaries(data);
    renderFlags(data);

    // Per-family cards, each with only its own relevant IOCs
    const claimed = new Set();
    const families = document.createElement('div');
    families.className = 'families';

    const predicted = (data.attack_families && data.attack_families.predicted_families) || [];
    const allScores = (data.attack_families && data.attack_families.all_family_scores) || [];
    const familyList = predicted.length ? predicted : allScores.filter(f => f.family !== 'Benign Admin').slice(0, 1);

    let renderedAny = false;
    familyList.forEach(f => {
      const groups = collectFamilyIndicators(f.family, data, claimed);
      if (!groups.length) return; // eliminate unused/empty family space
      renderedAny = true;

      const card = document.createElement('div');
      card.className = 'family-card';

      const head = document.createElement('div');
      head.className = 'family-head';
      const name = document.createElement('span');
      name.className = 'family-name';
      name.textContent = f.family;
      head.appendChild(name);
      const prob = document.createElement('span');
      prob.className = 'family-prob';
      prob.textContent = `${Number(f.probability).toFixed(1)}%`;
      head.appendChild(prob);
      card.appendChild(head);

      const body = document.createElement('div');
      body.className = 'family-body';
      groups.forEach(g => body.appendChild(g));
      card.appendChild(body);

      families.appendChild(card);
    });

    if (renderedAny) {
      results.appendChild(families);
    }

    // IOCs not owned by any rendered family
    const ev = data.evidence || {};
    const others = ['network', 'ports', 'registry', 'files']
      .filter(t => !claimed.has(t)).map(t => artifactGroup(t, ev)).filter(Boolean);
    if (others.length) {
      const { sec, body } = section('Other IOCs');
      others.forEach(g => body.appendChild(g));
      results.appendChild(sec);
    }

    if (!renderedAny && !others.length && data.analyst_hint) {
      const note = document.createElement('div');
      note.className = 'empty-note';
      note.textContent = data.analyst_hint;
      results.appendChild(note);
    }
  }

  function renderResults(data) {
    results.innerHTML = '';
    results.classList.remove('hidden');
    const label = (data.label || '').toLowerCase();
    if (label === 'benign') {
      renderBenign(data);
    } else {
      renderSuspicious(data);
    }
  }

  function renderError(message) {
    results.innerHTML = '';
    results.classList.remove('hidden');
    const card = document.createElement('div');
    card.className = 'verdict-card';
    card.textContent = message;
    results.appendChild(card);
  }

  async function runScan() {
    const command = cmdInput.value.trim();
    if (!command) {
      scanStatus.textContent = 'Enter a command to scan.';
      return;
    }

    scanBtn.disabled = true;
    scanStatus.textContent = 'Scanning…';

    try {
      const resp = await fetch('/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        renderError(data.error || 'Scan failed.');
        scanStatus.textContent = 'Error';
        return;
      }
      renderResults(data);
      scanStatus.textContent = 'Done';
    } catch (err) {
      renderError('Network error: ' + err.message);
      scanStatus.textContent = 'Error';
    } finally {
      scanBtn.disabled = false;
    }
  }

  scanBtn.addEventListener('click', runScan);
  cmdInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runScan();
    }
  });
})();

