(() => {
  const shell = document.querySelector('[data-endpoint]');
  const endpoint = new URL(shell.dataset.endpoint, location.href);
  const exampleEndpoint = new URL(shell.dataset.exampleEndpoint || endpoint.href, location.href);
  const commandInput = document.getElementById('command');
  const tier2 = document.getElementById('tier2');
  const allIocs = document.getElementById('allIocs');
  const iocOptions = Array.from(document.querySelectorAll('[data-ioc]'));
  const runs = new Map();
  const curlText = {};

  const themeButtons = document.querySelectorAll('[data-theme-btn]');
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('genos-theme', theme); } catch (_) { /* Theme still applies. */ }
    themeButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.themeBtn === theme)));
  }
  themeButtons.forEach(button => button.addEventListener('click', () => applyTheme(button.dataset.themeBtn)));
  applyTheme(document.documentElement.dataset.theme || 'dark');

  // POSIX shell quoting keeps the scanned command literal, including quotes,
  // dollar signs, backticks, pipes, and newlines. curl performs URL encoding.
  function shellQuote(value) {
    return "'" + value.replace(/'/g, "'\"'\"'") + "'";
  }

  function selectedIocs() {
    return iocOptions.filter(option => option.checked).map(option => option.dataset.ioc);
  }

  function parameters(custom) {
    const params = new URLSearchParams({ command: commandInput.value.trim() });
    if (custom && tier2.checked) params.set('tier2', 'true');
    const iocs = custom ? selectedIocs() : [];
    if (iocs.length) params.set('iocs', iocs.length === iocOptions.length ? 'all' : iocs.join(','));
    return params;
  }

  function makeCurl(params) {
    const parts = ['curl ' + shellQuote(exampleEndpoint.href)];
    params.forEach((value, key) => parts.push('  --data-urlencode ' + shellQuote(key + '=' + value)));
    return parts.join(' \\\n');
  }

  function exampleResponse(custom) {
    // Illustrative values only. Actual results are fetched by the Run buttons.
    const result = { label: 'Context_Dependent', label_confidence: 99.0, deobfuscated_cmd: null };
    if (custom && tier2.checked) {
      result.tier2 = { status: 'completed', stage: 'Credential Access', stage_confidence: 98.4 };
    }
    const iocs = custom ? selectedIocs() : [];
    if (iocs.length) result.iocs = Object.fromEntries(iocs.map(field => [field, field === 'files' ? ['/etc/shadow'] : []]));
    return result;
  }

  function render() {
    runs.forEach(controller => controller.abort());
    runs.clear();
    const selected = selectedIocs();
    allIocs.checked = selected.length === iocOptions.length;
    allIocs.indeterminate = selected.length > 0 && selected.length < iocOptions.length;
    const summary = [tier2.checked ? 'Tier 1 + Tier 2' : 'Tier 1 only'];
    if (selected.length) summary.push(selected.length === iocOptions.length ? 'all IOCs' : selected.length + ' IOC types');
    document.getElementById('selectionSummary').textContent = summary.join(' · ');
    document.getElementById('copyStatus').textContent = '';
    curlText.basic = document.getElementById('basicCurl').textContent;
    document.getElementById('customCurl').textContent = curlText.custom = makeCurl(parameters(true));
    document.getElementById('customResponse').textContent = JSON.stringify(exampleResponse(true), null, 2);
    ['basic', 'custom'].forEach(name => {
      const status = document.getElementById(name + 'Status');
      if (name === 'custom') status.textContent = 'Illustrative values';
      status.classList.remove('error', 'loading');
    });
    document.querySelectorAll('[data-copy], [data-run]').forEach(button => {
      button.disabled = button.dataset.copy === 'basic' ? false : !commandInput.value.trim();
    });
  }

  async function copyRequest(name) {
    const text = curlText[name];
    const status = document.getElementById('copyStatus');
    const button = document.querySelector('[data-copy="' + name + '"]');
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const active = document.activeElement;
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.select();
        let copied;
        try { copied = document.execCommand('copy'); }
        finally { textarea.remove(); active?.focus(); }
        if (!copied) throw new Error('Clipboard unavailable');
      }
      const originalText = button.dataset.originalText || button.textContent;
      button.dataset.originalText = originalText;
      button.textContent = 'Copied';
      button.classList.add('copied');
      button.parentElement.querySelector('.copy-feedback')?.remove();
      const feedback = document.createElement('span');
      feedback.className = 'copy-feedback';
      feedback.textContent = 'Copied to Clipboard';
      button.parentElement.appendChild(feedback);
      window.setTimeout(() => {
        feedback.remove();
        button.textContent = originalText;
        button.classList.remove('copied');
      }, 1800);
      status.textContent = '';
    } catch (_) {
      status.textContent = 'Copy is unavailable in this browser. Select the curl command and copy it manually.';
    }
  }

  async function runRequest(name) {
    if (!commandInput.reportValidity()) return;
    runs.get(name)?.abort();
    const controller = new AbortController();
    runs.set(name, controller);
    const button = document.querySelector('[data-run="' + name + '"]');
    const status = document.getElementById(name + 'Status');
    const responseCode = document.getElementById(name + 'Response');
    const url = new URL(endpoint);
    url.search = parameters(name === 'custom').toString();
    button.disabled = true;
    status.textContent = 'Analyzing…';
    status.classList.remove('error');
    status.classList.add('loading');
    responseCode.textContent = 'Waiting for the API…';
    try {
      const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
      const data = await response.json();
      if (runs.get(name) !== controller) return;
      responseCode.textContent = JSON.stringify(data, null, 2);
      status.textContent = 'HTTP ' + response.status;
      status.classList.toggle('error', !response.ok);
      status.classList.remove('loading');
    } catch (error) {
      if (controller.signal.aborted || runs.get(name) !== controller) return;
      status.textContent = 'Request failed';
      status.classList.add('error');
      status.classList.remove('loading');
      responseCode.textContent = 'Unable to reach the API or read its response. Please try again.';
    } finally {
      if (runs.get(name) === controller) {
        runs.delete(name);
        button.disabled = !commandInput.value.trim();
      }
    }
  }

  commandInput.addEventListener('input', render);
  tier2.addEventListener('change', render);
  iocOptions.forEach(option => option.addEventListener('change', render));
  allIocs.addEventListener('change', () => {
    iocOptions.forEach(option => { option.checked = allIocs.checked; });
    render();
  });
  document.getElementById('resetOptions').addEventListener('click', () => {
    tier2.checked = false;
    iocOptions.forEach(option => { option.checked = false; });
    render();
  });
  document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', () => copyRequest(button.dataset.copy)));
  document.querySelectorAll('[data-run]').forEach(button => button.addEventListener('click', () => runRequest(button.dataset.run)));
  render();
})();
