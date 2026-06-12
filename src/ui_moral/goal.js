/* ═══════════════════════════════════════════════════════════
   GOAL.JS — Goal Setup page
   Handles: goal input, LLM block generation, session creation,
            navigation to workspace, localStorage history
════════════════════════════════════════════════════════════ */

'use strict';

// ─── Safe localStorage helpers (throws in private mode) ───
function lsGet(key, fallback = null) {
  try { return localStorage.getItem(key); } catch { return fallback; }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}

// ─── State ─────────────────────────────────────────────────
const state = {
  level:        'beginner',
  duration:     '45',
  preference:   'balanced',
  selfEfficacy: '3',   // Zimmermann SRL Forethought: self-efficacy belief (1-5)
  lastBlock:    null,  // { goal, block }
  apiKey:       '',
  apiProvider:  'deepseek',
};

// ─── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSavedOptions();
});

/* ══════════════════════════════════════════════════════════
   OPTIONS (Level / Duration / Preference)
══════════════════════════════════════════════════════════ */
function selectOpt(group, btn) {
  const groupMap = { level: 'level-btns', duration: 'duration-btns', preference: 'preference-btns', selfEfficacy: 'efficacy-btns' };
  const container = document.getElementById(groupMap[group]);
  if (!container) return;
  container.querySelectorAll('.goal-opt-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  state[group] = btn.dataset.value;
  saveOptions();
}

function loadSavedOptions() {
  try {
    const saved = JSON.parse(localStorage.getItem('goal_options') || '{}');
    if (saved.level)         state.level         = saved.level;
    if (saved.duration)      state.duration      = saved.duration;
    if (saved.preference)    state.preference    = saved.preference;
    if (saved.selfEfficacy)  state.selfEfficacy  = saved.selfEfficacy;

    // Apply saved selections to buttons
    applySelection('level-btns',      state.level);
    applySelection('duration-btns',   state.duration);
    applySelection('preference-btns', state.preference);
    applySelection('efficacy-btns',   state.selfEfficacy);
  } catch {}
}

function applySelection(containerId, value) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.querySelectorAll('.goal-opt-btn').forEach(btn => {
    btn.classList.toggle('selected', btn.dataset.value === value);
  });
}

function saveOptions() {
  lsSet('goal_options', JSON.stringify({
    level:        state.level,
    duration:     state.duration,
    preference:   state.preference,
    selfEfficacy: state.selfEfficacy,
  }));
}

/* ══════════════════════════════════════════════════════════
   CHIPS — quick goal templates
══════════════════════════════════════════════════════════ */
function setGoal(text) {
  document.getElementById('goal-input').value = text;
  document.getElementById('goal-input').focus();
}

/* ══════════════════════════════════════════════════════════
   API KEY — provider selection and validation
══════════════════════════════════════════════════════════ */
function selectProvider(provider, btn) {
  state.apiProvider = provider;
  document.querySelectorAll('#provider-tabs .goal-opt-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
}

async function checkApiKey() {
  const key = document.getElementById('api-key-input').value.trim();
  if (!key) {
    showKeyStatus('error', '❌ Введи API-ключ');
    return;
  }

  const btn = document.getElementById('check-key-btn');
  btn.disabled = true;
  showKeyStatus('loading', 'Проверяем...');

  try {
    const resp = await fetch('/api/validate-key', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ api_key: key, provider: state.apiProvider }),
    });
    const data = await resp.json();
    if (data.valid) {
      state.apiKey = key;
      showKeyStatus('ok', '✅ Ключ принят — можно начинать');
      // Reveal main form
      const wrapper = document.getElementById('main-form-wrapper');
      wrapper.style.display = 'flex';
      // Scroll to main form
      wrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // Lock key input
      document.getElementById('api-key-input').disabled = true;
      btn.textContent = '✅ Проверено';
    } else {
      showKeyStatus('error', `❌ ${data.error || 'Ключ не прошёл проверку'}`);
      btn.disabled = false;
    }
  } catch (e) {
    showKeyStatus('error', `❌ Ошибка сети: ${e.message}`);
    btn.disabled = false;
  }
}

function showKeyStatus(type, msg) {
  const el = document.getElementById('key-status');
  el.className = `goal-key-status ${type}`;
  el.textContent = msg;
}

/* ══════════════════════════════════════════════════════════
   BUILD GOAL
══════════════════════════════════════════════════════════ */
async function buildGoal() {
  if (!state.apiKey) {
    showError('Сначала введи и проверь API-ключ в блоке выше.');
    return;
  }
  const goalText = document.getElementById('goal-input').value.trim();
  if (!goalText) {
    showError('Введи цель обучения — что именно хочешь научиться?');
    return;
  }

  // UI: loading state
  setLoading(true);
  hideError();
  document.getElementById('preview-block').classList.remove('visible');

  try {
    const resp = await fetch('/api/goal/build', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        goal_text:      goalText,
        language:       'python',
        level:          state.level,
        duration_min:   parseInt(state.duration, 10),
        preference:     state.preference,
        api_key:        state.apiKey,
        api_provider:   state.apiProvider,
        self_efficacy:  parseInt(state.selfEfficacy, 10) || 3,
      }),
    });

    if (!resp.ok) {
      throw new Error(`Сервер вернул ${resp.status}`);
    }

    const data = await resp.json();
    state.lastBlock = data;

    renderPreview(data);
    document.getElementById('preview-block').classList.add('visible');

    // Scroll to preview
    document.getElementById('preview-block').scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (e) {
    showError(`Не удалось сгенерировать план: ${e.message}. Попробуй ещё раз.`);
    console.error(e);
  } finally {
    setLoading(false);
  }
}

/* ══════════════════════════════════════════════════════════
   RENDER PREVIEW
══════════════════════════════════════════════════════════ */
function renderPreview(data) {
  const goal  = data.goal  || {};
  const block = data.block || {};

  // Title
  document.getElementById('preview-title').textContent = goal.title || 'Учебный план';

  // Outcomes
  const outcomesList = document.getElementById('preview-outcomes');
  outcomesList.innerHTML = '';
  (block.outcomes || []).forEach(out => {
    const li = document.createElement('li');
    li.className = 'goal-outcome';
    li.textContent = out;
    outcomesList.appendChild(li);
  });

  // Prerequisites
  const prereqsEl = document.getElementById('preview-prereqs');
  prereqsEl.innerHTML = '';
  if ((block.prerequisites || []).length === 0) {
    prereqsEl.innerHTML = '<span class="goal-prereq">Нет требований</span>';
  } else {
    (block.prerequisites || []).forEach(pr => {
      const span = document.createElement('span');
      span.className = 'goal-prereq';
      span.textContent = pr;
      prereqsEl.appendChild(span);
    });
  }

  // Steps
  const stepsEl = document.getElementById('preview-steps');
  stepsEl.innerHTML = '';
  const typeIcon = { micro_lesson: '📖', exercise: '✏️', task: '🔧', reflection: '💭' };
  (block.steps || []).forEach((step, idx) => {
    const div = document.createElement('div');
    div.className = 'goal-step-preview';
    const icon = typeIcon[step.type] || '📌';
    div.innerHTML = `
      <span class="goal-step-num">${idx + 1}</span>
      <span>${icon} ${escapeHtml(step.title)}</span>
    `;
    stepsEl.appendChild(div);
  });

  // Criteria
  document.getElementById('preview-criteria').textContent =
    goal.success_criteria || 'Пройти все автотесты';
}

/* ══════════════════════════════════════════════════════════
   START SESSION → redirect to workspace
══════════════════════════════════════════════════════════ */
async function startSession() {
  if (!state.lastBlock) return;

  const btn = document.getElementById('start-btn');
  btn.disabled = true;
  btn.textContent = 'Создаём сессию...';

  try {
    const resp = await fetch('/api/session/create', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        goal_id:       state.lastBlock.goal?.id  || 'goal_demo',
        block_id:      state.lastBlock.block?.id || 'block_demo',
        user_prefs:    {},
        block:         state.lastBlock.block || null,
        goal:          state.lastBlock.goal  || null,
        api_key:       state.apiKey,
        api_provider:  state.apiProvider,
        self_efficacy: parseInt(state.selfEfficacy, 10) || 3,
      }),
    });

    if (!resp.ok) throw new Error(`Сервер вернул ${resp.status}`);

    const session = await resp.json();
    const sessionId = session.session_id;

    // Save block to localStorage for workspace
    lsSet('last_block', JSON.stringify(state.lastBlock));
    lsSet('last_session_id', sessionId);
    sessionStorage.setItem(`session_token_${sessionId}`, session.session_token);
    sessionStorage.setItem(`session_api_key_${sessionId}`, state.apiKey);
    sessionStorage.setItem(`session_api_provider_${sessionId}`, state.apiProvider);

    // Redirect
    window.location.href = `/workspace?session_id=${sessionId}`;

  } catch (e) {
    showError(`Не удалось создать сессию: ${e.message}`);
    btn.disabled = false;
    btn.textContent = '🚀 Начать занятие';
  }
}

/* ══════════════════════════════════════════════════════════
   REBUILD / SAVE
══════════════════════════════════════════════════════════ */
function rebuildGoal() {
  document.getElementById('preview-block').classList.remove('visible');
  buildGoal();
}

/* ══════════════════════════════════════════════════════════
   UI HELPERS
══════════════════════════════════════════════════════════ */
function setLoading(on) {
  document.getElementById('build-btn').disabled = on;
  document.getElementById('loading').classList.toggle('visible', on);
  if (on) {
    document.getElementById('build-btn').textContent = 'Генерирую...';
  } else {
    document.getElementById('build-btn').textContent = '✨ Сформировать учебный блок';
  }
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.classList.add('visible');
}

function hideError() {
  document.getElementById('error-msg').classList.remove('visible');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
