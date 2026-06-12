/* ═══════════════════════════════════════════════════════════
   WORKSPACE.JS — Python Tutor Learning Studio
   Manages: resumable session, CodeMirror editor, chat WS,
            Run/Test buttons, metrics polling and stepper
════════════════════════════════════════════════════════════ */

'use strict';

// ─── Safe localStorage helpers (throws in private mode) ───
function lsGet(key, fallback = null) {
  try { return localStorage.getItem(key); } catch (e) { return fallback; }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, value); } catch (e) {}
}

// ─── Session state persistence helpers ────────────────────
function saveStepCode(sessionId, stepIdx, code) {
  try {
    const map = JSON.parse(localStorage.getItem(`ws_codes_${sessionId}`) || '{}');
    if (code === null || code === undefined) { delete map[stepIdx]; }
    else { map[stepIdx] = code; }
    localStorage.setItem(`ws_codes_${sessionId}`, JSON.stringify(map));
  } catch (e) {}
}
function loadStepCodes(sessionId) {
  try { return JSON.parse(localStorage.getItem(`ws_codes_${sessionId}`) || '{}'); } catch (e) { return {}; }
}
function saveProgress(sessionId, completedSet) {
  try { localStorage.setItem(`ws_progress_${sessionId}`, JSON.stringify([...completedSet])); } catch (e) {}
}
function loadProgress(sessionId) {
  try { return new Set(JSON.parse(localStorage.getItem(`ws_progress_${sessionId}`) || '[]')); } catch (e) { return new Set(); }
}
function saveStepTaskId(sessionId, stepIdx, taskId) {
  try {
    const map = JSON.parse(localStorage.getItem(`ws_taskids_${sessionId}`) || '{}');
    if (taskId) map[stepIdx] = taskId; else delete map[stepIdx];
    localStorage.setItem(`ws_taskids_${sessionId}`, JSON.stringify(map));
  } catch (e) {}
}
function patchStepTaskIds(steps, sessionId) {
  try {
    const saved = JSON.parse(localStorage.getItem(`ws_taskids_${sessionId}`) || '{}');
    steps.forEach((step, idx) => { if (!step.task_id && saved[idx]) step.task_id = saved[idx]; });
  } catch (e) {}
}

// ─── State ────────────────────────────────────────────────
const state = {
  sessionId:      null,
  sessionToken:   null,
  taskId:         null,
  currentStep:    0,
  steps:          [],
  block:          null,
  goal:           null,
  level:          'beginner',
  timerSeconds:   0,
  timerInterval:  null,
  lastFailedTests: [],
  lastOutput:     '',
  theoryCache:    {},   // { stepIndex: htmlString } — кэш сгенерированной теории
  completedSteps: new Set(),   // индексы завершённых шагов
  stepTestResults: {},          // { stepIndex: { passed, total } } — результаты тестов по шагам
  stepCodeCache:   {},          // { stepIndex: codeString } — код пользователя по шагам
  maxSrlPhaseIdx: -1,           // monotonic SRL phase (Zimmermann): 0=Планирование, 1=Выполнение, 2=Рефлексия
  reflSummaryHTML: null,        // кэш итога рефлексии (не сбрасывается при переходе между шагами)
  isLoadingStep:  false,        // блокировка повторного activateStep пока идёт загрузка шага
};

// ─── WebSocket handles ─────────────────────────────────────
let tutorWS   = null;
let tutorPingInterval = null;
let metricsInterval = null;
let theoryExpandInFlight = false;
let currentBotMsgEl = null;

// ─── CodeMirror instance ───────────────────────────────────
let editor = null;

const PASS_RATIO = 0.6;

function requiredPassedCount(total) {
  return Math.max(1, Math.ceil((Number(total) || 0) * PASS_RATIO));
}

function formatPassRequirement(total) {
  const required = requiredPassedCount(total);
  return `${required}/${total}`;
}

function isStepPassedByTests(idx) {
  const tr = state.stepTestResults[idx];
  return !!(tr && tr.total > 0 && tr.passed >= requiredPassedCount(tr.total));
}

function isStepPerfectByTests(idx) {
  const tr = state.stepTestResults[idx];
  return !!(tr && tr.total > 0 && tr.passed === tr.total);
}

function isStepAvailableForProgress(idx) {
  const step = state.steps[idx];
  if (!step) return false;
  if (step.type === 'exercise' || step.type === 'task') {
    return state.completedSteps.has(idx) || isStepPassedByTests(idx);
  }
  return state.completedSteps.has(idx);
}

function arePrereqsDone(idx) {
  return state.steps.slice(0, idx).every((_, j) => isStepAvailableForProgress(j));
}

function stepProgressHint(idx) {
  const tr = state.stepTestResults[idx];
  if (tr && tr.total > 0) {
    return `Сейчас пройдено ${tr.passed}/${tr.total}; зачёт от ${formatPassRequirement(tr.total)}.`;
  }
  return 'Запусти Test для текущего задания; Run не обновляет статус шага.';
}

function showToast(message, type = 'info', timeoutMs = 4500) {
  const root = document.getElementById('toast-root');
  if (!root) return null;
  const el = document.createElement('div');
  el.className = `ws-toast ws-toast--${type}`;
  el.textContent = message;
  root.appendChild(el);
  const close = () => {
    el.classList.add('is-hiding');
    setTimeout(() => el.remove(), 180);
  };
  if (timeoutMs) setTimeout(close, timeoutMs);
  return close;
}

function trackLongLLMRequest(label) {
  let closeToast = null;
  const timer = setTimeout(() => {
    closeToast = showToast(`${label}: LLM/API ещё генерирует ответ. Система не зависла, ждём ответ провайдера.`, 'wait', 0);
  }, 9000);
  return () => {
    clearTimeout(timer);
    if (closeToast) closeToast();
  };
}


/* ══════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initSessionId();
  initCodeMirror();
  initTimer();
  initChatInput();
  loadSession();
});

function initSessionId() {
  const params = new URLSearchParams(window.location.search);
  state.sessionId = params.get('session_id') || lsGet('last_session_id') || generateId();
  state.sessionToken = sessionStorage.getItem(`session_token_${state.sessionId}`) || '';
  try { localStorage.removeItem(`session_token_${state.sessionId}`); } catch (e) {}
  lsSet('last_session_id', state.sessionId);
}

function generateId() {
  return 'sess_' + Math.random().toString(36).slice(2, 10);
}

/* ══════════════════════════════════════════════════════════
   CODEMIRROR
══════════════════════════════════════════════════════════ */
function initCodeMirror() {
  const el = document.getElementById('code-editor');
  editor = CodeMirror(el, {
    mode:           'python',
    theme:          'monokai',
    lineNumbers:    true,
    matchBrackets:  true,
    autoCloseBrackets: true,
    indentUnit:     4,
    tabSize:        4,
    indentWithTabs: false,
    lineWrapping:   false,
    autofocus:      true,
    extraKeys: {
      'Tab':              (cm) => cm.execCommand('insertSoftTab'),
      'Ctrl-Enter':       runCode,
      'Shift-Enter':      runCode,
    },
  });
  editor.setSize('100%', '100%');
  let _codeSaveTimer = null;
  let _codeSavedStatusTimer = null;
  editor.on('change', () => {
    clearTimeout(_codeSaveTimer);
    _codeSaveTimer = setTimeout(() => {
      if (state.sessionId) {
        saveStepCode(state.sessionId, state.currentStep, editor.getValue());
        persistWorkspaceState({ editor_code: editor.getValue() });
        setConsoleStatus('Код сохранён');
        clearTimeout(_codeSavedStatusTimer);
        _codeSavedStatusTimer = setTimeout(() => {
          const status = document.getElementById('console-status');
          if (status && status.textContent === 'Код сохранён') setConsoleStatus('');
        }, 1800);
      }
    }, 1500);
  });
}

/* ══════════════════════════════════════════════════════════
   SESSION LOADING
══════════════════════════════════════════════════════════ */
function authHeaders(extra = {}) {
  return Object.assign({ 'X-Session-Token': state.sessionToken }, extra);
}

async function readJsonResponse(resp, fallbackMessage) {
  const raw = await resp.text();
  let data = {};
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch (_) {
    throw new Error(`${fallbackMessage}: сервер вернул некорректный ответ (${resp.status}).`);
  }
  if (!resp.ok || data.status === 'error') {
    throw new Error(data.detail || data.error || `${fallbackMessage} (${resp.status}).`);
  }
  return data;
}

async function persistWorkspaceState(patch) {
  if (!state.sessionId || !state.sessionToken) return;
  try {
    await fetch(`/api/session/${state.sessionId}/state`, {
      method: 'PATCH',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(patch),
    });
  } catch (e) { console.warn('Could not persist workspace state:', e); }
}

async function resumeSession() {
  if (!state.sessionToken) return;
  const apiKey = sessionStorage.getItem(`session_api_key_${state.sessionId}`) || '';
  const apiProvider = sessionStorage.getItem(`session_api_provider_${state.sessionId}`) || 'deepseek';
  const resp = await fetch(`/api/session/${state.sessionId}/resume`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_token: state.sessionToken, api_key: apiKey, api_provider: apiProvider }),
  });
  await readJsonResponse(resp, 'Не удалось восстановить учебную сессию');
}

async function loadSession() {
  try {
    await resumeSession();
    const resp = await fetch(`/api/session/${state.sessionId}`, { headers: authHeaders() });
    if (resp.ok) {
      const data = await readJsonResponse(resp, 'Не удалось загрузить учебную сессию');
      applySessionData(data);
    } else {
      // Session doesn't exist — create a minimal one from last_block in localStorage
      const lastBlock = getLastBlock();
      if (lastBlock) {
        applyLocalBlock(lastBlock);
      } else {
        // Redirect to goal setup if no session data
        showOfflineMode();
      }
    }
  } catch (e) {
    console.warn('Could not load session, using offline mode:', e);
    const lastBlock = getLastBlock();
    if (lastBlock) applyLocalBlock(lastBlock);
    else showOfflineMode();
  }

  // Connect WS after session setup
  connectTutorWS();
  pollMetrics();
}

function applySessionData(data) {
  state.taskId      = data.task_id || null;
  state.currentStep = data.current_step || 0;
  state.block       = data.block;
  state.theoryCache = data.theory_cache || {};
  if (data.last_test_result) state.stepTestResults[state.currentStep] = data.last_test_result.summary || {};
  if (data.last_run_output) { state.lastOutput = data.last_run_output; showOutput(data.last_run_output); }
  if (data.reflection_result) state.reflSummaryHTML = buildReflectionSummaryHTML(data.reflection_result);
  const chatContainer = document.getElementById('chat-messages');
  if (chatContainer && Array.isArray(data.chat_history) && data.chat_history.length) {
    chatContainer.innerHTML = '';
    data.chat_history.forEach(msg => appendChatMessage(msg.role, msg.content || ''));
  }

  // Restore persisted code cache, progress, and step task_ids before building stepper
  state.stepCodeCache   = loadStepCodes(state.sessionId);
  state.completedSteps  = loadProgress(state.sessionId);
  if (data.block && data.block.steps) {
    patchStepTaskIds(data.block.steps, state.sessionId);
    // Server state is authoritative: restore code for every generated task,
    // not only for the currently displayed step or the current browser cache.
    const serverCodes = data.editor_code_per_task || {};
    const serverTestResults = data.test_results || {};
    data.block.steps.forEach((step, idx) => {
      if (step.task_id && serverCodes[step.task_id] !== undefined) {
        state.stepCodeCache[idx] = serverCodes[step.task_id];
        saveStepCode(state.sessionId, idx, serverCodes[step.task_id]);
      }
      if (step.task_id && serverTestResults[step.task_id] && serverTestResults[step.task_id].summary) {
        state.stepTestResults[idx] = serverTestResults[step.task_id].summary;
      }
    });
  }
  if (data.editor_code) state.stepCodeCache[state.currentStep] = data.editor_code;

  if (data.goal) {
    state.goal = data.goal;
    document.getElementById('goal-title').textContent = data.goal.title || 'Python Tutor';
    document.title = data.goal.title || 'Python Tutor';
    if (data.goal.level) initLevelButton(data.goal.level);
  }

  if (data.block) {
    buildStepper(data.block.steps || []);
  }

  // Activate the current step (will generate task if needed)
  if (state.steps.length > 0) {
    activateStep(state.currentStep);
  } else if (state.taskId) {
    loadTask(state.taskId);
  }
}

function applyLocalBlock(block) {
  state.block = block;
  state.goal = block.goal || null;
  state.stepCodeCache  = loadStepCodes(state.sessionId);
  state.completedSteps = loadProgress(state.sessionId);
  const steps = (block.block && block.block.steps) ? block.block.steps : [];
  patchStepTaskIds(steps, state.sessionId);
  if (block.goal) {
    document.getElementById('goal-title').textContent = block.goal.title || 'Python Tutor';
    if (block.goal.level) initLevelButton(block.goal.level);
  }
  buildStepper(steps);
  loadTask(state.taskId);
}

const LEVEL_LABELS = { beginner: 'Новичок', intermediate: 'Средний', advanced: 'Продвинутый' };

/* Populate the task-card meta chips: topic, level and an estimated time
   derived from the session duration split across the coding steps (problem §8). */
function updateTaskMeta(stepType) {
  const meta = document.getElementById('task-meta');
  if (!meta) return;
  if (stepType !== 'task' && stepType !== 'exercise') { meta.style.display = 'none'; return; }

  const goal = getCurrentGoal();
  const topic = goal.title || goal.requested_topic || (typeof goal === 'string' ? goal : '') || 'Python';
  const level = goal.level || state.level || 'beginner';

  // Estimated time: split the lesson duration across the coding steps,
  // exercises being lighter than the main task.
  const duration = Number(goal.duration_min) || 45;
  const codingSteps = (state.steps || []).filter(s => s.type === 'task' || s.type === 'exercise').length || 1;
  let estMin = Math.round((duration * 0.6) / codingSteps);
  if (stepType === 'exercise') estMin = Math.max(3, Math.round(estMin * 0.6));
  estMin = Math.min(Math.max(estMin, 3), duration);

  document.getElementById('task-meta-topic').textContent = `📚 ${topic}`;
  document.getElementById('task-meta-level').textContent = `🎯 ${LEVEL_LABELS[level] || level}`;
  document.getElementById('task-meta-level').className =
    'ws-task-chip ws-task-chip--level ws-level-' + (LEVEL_LABELS[level] ? level : 'beginner');
  document.getElementById('task-meta-time').textContent = `⏱ ~${estMin} мин`;
  meta.style.display = '';
}

/* At-a-glance pass ratio on the task card (problem §8). */
function updateTaskProgress(passed, total) {
  const box  = document.getElementById('task-test-progress');
  const text = document.getElementById('task-progress-text');
  const fill = document.getElementById('task-progress-fill');
  const status = document.getElementById('task-progress-status');
  if (!box) return;
  if (!total || total <= 0) { box.style.display = 'none'; return; }
  const pct = Math.round((passed / total) * 100);
  const required = requiredPassedCount(total);
  text.textContent = `${passed} / ${total} пройдено`;
  if (status) {
    status.className = 'ws-task-progress-status';
    if (passed === total) {
      status.textContent = `Полный зачёт: 100%.`;
      status.classList.add('is-complete');
    } else if (passed >= required) {
      status.textContent = `Зачтено частично. Для перехода достаточно ${formatPassRequirement(total)}, можно довести до 100%.`;
      status.classList.add('is-partial');
    } else {
      status.textContent = `Зачёт: от ${formatPassRequirement(total)}. Нажми Test после исправлений; Run не обновляет статус.`;
      status.classList.add('is-needed');
    }
  }
  fill.style.width = pct + '%';
  fill.className = 'ws-task-progress-fill ' + (passed === total ? 'is-complete' : (passed > 0 ? 'is-partial' : 'is-empty'));
  box.style.display = '';
}

function getLastBlock() {
  try {
    return JSON.parse(localStorage.getItem('last_block') || 'null');
  } catch (e) { return null; }
}

function getCurrentGoal() {
  const saved = getLastBlock() || {};
  return state.goal || saved.goal || {};
}

function getCurrentGoalTitle() {
  const goal = getCurrentGoal();
  return goal.requested_topic || goal.title || (typeof goal === 'string' ? goal : '') || 'Python';
}

function getCurrentLevel() {
  const goal = getCurrentGoal();
  return goal.level || state.level || 'beginner';
}

function showOfflineMode() {
  document.getElementById('goal-title').textContent = 'Демо-режим';
  buildStepper([
    { step_id: 's1', type: 'micro_lesson', title: 'Введение в тему', task_id: null },
    { step_id: 's2', type: 'task',         title: 'Практическое задание', task_id: null },
    { step_id: 's3', type: 'reflection',   title: 'Рефлексия', task_id: null },
  ]);
  editor.setValue('# Напиши своё решение здесь\n');
}

/* ══════════════════════════════════════════════════════════
   TASK LOADING
══════════════════════════════════════════════════════════ */
async function loadTask(taskId, stepIndex) {
  if (!taskId) return;
  state.taskId = taskId;
  try {
    const resp = await fetch(`/api/task/${taskId}?session_id=${encodeURIComponent(state.sessionId)}`, { headers: authHeaders() });
    if (resp.ok) {
      const data = await resp.json();
      renderTaskStatement(data.statement || '');
      const cachedCode = state.stepCodeCache[stepIndex != null ? stepIndex : state.currentStep];
      editor.setValue(cachedCode !== undefined ? cachedCode : (data.starter_code || '# Напиши своё решение здесь\n'));
      editor.clearHistory();
      editor.focus();
    } else if (resp.status === 404) {
      // task_id from LLM is invalid — generate the task instead
      const stepIdx = stepIndex != null ? stepIndex : state.currentStep;
      const step = state.steps[stepIdx];
      if (step) {
        step.task_id = null;
        await generateAndLoadTask(stepIdx, step);
      } else {
        loadFallbackTask();
      }
    } else {
      loadFallbackTask();
    }
  } catch (e) {
    loadFallbackTask();
  }
}

function loadFallbackTask() {
  renderTaskStatement('Задание пока не загружено. Нажми на шаг с заданием в плане слева, чтобы сгенерировать задание.');
  editor.setValue('# Напиши своё решение здесь\n');
}

function renderTaskStatement(mdText) {
  // Simple markdown-like rendering (no full library needed)
  const el = document.getElementById('task-statement');
  let html = mdText
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/^#{1,3}\s+(.+)$/gm, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
  el.innerHTML = html;
}

/* ══════════════════════════════════════════════════════════
   STEPPER
══════════════════════════════════════════════════════════ */
function buildStepper(steps) {
  state.steps = steps;
  // Theory/reflection-like non-code steps may be considered visited when the
  // session resumes on a later step. Code steps are unlocked only by tests.
  for (let i = 0; i < state.currentStep; i++) {
    const step = steps[i];
    if (step && step.type !== 'exercise' && step.type !== 'task') {
      state.completedSteps.add(i);
    }
  }
  const ol = document.getElementById('stepper');
  ol.innerHTML = '';

  steps.forEach((step, idx) => {
    const li = document.createElement('li');
    li.className = 'ws-step';
    if (idx === state.currentStep) li.classList.add('ws-step--active');
    if (idx < state.currentStep)   li.classList.add('ws-step--completed');

    const numEl = document.createElement('span');
    numEl.className = 'ws-step-num';
    const isLockedInit = step.type === 'reflection' && !arePrereqsDone(idx);
    numEl.textContent = isLockedInit ? '🔒' : (idx < state.currentStep ? '✓' : String(idx + 1));

    const typeIcon = { micro_lesson: '📖', exercise: '✏️', task: '🔧', reflection: '💭' }[step.type] || '📌';
    const textEl = document.createElement('span');
    textEl.className = 'ws-step-text';
    textEl.textContent = `${typeIcon} ${step.title}`;

    li.appendChild(numEl);
    li.appendChild(textEl);
    li.addEventListener('click', () => activateStep(idx));
    ol.appendChild(li);
  });

  updateStepIndicator();
}

function goToNextStep() {
  const nextIdx = state.currentStep + 1;
  if (nextIdx < state.steps.length) {
    document.getElementById('next-step-container').style.display = 'none';
    activateStep(nextIdx);
  }
}

function _refreshStepperClasses() {
  document.querySelectorAll('.ws-step').forEach((el, i) => {
    const step = state.steps[i];
    const tr = state.stepTestResults[i];
    const pct = tr && tr.total > 0 ? tr.passed / tr.total : null;
    const isPerfect = isStepPerfectByTests(i);
    const isPassed = isStepAvailableForProgress(i);
    const isPartialPass = !!(step && (step.type === 'exercise' || step.type === 'task') && isPassed && !isPerfect);
    const isCompleted = isPassed && !isPartialPass;
    const prereqsDone = arePrereqsDone(i);
    const isLocked = step && step.type === 'reflection' && !prereqsDone;

    el.classList.toggle('ws-step--active',    i === state.currentStep);
    el.classList.toggle('ws-step--completed', isCompleted && !isLocked);
    el.classList.toggle('ws-step--partial',   isPartialPass && !isLocked);
    el.classList.toggle('ws-step--failed',    !isPassed && pct !== null && pct < PASS_RATIO);
    el.classList.toggle('ws-step--locked',    isLocked);

    const numEl = el.querySelector('.ws-step-num');
    if (numEl) {
      if (isLocked)         numEl.textContent = '🔒';
      else if (isCompleted) numEl.textContent = '✓';
      else if (isPartialPass) numEl.textContent = '•';
      else                  numEl.textContent = String(i + 1);
    }
  });
}

async function activateStep(idx) {
  if (state.isLoadingStep) return;
  const previousStepIndex = state.currentStep;

  // Forward navigation is allowed only when every previous step is passed.
  // Going back is always allowed.
  const targetStep = state.steps[idx];
  if (idx > state.currentStep) {
    const currentStep = state.steps[state.currentStep];
    if (currentStep && currentStep.type !== 'exercise' && currentStep.type !== 'task') {
      state.completedSteps.add(state.currentStep);
      saveProgress(state.sessionId, state.completedSteps);
    }
  }
  if (targetStep && idx > state.currentStep && !arePrereqsDone(idx)) {
    appendChatMessage('assistant', '🔒 Шаги нужно проходить последовательно. Для кодовых заданий достаточно пройти минимум ' +
      `60% автотестов. ${stepProgressHint(state.currentStep)} После этого можно двигаться дальше или доделать задачу до 100%.`);
    return;
  }

  // Сохранить код текущего шага перед переключением (только при реальном переходе на другой шаг)
  if (idx !== state.currentStep) {
    const prevStepForCache = state.steps[state.currentStep];
    if (prevStepForCache && (prevStepForCache.type === 'exercise' || prevStepForCache.type === 'task') && editor) {
      state.stepCodeCache[state.currentStep] = editor.getValue();
      saveStepCode(state.sessionId, state.currentStep, editor.getValue());
    }
  }

  document.getElementById('next-step-container').style.display = 'none';
  state.currentStep = idx;
  const step = state.steps[idx];

  // Notify the server before making this task active. Otherwise a fast click on
  // Run/Test after returning to an earlier task can be rejected as belonging to
  // another active task in the session.
  try {
    const stepResp = await fetch(`/api/session/${state.sessionId}/step`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ step: idx }),
    });
    await readJsonResponse(stepResp, 'Не удалось перейти к выбранному шагу');
  } catch (e) {
    state.currentStep = previousStepIndex;
    const extra = String(e.message || '').includes('автопроверку')
      ? ` ${stepProgressHint(previousStepIndex)}`
      : '';
    appendChatMessage('assistant', `⚠️ ${e.message}${extra}`);
    _refreshStepperClasses();
    updateStepIndicator();
    return;
  }

  // Load content based on step type
  state.isLoadingStep = true;
  try {
    if (step) {
      if (step.type === 'task' || step.type === 'exercise') {
        // Restore left panel task elements
        document.getElementById('step-goal-card').style.display = 'none';
        document.getElementById('task-statement').style.display = '';
        document.getElementById('task-card').querySelector('.ws-criteria-block').style.display = '';
        document.getElementById('task-card').querySelector('.ws-run-row').style.display = '';
        updateTaskMeta(step.type);
        // Restore the test-progress bar for this step if it was run before.
        const prevTr = state.stepTestResults[idx];
        updateTaskProgress(prevTr && prevTr.passed != null ? prevTr.passed : 0, prevTr && prevTr.total != null ? prevTr.total : 0);
        switchToTab('code');
        if (step.task_id) {
          loadTask(step.task_id, idx);
        } else {
          await generateAndLoadTask(idx, step);
        }
      } else if (step.type === 'micro_lesson') {
        await generateAndLoadTheory(idx, step);
      } else if (step.type === 'reflection') {
        showReflectionPanel();
      }
      // Update SRL phase metric — monotonically advancing (Zimmermann cycle).
      // Phase only moves forward; navigating back to an earlier step does NOT revert it.
      const srlPhasePriority = { micro_lesson: 0, exercise: 1, task: 1, reflection: 2 };
      const srlPhaseLabels   = { micro_lesson: '🧭 Планирование', exercise: '⚙️ Выполнение', task: '⚙️ Выполнение', reflection: '🪞 Рефлексия' };
      const newPriority = srlPhasePriority[step.type] != null ? srlPhasePriority[step.type] : -1;
      if (newPriority > state.maxSrlPhaseIdx) {
        state.maxSrlPhaseIdx = newPriority;
        setMetricEl('m-phase', srlPhaseLabels[step.type] || '—');
      }
    }
  } finally {
    if (step) updateTestButtonState(step.type);
    state.isLoadingStep = false;
  }

  // Update visual
  _refreshStepperClasses();
  updateStepIndicator();
}

async function generateAndLoadTask(stepIndex, step) {
  setConsoleStatus('Генерирую задание...');
  showOutput('Генерация задания с помощью ИИ...\n');
  const stopLLMNotice = trackLongLLMRequest('Генерация задания');

  // Get goal title from block data
  const goalTitle = getCurrentGoalTitle();

  try {
    const resp = await fetch('/api/task/generate', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        session_id:      state.sessionId,
        step_index:      stepIndex,
        step_type:       step.type || 'task',
        step_title:      step.title || '',
        step_content_md: step.content_md || '',
        goal_title:      goalTitle,
        level:           getCurrentLevel(),
      }),
    });

    if (!resp.ok) {
      let detail = `Сервер вернул ${resp.status}`;
      try { const err = await resp.json(); detail = err.detail || detail; } catch (e) {}
      throw new Error(detail);
    }

    const data = await readJsonResponse(resp, 'Не удалось сгенерировать задание');
    // Save task_id back to the step and to localStorage so refresh doesn't re-generate
    step.task_id = data.task_id;
    state.taskId = data.task_id;
    saveStepTaskId(state.sessionId, stepIndex, data.task_id);

    renderTaskStatement(data.statement || '');
    const cachedCode = state.stepCodeCache[stepIndex];
    editor.setValue(cachedCode !== undefined ? cachedCode : (data.starter_code || '# Напиши своё решение здесь\n'));
    editor.clearHistory();
    editor.focus();
    setConsoleStatus('');
    showOutput('Задание готово!\n');
  } catch (e) {
    setConsoleStatus('');
    const msg = `⚠️ Не удалось сгенерировать задание:\n${e.message}\n\nПроверьте API: /api/health`;
    showOutput(msg);
    renderTaskStatement(`**Ошибка генерации**\n\n${e.message}\n\nПроверьте: [/api/health](/api/health)`);
    console.error('generateAndLoadTask error:', e);
  } finally {
    stopLLMNotice();
  }
}

function _showTheoryPanel(step) {
  document.getElementById('step-goal-card').style.display = '';
  document.getElementById('step-goal-text').textContent = step.title || '';
  document.getElementById('task-statement').style.display = 'none';
  document.getElementById('task-meta').style.display = 'none';
  document.getElementById('task-test-progress').style.display = 'none';
  document.getElementById('task-card').querySelector('.ws-criteria-block').style.display = 'none';
  document.getElementById('task-card').querySelector('.ws-run-row').style.display = 'none';
  document.getElementById('tab-theory').style.display = '';
  switchToTab('theory');
}

async function generateAndLoadTheory(stepIndex, step) {
  _showTheoryPanel(step);

  // Если теория уже сгенерирована — показать из кэша
  if (state.theoryCache[stepIndex]) {
    document.getElementById('theory-display').innerHTML = state.theoryCache[stepIndex];
    if (stepIndex < state.steps.length - 1) {
      document.getElementById('next-step-container').style.display = '';
    }
    return;
  }

  document.getElementById('theory-display').innerHTML = '<p style="color:#9ca3af">Загружаю теорию...</p>';
  setConsoleStatus('Генерирую теорию...');
  const stopLLMNotice = trackLongLLMRequest('Генерация теории');

  const goalTitle = getCurrentGoalTitle();

  try {
    const resp = await fetch('/api/task/generate', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        session_id:      state.sessionId,
        step_index:      stepIndex,
        step_type:       'micro_lesson',
        step_title:      step.title || '',
        step_content_md: step.content_md || '',
        goal_title:      goalTitle,
        level:           getCurrentLevel(),
      }),
    });
    const data = await readJsonResponse(resp, 'Не удалось сгенерировать теоретический блок');
    const theoryText = data.statement || '';
    const html = renderTheoryMarkdown(theoryText) + _expandButtonHTML();
    state.theoryCache[stepIndex] = html;
    persistWorkspaceState({ theory_cache: state.theoryCache });
    document.getElementById('theory-display').innerHTML = html;
    editor.setValue(data.starter_code || '# Попробуй концепцию в редакторе\n');
    editor.clearHistory();
    setConsoleStatus('');
    if (stepIndex < state.steps.length - 1) {
      document.getElementById('next-step-container').style.display = '';
    }
  } catch (e) {
    setConsoleStatus('');
    const fallback = step.content_md || step.title || 'Изучи эту тему.';
    const html = renderTheoryMarkdown(fallback) + _expandButtonHTML();
    state.theoryCache[stepIndex] = html;
    persistWorkspaceState({ theory_cache: state.theoryCache });
    document.getElementById('theory-display').innerHTML = html;
    if (stepIndex < state.steps.length - 1) {
      document.getElementById('next-step-container').style.display = '';
    }
  } finally {
    stopLLMNotice();
  }
}

function _expandButtonHTML() {
  const isLast = state.currentStep >= state.steps.length - 1;
  const nextBtnHtml = !isLast
    ? `<button class="ws-theory-next-btn" onclick="goToNextStep()">→ Следующий шаг</button>`
    : '';
  return `<div class="ws-theory-expand-section" id="theory-expand-section">
  ${nextBtnHtml}
  <button class="ws-theory-expand-btn" onclick="showExpandForm()">📚 Расширить теорию</button>
  <div class="ws-theory-expand-form" id="theory-expand-form" style="display:none">
    <textarea id="theory-expand-input" class="ws-theory-expand-input"
      placeholder="Что хочешь узнать подробнее? (напр: как работает наследование)" rows="2"></textarea>
    <div class="ws-theory-expand-actions">
      <button onclick="submitExpandTheory()" class="ws-btn ws-btn--run ws-btn--sm">Добавить</button>
      <button onclick="hideExpandForm()" class="ws-btn ws-btn--sm">Отмена</button>
    </div>
  </div>
</div>`;
}

function showExpandForm() {
  const f = document.getElementById('theory-expand-form');
  if (f) f.style.display = '';
}

function hideExpandForm() {
  const f = document.getElementById('theory-expand-form');
  if (f) f.style.display = 'none';
}

async function submitExpandTheory() {
  if (theoryExpandInFlight) return;
  const input = document.getElementById('theory-expand-input');
  const userRequest = input ? input.value.trim() : '';
  if (!userRequest) return;
  theoryExpandInFlight = true;
  const stopLLMNotice = trackLongLLMRequest('Расширение теории');

  const goalTitle = getCurrentGoalTitle();
  const currentStepData = state.steps[state.currentStep] || {};
  const baseTopic = currentStepData.title || goalTitle;

  const section = document.getElementById('theory-expand-section');
  if (section) {
    section.insertAdjacentHTML('beforebegin',
      '<div class="ws-theory-expansion ws-theory-expansion--loading" id="theory-expansion-loading"><em style="color:#9ca3af">Генерирую раздел...</em></div>'
    );
  }
  if (input) input.value = '';
  hideExpandForm();

  try {
    const resp = await fetch('/api/theory/expand', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        session_id:   state.sessionId,
        goal_title:   goalTitle,
        base_topic:   baseTopic,
        user_request: userRequest,
      }),
    });
    const data = await readJsonResponse(resp, 'Не удалось расширить теоретический блок');
    const html = renderTheoryMarkdown(data.statement || '');
    const loader = document.getElementById('theory-expansion-loading');
    if (loader) {
      loader.outerHTML = `<div class="ws-theory-expansion"><hr class="ws-theory-divider"><h3>📌 ${escapeHtml(userRequest)}</h3>${html}</div>`;
    }
    // Обновить кэш
    state.theoryCache[state.currentStep] = document.getElementById('theory-display').innerHTML;
  } catch (e) {
    const loader = document.getElementById('theory-expansion-loading');
    if (loader) loader.remove();
  } finally {
    stopLLMNotice();
    theoryExpandInFlight = false;
  }
}

function showStepContent(md) {
  renderTaskStatement(md);
  editor.setValue('# Попробуй концепцию в редакторе\n');
}

/* ══════════════════════════════════════════════════════════
   REFLECTION PANEL (SRL Self-Reflection phase)
══════════════════════════════════════════════════════════ */
let _reflRating = 0;

function buildReflectionSummaryHTML(data) {
  const metrics = (data && data.actual_metrics) || null;
  const metricHtml = metrics ? `
      <div class="ws-refl-metrics">
        <span>✅ Точность: ${escapeHtml(String(metrics.accuracy != null ? metrics.accuracy : '—'))}</span>
        <span>💡 Подсказки: ${escapeHtml(String(metrics.hints != null ? metrics.hints : '—'))}</span>
        <span>🕒 Шагов: ${escapeHtml(String(metrics.steps_done != null ? metrics.steps_done : '—'))}</span>
      </div>` : '';
  return `
      <div class="ws-refl-summary-title">📋 Итог занятия</div>
      <div class="ws-refl-summary-text">${escapeHtml((data && data.summary) || '—')}</div>
      ${metricHtml}
      <button class="ws-refl-restart-btn" onclick="endSessionAndRestart()">🔄 Завершить и начать заново</button>`;
}

function showNewSessionButton() {
  const newSessionContainer = document.getElementById('new-session-btn-container');
  const newSessionBtn = document.getElementById('new-session-btn');
  if (newSessionContainer) newSessionContainer.style.display = '';
  if (newSessionBtn) newSessionBtn.onclick = endSessionAndRestart;
}

function showReflectionPanel() {
  // Hide editor, show reflection panel
  document.getElementById('code-editor').style.display = 'none';
  document.getElementById('theory-display').style.display = 'none';
  document.getElementById('reflection-panel').style.display = '';

  // Hide left-panel elements that don't apply to reflection
  document.getElementById('step-goal-card').style.display = 'none';
  document.getElementById('task-statement').style.display = 'none';
  document.getElementById('task-meta').style.display = 'none';
  document.getElementById('task-test-progress').style.display = 'none';
  const card = document.getElementById('task-card');
  if (card) {
    const cb = card.querySelector('.ws-criteria-block');
    const rr = card.querySelector('.ws-run-row');
    if (cb) cb.style.display = 'none';
    if (rr) rr.style.display = 'none';
  }

  // If the reflection summary was already submitted — restore it from cache
  if (state.reflSummaryHTML) {
    document.getElementById('refl-form').style.display = 'none';
    const summaryEl = document.getElementById('refl-summary');
    summaryEl.innerHTML = state.reflSummaryHTML;
    if (!summaryEl.querySelector('.ws-refl-restart-btn')) {
      summaryEl.insertAdjacentHTML('beforeend', '<button class="ws-refl-restart-btn" onclick="endSessionAndRestart()">🔄 Завершить и начать заново</button>');
    }
    summaryEl.style.display = '';
    showNewSessionButton();
    return;
  }

  // Reset form for first visit
  _reflRating = 0;
  document.querySelectorAll('.ws-refl-star').forEach(b => b.classList.remove('selected'));
  ['refl-success', 'refl-difficulty', 'refl-next'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  document.getElementById('refl-form').style.display = '';
  document.getElementById('refl-summary').style.display = 'none';
}

function hideReflectionPanel() {
  document.getElementById('reflection-panel').style.display = 'none';
  document.getElementById('code-editor').style.display = '';
}

function selectRating(v) {
  _reflRating = v;
  document.querySelectorAll('.ws-refl-star').forEach(b => {
    b.classList.toggle('selected', parseInt(b.dataset.v) <= v);
  });
}

async function submitReflection() {
  const btn = document.querySelector('.ws-refl-submit-btn');
  if (!btn) return;

  const successEl = document.getElementById('refl-success');
  const whatWorked  = ((successEl && successEl.value) || '').trim();
  const difficultyEl = document.getElementById('refl-difficulty');
  const difficulty  = ((difficultyEl && difficultyEl.value) || '').trim();
  const nextTimeEl = document.getElementById('refl-next');
  const nextTime    = ((nextTimeEl && nextTimeEl.value) || '').trim();

  if (!_reflRating) {
    alert('Поставь оценку от 1 до 5');
    return;
  }
  if (!whatWorked && !difficulty && !nextTime) {
    alert('Заполни хотя бы одно текстовое поле рефлексии');
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Отправляем...';

  try {
    const resp = await fetch(`/api/session/${state.sessionId}/reflect`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        self_rating:  _reflRating,
        what_worked:  whatWorked  || '—',
        difficulty:   difficulty  || '—',
        next_time:    nextTime    || '—',
      }),
    });

    const data = await readJsonResponse(resp, 'Не удалось сохранить рефлексию');
    const summaryEl = document.getElementById('refl-summary');
    summaryEl.innerHTML = buildReflectionSummaryHTML(data);
    document.getElementById('refl-form').style.display = 'none';
    summaryEl.style.display = '';
    // Cache summary so it survives step navigation
    state.reflSummaryHTML = summaryEl.innerHTML;
    // Mark reflection step as completed (green) permanently
    state.completedSteps.add(state.currentStep);
    saveProgress(state.sessionId, state.completedSteps);
    _refreshStepperClasses();
    // Show fresh-session button after reflection is done
    showNewSessionButton();
  } catch (e) {
    btn.disabled = false;
    btn.textContent = '📤 Отправить и получить итог занятия';
    console.error('Reflection submit error:', e);
  }
}

function endSessionAndRestart() {
  if (state.sessionId) {
    try {
      sessionStorage.removeItem(`session_token_${state.sessionId}`);
      sessionStorage.removeItem(`session_api_key_${state.sessionId}`);
      sessionStorage.removeItem(`session_api_provider_${state.sessionId}`);
      localStorage.removeItem(`ws_codes_${state.sessionId}`);
      localStorage.removeItem(`ws_progress_${state.sessionId}`);
      localStorage.removeItem(`ws_taskids_${state.sessionId}`);
      localStorage.removeItem('last_session_id');
    } catch (e) {}
  }
  window.location.href = '/';
}

function updateTestButtonState(stepType) {
  const isCode = stepType === 'task' || stepType === 'exercise';
  document.querySelectorAll('.ws-btn--test').forEach(btn => {
    btn.disabled = !isCode;
    btn.style.opacity = isCode ? '1' : '0.4';
  });
  const titleEl = document.querySelector('#task-card .ws-section-title');
  if (titleEl) {
    const titles = {
      micro_lesson: '📖 Теория',
      exercise:     '✏️ Упражнение',
      task:         '🔧 Задание',
      reflection:   '💭 Рефлексия',
    };
    titleEl.textContent = titles[stepType] || '📝 Текущее задание';
  }
}

function updateStepIndicator() {
  const el = document.getElementById('step-indicator');
  const total = state.steps.length;
  if (total > 0) {
    el.textContent = `Шаг ${state.currentStep + 1} / ${total}`;
  }
}

/* ══════════════════════════════════════════════════════════
   CHAT WebSocket
══════════════════════════════════════════════════════════ */
function connectTutorWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws/tutor/${state.sessionId}?token=${encodeURIComponent(state.sessionToken)}`;

  tutorWS = new WebSocket(url);

  tutorWS.onopen = () => {
    setWSStatus('connected');
    // Очистить стейт оборванного стрима при переподключении
    if (currentBotMsgEl) {
      currentBotMsgEl = null;
    }
    const chatContainer = document.getElementById('chat-messages');
    if (chatContainer) {
      const typing = chatContainer.querySelector('.ws-msg--typing');
      if (typing) typing.remove();
    }
    clearInterval(tutorPingInterval);
    tutorPingInterval = setInterval(() => {
      if (tutorWS && tutorWS.readyState === WebSocket.OPEN) tutorWS.send('ping');
    }, 25000);
  };

  tutorWS.onmessage = (e) => handleTutorMessage(e.data);

  tutorWS.onerror = () => setWSStatus('error');
  tutorWS.onclose = () => {
    setWSStatus('error');
    clearInterval(tutorPingInterval);
    tutorPingInterval = null;
    // Reconnect after 3s
    setTimeout(connectTutorWS, 3000);
  };
}

function handleTutorMessage(raw) {
  // Support both old plain-text STREAM_* protocol and new JSON format
  if (raw === 'pong') return;

  // Try JSON first
  if (raw.startsWith('{')) {
    try {
      const msg = JSON.parse(raw);
      if (msg.type === 'tutor_message' && msg.role === 'assistant') {
        appendChatMessage('assistant', msg.content || '');
      }
      return;
    } catch (e) {}
  }

  if (raw === 'STREAM_START') {
    currentBotMsgEl = createBotMessage();
    return;
  }
  if (raw.startsWith('STREAM_CHUNK:')) {
    const chunk = raw.slice('STREAM_CHUNK:'.length);
    if (currentBotMsgEl) appendChunk(currentBotMsgEl, chunk);
    return;
  }
  if (raw === 'STREAM_DONE') {
    if (currentBotMsgEl) {
      const rawText = currentBotMsgEl.textContent;
      currentBotMsgEl.innerHTML = renderMarkdown(rawText);
    }
    currentBotMsgEl = null;
    scrollChatToBottom();
    return;
  }
  if (raw.startsWith('STREAM_ERROR:')) {
    const errText = raw.slice('STREAM_ERROR:'.length);
    if (currentBotMsgEl) appendChunk(currentBotMsgEl, `\n⚠️ Ошибка: ${errText}`);
    currentBotMsgEl = null;
    return;
  }
  if (raw.startsWith('FEEDBACK_REQUEST:')) {
    // ignore for now
    return;
  }

  // Fallback: plain text
  appendChatMessage('assistant', raw);
}

/* ── Chat rendering helpers ── */
function createBotMessage() {
  const container = document.getElementById('chat-messages');
  const typing = container.querySelector('.ws-msg--typing');
  if (typing) typing.remove();
  const msgDiv = document.createElement('div');
  msgDiv.className = 'ws-msg ws-msg--assistant';
  const contentDiv = document.createElement('div');
  contentDiv.className = 'ws-msg-content';
  msgDiv.appendChild(contentDiv);
  container.appendChild(msgDiv);
  scrollChatToBottom();
  return contentDiv;
}

function appendChunk(contentEl, chunk) {
  contentEl.textContent += chunk;
  scrollChatToBottom();
}

function appendChatMessage(role, text) {
  const container = document.getElementById('chat-messages');

  // Remove typing indicator if present
  const typing = container.querySelector('.ws-msg--typing');
  if (typing) typing.remove();

  const msgDiv = document.createElement('div');
  msgDiv.className = `ws-msg ws-msg--${role}`;
  const contentDiv = document.createElement('div');
  contentDiv.className = 'ws-msg-content';

  // Basic markdown for assistant messages
  if (role === 'assistant') {
    contentDiv.innerHTML = renderMarkdown(text);
  } else {
    contentDiv.textContent = text;
  }

  msgDiv.appendChild(contentDiv);
  container.appendChild(msgDiv);
  scrollChatToBottom();
}

function showTypingIndicator() {
  const container = document.getElementById('chat-messages');
  if (container.querySelector('.ws-msg--typing')) return;
  const msgDiv = document.createElement('div');
  msgDiv.className = 'ws-msg ws-msg--assistant ws-msg--typing';
  const contentDiv = document.createElement('div');
  contentDiv.className = 'ws-msg-content';
  contentDiv.innerHTML = '<span class="ws-typing-dot"></span><span class="ws-typing-dot"></span><span class="ws-typing-dot"></span>';
  msgDiv.appendChild(contentDiv);
  container.appendChild(msgDiv);
  scrollChatToBottom();
}

function scrollChatToBottom() {
  const el = document.getElementById('chat-messages');
  el.scrollTop = el.scrollHeight;
}

function renderMarkdown(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```python\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/```\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function renderTheoryMarkdown(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```python\n?([\s\S]*?)```/g, (_, code) =>
      `<pre><code>${code}</code></pre>`)
    .replace(/```\n?([\s\S]*?)```/g, (_, code) =>
      `<pre><code>${code}</code></pre>`)
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
}

function switchToTab(tab) {
  // Always hide reflection panel when switching to a code/theory tab
  document.getElementById('reflection-panel').style.display = 'none';
  const isTheory = tab === 'theory';
  document.getElementById('code-editor').style.display     = isTheory ? 'none' : '';
  document.getElementById('theory-display').style.display  = isTheory ? '' : 'none';
  document.getElementById('tab-main').classList.toggle('ws-file-tab--active', !isTheory);
  document.getElementById('tab-theory').classList.toggle('ws-file-tab--active', isTheory);
}

/* ── Sending messages ── */
function sendChatMessage(text) {
  if (!text.trim()) return;
  if (!tutorWS || tutorWS.readyState !== WebSocket.OPEN) {
    appendChatMessage('assistant', '⚠️ Нет соединения с тьютором. Обновите страницу.');
    return;
  }
  appendChatMessage('user', text);
  showTypingIndicator();
  tutorWS.send(text);
}

function sendQuick(command) {
  const commandLabels = {
    '/ask_question':    'Задай мне наводящий вопрос',
    '/hint':            'Дай мне подсказку',
    '/explain_failure': 'Разбери, почему упал тест',
    '/review_code':     'Проверь мой код',
  };
  const label = commandLabels[command] || command;

  let msg = command;

  // /explain_failure — добавить контекст упавших тестов
  if (command === '/explain_failure' && state.lastFailedTests.length > 0) {
    const failInfo = state.lastFailedTests
      .map(t => `${t.name}: ${t.error || ''}`)
      .join('\n');
    msg = `${command}\nПровалившиеся тесты:\n${failInfo}`;
  }

  // /review_code — добавить текущий код из редактора
  if (command === '/review_code') {
    const code = editor.getValue();
    if (code.trim()) msg = `${command}\n${code}`;
  }

  if (!tutorWS || tutorWS.readyState !== WebSocket.OPEN) {
    appendChatMessage('assistant', '⚠️ Нет соединения с тьютором. Дождитесь переподключения.');
    return;
  }
  appendChatMessage('user', label);
  showTypingIndicator();
  tutorWS.send(msg);
}

function updateQuickBtns() {
  const btn = document.getElementById('btn-explain-failure');
  if (btn) btn.disabled = state.lastFailedTests.length === 0;
}

/* ── Chat input handling ── */
function initChatInput() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');

  sendBtn.addEventListener('click', () => {
    const text = input.value.trim();
    if (text) {
      sendChatMessage(text);
      input.value = '';
      input.style.height = 'auto';
    }
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const text = input.value.trim();
      if (text) {
        sendChatMessage(text);
        input.value = '';
        input.style.height = 'auto';
      }
    }
  });

  // Auto-resize textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 100) + 'px';
  });
}

/* ══════════════════════════════════════════════════════════
   RUN CODE
══════════════════════════════════════════════════════════ */
async function runCode() {
  const code = editor.getValue();
  if (!code.trim()) return;

  setConsoleStatus('Запуск...');
  showOutput('Запуск кода...\n');
  switchConsole('output');

  try {
    const resp = await fetch('/api/run', {
      method:  'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body:    JSON.stringify({
        session_id: state.sessionId,
        task_id:    state.taskId,
        code:       code,
      }),
    });

    const result = await readJsonResponse(resp, 'Не удалось выполнить код');
    const output = result.output || '(нет вывода)';
    state.lastOutput = output;
    showOutput(output);
    setConsoleStatus('');
  } catch (e) {
    const isNetErr = !e.message || e.message.includes('pattern') || e.message.includes('fetch') || e.message.includes('network');
    showOutput(isNetErr ? '⚠️ Сервер недоступен. Проверьте соединение и перезагрузите страницу.' : `⚠️ Ошибка запуска: ${e.message}`);
    setConsoleStatus('');
  }
}

/* ══════════════════════════════════════════════════════════
   TEST CODE (Autograde)
══════════════════════════════════════════════════════════ */
async function testCode() {
  const code = editor.getValue();
  if (!code.trim()) return;

  setConsoleStatus('Тестирование...');

  try {
    const resp = await fetch('/api/autograde', {
      method:  'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body:    JSON.stringify({
        session_id: state.sessionId,
        task_id:    state.taskId,
        files:      { 'starter_code.py': code },
      }),
    });

    const result = await readJsonResponse(resp, 'Не удалось запустить тесты');
    setConsoleStatus('');
    renderTestResults(result);
    const summary = result.summary || {};
    const passed = Number(summary.passed) || 0;
    const total = Number(summary.total) || 0;
    const requiredPassed = requiredPassedCount(total);
    const passedAll = total > 0 && passed === total;
    const passedEnough = total > 0 && passed >= requiredPassed;

    // Auto-report failures to tutor
    const failed = (result.tests || []).filter(t => t.status === 'failed' || t.status === 'error');
    state.lastFailedTests = failed;
    updateQuickBtns();

    if (failed.length > 0 && !passedEnough) {
      const studentCode = editor.getValue() || '';
      const testDetails = failed.map(t => {
        let detail = `❌ ${t.name}`;
        if (t.error) detail += `: ${t.error}`;
        if (t.trace) detail += `\n${t.trace}`;
        return detail;
      }).join('\n');

      const failMsg = [
        `AutogradeResult: упало тестов ${failed.length} из ${(result.summary && result.summary.total) || '?'}`,
        '',
        '=== КОД СТУДЕНТА ===',
        studentCode,
        '',
        '=== УПАВШИЕ ТЕСТЫ ===',
        testDetails,
      ].join('\n');

      // Auto-send to tutor
      if (tutorWS && tutorWS.readyState === WebSocket.OPEN) {
        showTypingIndicator();
        try { tutorWS.send(failMsg); } catch (e) { const t = document.querySelector('.ws-msg--typing'); if (t) t.remove(); console.warn('WS send error:', e); }
      }
    }

    if (result.summary) {
      state.stepTestResults[state.currentStep] = { passed, total };
      updateTestMetric(passed, total);
      _refreshStepperClasses();

      if (passedEnough) {
        state.completedSteps.add(state.currentStep);
        saveProgress(state.sessionId, state.completedSteps);
        _refreshStepperClasses();
        if (state.currentStep < state.steps.length - 1) {
          if (passedAll) {
            appendChatMessage('assistant', '🎉 Все тесты прошли! Отличная работа.');
          } else {
            appendChatMessage('assistant', `✅ Шаг зачтён: пройдено ${passed}/${total} тестов, зачёт от ${formatPassRequirement(total)}. Можно перейти дальше или доделать задачу до 100%.`);
          }
          document.getElementById('next-step-container').style.display = '';
        } else {
          appendChatMessage('assistant', '🏆 Поздравляю! Ты завершил все шаги занятия.');
        }
      } else if (total > 0) {
        appendChatMessage('assistant', `Пока пройдено ${passed}/${total} тестов. Для перехода дальше нужно минимум ${requiredPassed}/${total}; исправь решение и нажми Test ещё раз. Run не обновляет статус шага.`);
      }
    }
  } catch (e) {
    setConsoleStatus('');
    const isNetErr = !e.message || e.message.includes('pattern') || e.message.includes('fetch') || e.message.includes('network');
    appendChatMessage('assistant', isNetErr ? '⚠️ Сервер недоступен. Проверьте соединение и перезагрузите страницу.' : `⚠️ Не удалось запустить тесты: ${e.message}`);
  }
}

function humanizeTestName(name) {
  return name
    .replace(/^test_/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

function explainError(t) {
  switch (t.error_kind) {
    case 'assertion':
      return 'Результат не совпал с ожидаемым.';
    case 'name_error': {
      const m = (t.error_message || '').match(/name '(.+?)' is not defined/);
      return m
        ? `Имя <code>${escapeHtml(m[1])}</code> не определено — проверь, объявлена ли функция или переменная.`
        : 'Обращение к необъявленному имени. Проверь названия функций и переменных.';
    }
    case 'type_error':
      return `Ошибка типа данных — функция получила не тот тип аргумента.<br><small>${escapeHtml(t.error_message || '')}</small>`;
    case 'syntax_error':
      return 'Синтаксическая ошибка — Python не смог прочитать твой код. Проверь скобки, двоеточия и отступы.';
    case 'recursion':
      return 'Переполнение стека — возможно, рекурсия не имеет базового случая и зацикливается.';
    default:
      return escapeHtml(t.error_message || t.error || 'Тест завершился с ошибкой.');
  }
}

function renderTestResults(result) {
  const panel   = document.getElementById('test-results');
  const summary = document.getElementById('test-summary');
  const list    = document.getElementById('test-list');
  const action  = document.getElementById('test-next-action');

  panel.style.display = 'block';

  const passed = result.summary && result.summary.passed != null ? result.summary.passed : 0;
  const failed = result.summary && result.summary.failed != null ? result.summary.failed : 0;
  const total  = result.summary && result.summary.total != null ? result.summary.total : 0;
  const required = requiredPassedCount(total);
  const passedEnough = total > 0 && passed >= required;
  const passedAll = total > 0 && passed === total;

  summary.textContent = total > 0
    ? `${passed} / ${total} прошло · зачёт от ${formatPassRequirement(total)}`
    : `${passed} / ${total} прошло`;
  summary.className = 'ws-test-summary ' + (
    passedAll ? 'all-pass' : (passedEnough ? 'partial-pass' : 'has-fail')
  );
  updateTaskProgress(passed, total);

  if (total === 0 && result.raw_output) {
    showOutput('⚠️ pytest не нашёл тесты или завершился с ошибкой:\n' + result.raw_output);
  }

  if (action) {
    action.innerHTML = '';
    action.style.display = total > 0 ? '' : 'none';
    if (total > 0 && passedEnough && state.currentStep < state.steps.length - 1) {
      const msg = document.createElement('span');
      msg.className = passedAll ? 'ws-test-next-msg is-complete' : 'ws-test-next-msg is-partial';
      msg.textContent = passedAll
        ? 'Все тесты пройдены. Можно переходить дальше.'
        : `Этап зачтён частично (${passed}/${total}). Можно перейти дальше или довести до 100%.`;
      const btn = document.createElement('button');
      btn.className = 'ws-test-next-btn';
      btn.textContent = '→ Перейти дальше';
      btn.addEventListener('click', goToNextStep);
      action.appendChild(msg);
      action.appendChild(btn);
    } else if (total > 0 && passedEnough) {
      const msg = document.createElement('span');
      msg.className = passedAll ? 'ws-test-next-msg is-complete' : 'ws-test-next-msg is-partial';
      msg.textContent = passedAll
        ? 'Финальный кодовый шаг пройден на 100%.'
        : `Финальный кодовый шаг зачтён (${passed}/${total}); можно довести до 100%.`;
      action.appendChild(msg);
    } else if (total > 0) {
      const msg = document.createElement('span');
      msg.className = 'ws-test-next-msg is-needed';
      msg.textContent = `Для перехода дальше нужно минимум ${formatPassRequirement(total)}. После исправлений нажми Test; Run не обновляет статус шага.`;
      action.appendChild(msg);
    }
  }

  list.innerHTML = '';
  (result.tests || []).forEach(t => {
    const item = document.createElement('div');
    item.className = `ws-test-item ${t.status}`;

    const icon   = t.status === 'passed' ? '✅' : (t.status === 'failed' ? '❌' : '⚠️');
    const label  = t.status === 'passed' ? 'PASS' : (t.status === 'failed' ? 'FAIL' : 'ERR');

    const header = document.createElement('div');
    header.className = 'ws-test-item-header';
    header.innerHTML = `
      <span class="ws-test-icon">${icon}</span>
      <span class="ws-test-name">${escapeHtml(humanizeTestName(t.name))}</span>
      <span class="ws-test-status">${label}</span>`;

    item.appendChild(header);

    if (t.status !== 'passed') {
      // Plain-language explanation (always visible)
      const explain = document.createElement('div');
      explain.className = 'ws-test-explain';

      if (t.error_kind === 'assertion' && t.got != null && t.expected != null) {
        explain.innerHTML = `
          <div class="ws-got-expected">
            <div class="ws-ge-row ws-ge-got">
              <span class="ws-ge-label">Полученный вывод</span>
              <code class="ws-ge-value">${escapeHtml(t.got)}</code>
            </div>
            <div class="ws-ge-row ws-ge-expected">
              <span class="ws-ge-label">Ожидаемый вывод</span>
              <code class="ws-ge-value">${escapeHtml(t.expected)}</code>
            </div>
            ${t.error_line ? `<div class="ws-ge-line">📍 Строка в коде: <strong>${escapeHtml(String(t.error_line))}</strong></div>` : ''}
          </div>`;
      } else {
        explain.innerHTML = `<div class="ws-explain-msg">${explainError(t)}</div>`;
        if (t.error_line) {
          explain.innerHTML += `<div class="ws-ge-line">📍 Строка в коде: <strong>${escapeHtml(String(t.error_line))}</strong></div>`;
        }
      }
      item.appendChild(explain);

      // Collapsible raw trace
      if (t.error || t.trace) {
        const toggle = document.createElement('button');
        toggle.className = 'ws-test-detail-toggle';
        toggle.textContent = 'Технические подробности ▼';
        toggle.addEventListener('click', (e) => {
          e.stopPropagation();
          item.classList.toggle('trace-open');
          toggle.textContent = item.classList.contains('trace-open')
            ? 'Технические подробности ▲'
            : 'Технические подробности ▼';
        });

        const trace = document.createElement('div');
        trace.className = 'ws-test-trace';
        if (t.trace) {
          trace.innerHTML = t.trace.split('\n').map(line => {
            const escaped = escapeHtml(line);
            if (line.startsWith('E ') || line.startsWith('E\t')) return `<span class="trace-error-line">${escaped}</span>`;
            if (line.trimStart().startsWith('assert ')) return `<span class="trace-assert-line">${escaped}</span>`;
            return `<span class="trace-line">${escaped}</span>`;
          }).join('\n');
        } else {
          trace.textContent = t.error || '';
        }

        item.appendChild(toggle);
        item.appendChild(trace);
      }
    }

    list.appendChild(item);
  });
}

function closeTestResults() {
  document.getElementById('test-results').style.display = 'none';
}

/* ══════════════════════════════════════════════════════════
   OUTPUT CONSOLE
══════════════════════════════════════════════════════════ */
function showOutput(text) {
  const el = document.getElementById('output-display');
  el.textContent = text;
  el.scrollTop = el.scrollHeight;
}
function setConsoleStatus(text) { document.getElementById('console-status').textContent = text; }
function switchConsole() { document.getElementById('output-display').style.display = 'block'; }
function toggleConsole() { document.getElementById('ws-console').classList.toggle('collapsed'); }

/* ══════════════════════════════════════════════════════════
   METRICS POLLING
══════════════════════════════════════════════════════════ */
function pollMetrics() {
  fetchMetrics();
  clearInterval(metricsInterval);
  metricsInterval = setInterval(fetchMetrics, 8000);
}

async function fetchMetrics() {
  try {
    const [metricsResp, affResp] = await Promise.all([
      fetch(`/api/metrics/${state.sessionId}`, { headers: authHeaders() }),
      fetch(`/api/affective/${state.sessionId}`, { headers: authHeaders() }),
    ]);

    if (metricsResp.ok) {
      const d = await metricsResp.json();
      const m = d.metrics || {};
      setMetricEl('m-accuracy', formatPercent(m.accuracy));
      setMetricEl('m-hints',    m.hints_used != null ? String(m.hints_used) : '—');
    }
    // m-phase is updated structurally in activateStep() from step.type (Zimmermann SRL cycle)
    if (affResp.ok) {
      const d = await affResp.json();
      const labelMap = { positive: '😊 Позитив', neutral: '😐 Нейтрально', frustrated: '😤 Стресс', bored: '😑 Скука', confused: '😕 Замешательство' };
      const affect = d.state || d.affective || {};
      setMetricEl('m-affect', labelMap[affect.label] || '—');
    }
  } catch (e) { console.warn('fetchMetrics error:', e); }
}

function setMetricEl(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}
function updateTestMetric(passed, total) {
  if (total > 0) setMetricEl('m-accuracy', `${passed}/${total}`);
}
function formatPercent(v) { return (v != null) ? `${Math.round(v * 100)}%` : '—'; }
function formatDecimal(v) { return (v != null) ? v.toFixed(2) : '—'; }

/* ══════════════════════════════════════════════════════════
   TIMER
══════════════════════════════════════════════════════════ */
function initTimer() {
  const savedStart = sessionStorage.getItem(`timer_${state.sessionId}`);
  const startTs = savedStart ? parseInt(savedStart) : Date.now();
  if (!savedStart) sessionStorage.setItem(`timer_${state.sessionId}`, startTs);

  function tick() {
    const elapsed = Math.floor((Date.now() - startTs) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    document.getElementById('session-timer').textContent = `${m}:${s}`;
  }
  tick();
  clearInterval(state.timerInterval);
  state.timerInterval = setInterval(tick, 1000);
}

/* ══════════════════════════════════════════════════════════
   WS STATUS
══════════════════════════════════════════════════════════ */
function setWSStatus(status) {
  const el = document.getElementById('ws-status');
  el.className = `ws-ws-status ws-ws-status--${status}`;
  const labels = { connecting: '● WS', connected: '● WS', error: '● WS' };
  el.textContent = labels[status] || '● WS';
}

/* ══════════════════════════════════════════════════════════
   MODE TOGGLE
══════════════════════════════════════════════════════════ */
const _levelLabel = { beginner: 'Новичок', intermediate: 'Средний', advanced: 'Продвинутый' };

function initLevelButton(level) {
  state.level = level;
  const btn = document.getElementById('mode-toggle');
  if (btn) btn.textContent = _levelLabel[level] || 'Новичок';
}


/* ══════════════════════════════════════════════════════════
   MOBILE PANEL SWITCHING
══════════════════════════════════════════════════════════ */
function showMobilePanel(panel) {
  const panelMap = { plan: 'panel-plan', code: 'panel-code', chat: 'panel-chat' };

  document.getElementById('panel-plan').classList.remove('mobile-active');
  document.getElementById('panel-code').classList.remove('mobile-active');
  document.getElementById('panel-chat').classList.remove('mobile-active');

  const target = panelMap[panel];
  if (target) document.getElementById(target).classList.add('mobile-active');

  // Update mobile tab highlight
  document.querySelectorAll('.ws-mobile-tab').forEach((tab, i) => {
    const panels = ['plan', 'code', 'chat'];
    tab.classList.toggle('active', panels[i] === panel);
  });

  // Refresh editor size
  if (panel === 'code' && editor) {
    setTimeout(() => editor.refresh(), 50);
  }
}

// Activate "code" panel by default on mobile
document.addEventListener('DOMContentLoaded', () => {
  if (window.innerWidth <= 1024) {
    showMobilePanel('code');
  }
});

/* ══════════════════════════════════════════════════════════
   UTILS
══════════════════════════════════════════════════════════ */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
