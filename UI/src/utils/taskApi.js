// 后台任务 API — 异步启动长时间操作 + 轮询进度
// API_BASE_URL + apiHeaders 统一从 api.js 导入,后者负责注入 X-API-Key
import { API_BASE_URL, apiHeaders } from './api';

const jsonHeaders = () => apiHeaders({ 'Content-Type': 'application/json' });

// -------------------------------------------------------------------
// 启动任务
// -------------------------------------------------------------------

export const startPreprocessTask = async (params) => {
  const response = await fetch(`${API_BASE_URL}/tasks/preprocess`, {
    method: 'POST',
    headers: jsonHeaders(),
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || '启动预处理任务失败');
  }
  return response.json();  // { task_id, status }
};

export const startIngestTask = async (params) => {
  const response = await fetch(`${API_BASE_URL}/tasks/ingest`, {
    method: 'POST',
    headers: jsonHeaders(),
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || '启动入库任务失败');
  }
  return response.json();
};

export const startDedupTask = async (params = {}) => {
  const response = await fetch(`${API_BASE_URL}/tasks/dedup`, {
    method: 'POST',
    headers: jsonHeaders(),
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || '启动去重任务失败');
  }
  return response.json();
};

// -------------------------------------------------------------------
// 查询状态
// -------------------------------------------------------------------

export const cancelTask = async (taskId) => {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/cancel`, {
    method: 'POST',
    headers: apiHeaders(),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || '取消任务失败');
  }
  return response.json();  // { task_id, status: "cancelled", message }
};

export const getTaskStatus = async (taskId) => {
  const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
    headers: apiHeaders(),
  });
  if (!response.ok) {
    if (response.status === 404) return null;  // 任务不存在(进程重启后)
    throw new Error('查询任务状态失败');
  }
  return response.json();
};

export const getActiveTasks = async () => {
  const response = await fetch(`${API_BASE_URL}/tasks/active`, {
    headers: apiHeaders(),
  });
  if (!response.ok) throw new Error('查询运行任务失败');
  return response.json();  // { tasks: [...] }
};

export const getLatestTask = async (taskType) => {
  const response = await fetch(`${API_BASE_URL}/tasks/latest/${taskType}`, {
    headers: apiHeaders(),
  });
  if (!response.ok) throw new Error('查询最新任务失败');
  return response.json();  // TaskInfo dict or { task_id: null, status: "none" }
};

// -------------------------------------------------------------------
// 轮询直到完成
// -------------------------------------------------------------------

/**
 * 轮询任务状态直到完成、失败或取消。
 *
 * @param {string} taskId
 * @param {function} onUpdate - 每次收到状态更新时调用,参数为 TaskInfo dict
 * @param {function} onComplete - 任务完成时调用
 * @param {function} onError - 任务失败时调用
 * @param {function} onCancel - 任务被取消时调用 (可选)
 * @param {number} intervalMs - 轮询间隔(毫秒),默认 2000
 * @returns {number} timer ID — 调用方可 clearInterval(timer) 停止轮询
 *
 * 重叠保护: 若上次 poll 仍在飞行,本次 tick 直接跳过(避免请求堆叠).
 * 容忍 404: 后端短暂未注册任务时,先重试若干次再放弃.
 */
export const pollTaskUntilDone = (taskId, onUpdate, onComplete, onError, onCancel, intervalMs = 2000) => {
  const timerRef = { id: null };
  let inFlight = false;
  let consecutiveErrors = 0;
  let consecutiveNotFound = 0;
  const MAX_CONSECUTIVE_ERRORS = 3;
  const MAX_CONSECUTIVE_NOTFOUND = 3;  // 容忍 3 次 404 — 任务可能刚注册或后端短暂卡顿

  const clearTimer = () => {
    if (timerRef.id !== null) {
      clearInterval(timerRef.id);
      timerRef.id = null;
    }
  };

  const poll = async () => {
    if (inFlight) return;          // overlap guard: 上次 poll 还没回来,跳过这次 tick
    inFlight = true;
    try {
      const status = await getTaskStatus(taskId);
      if (!status) {
        consecutiveNotFound++;
        if (consecutiveNotFound >= MAX_CONSECUTIVE_NOTFOUND) {
          clearTimer();
          if (onError) onError({ error: '任务状态丢失(后端可能重启)' });
        }
        return;
      }
      consecutiveErrors = 0;
      consecutiveNotFound = 0;
      if (onUpdate) onUpdate(status);
      if (status.status === 'completed') {
        clearTimer();
        if (onComplete) onComplete(status);
      } else if (status.status === 'failed') {
        clearTimer();
        if (onError) onError(status);
      } else if (status.status === 'cancelled') {
        clearTimer();
        if (onCancel) onCancel(status);
      }
    } catch (err) {
      consecutiveErrors++;
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        clearTimer();
        if (onError) onError({ error: `连续 ${MAX_CONSECUTIVE_ERRORS} 次网络错误,轮询已停止` });
        return;
      }
      console.warn('[TaskPoll] 网络错误,继续轮询:', err.message);
    } finally {
      inFlight = false;
    }
  };

  timerRef.id = setInterval(poll, intervalMs);
  poll();
  return timerRef.id;
};

/**
 * 在组件挂载时检查是否有运行中的任务,如果有则自动进入轮询模式。
 *
 * @param {string} taskType - "preprocess" | "ingest" | "dedup"
 * @param {function} onUpdate
 * @param {function} onComplete
 * @param {function} onError
 * @param {function} onCancel - 任务被取消时调用 (可选)
 * @param {number} intervalMs
 * @returns {{ taskId: string|null, timer: number|null }} 可用于组件 state
 */
export const resumeOrStartPolling = async (taskType, onUpdate, onComplete, onError, onCancel, intervalMs = 2000) => {
  const latest = await getLatestTask(taskType);

  if (!latest || !latest.task_id) {
    return { taskId: null, timer: null };  // 没有任务
  }

  // 终态任务: 触发对应回调,但返回 taskId=null,
  // 让调用方不会把"停止"按钮指向一个已完成的任务
  // (后端会以 400 Task cannot be cancelled 拒绝)。
  if (latest.status === 'completed') {
    if (onComplete) onComplete(latest);
    return { taskId: null, timer: null };
  }

  if (latest.status === 'failed') {
    if (onError) onError(latest);
    return { taskId: null, timer: null };
  }

  if (latest.status === 'cancelled') {
    if (onCancel) onCancel(latest);
    return { taskId: null, timer: null };
  }

  // running / pending → 开始轮询
  if (onUpdate) onUpdate(latest);
  const timer = pollTaskUntilDone(latest.task_id, onUpdate, onComplete, onError, onCancel, intervalMs);
  return { taskId: latest.task_id, timer };
};