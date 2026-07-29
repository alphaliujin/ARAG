import axios from 'axios';

// 全局 API 基础 URL
// 生产环境推荐使用相对路径 (例如 '/api/v1') + 反向代理,避免硬编码 host;
// 开发期回落到 localhost:8000 仅是为了 npm start 时方便,
// 部署到任何非 localhost 环境必须设置 REACT_APP_API_BASE_URL。
export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1';

// 可选的 API Key - 后端 AUTH_DISABLED=False 时必填。
// 通过 REACT_APP_API_KEY 注入(写到 UI/.env.local 或构建环境)。
const API_KEY = process.env.REACT_APP_API_KEY || '';

// ==================== 共享 Axios 实例 ====================
// 所有 backend 请求统一走这个实例:
//   - 自动注入 X-API-Key 头(后端 AUTH_DISABLED=False 时必须)
//   - 默认 baseURL,调用方写 '/scan/file' 而非完整 URL
//   - 默认 30 秒 timeout(单次请求可覆盖)
//
// 注意: fetch() 调用(taskApi.js、PreprocessPage.js 中)也需要 apiHeaders()。
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

apiClient.interceptors.request.use((config) => {
  if (API_KEY) {
    config.headers = config.headers || {};
    config.headers['X-API-Key'] = API_KEY;
  }
  return config;
});

/** 给 fetch() 调用使用的请求头工厂 - 与 apiClient 拦截器保持一致. */
export const apiHeaders = (extra = {}) => {
  const headers = { ...extra };
  if (API_KEY) headers['X-API-Key'] = API_KEY;
  return headers;
};

// ==================== 文件上传 (只保存, 不自动处理) ====================
// 所有请求一律走 apiClient (带 X-API-Key 拦截器);
// 此前用裸 axios 调用, 启用认证后会全部 401, ScanPage/DatabasePage/DocScan 全废。

export const scanDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  // 不手动设 Content-Type: axios/浏览器会为 FormData 自动加上带 boundary 的
  // multipart/form-data 头; 手动设成无 boundary 的 'multipart/form-data' 会导致
  // 后端 multipart 解析失败。
  const response = await apiClient.post('/scan/file', formData, {
    timeout: 60000,  // 1 分钟 (只保存文件, 不做处理)
  });

  return response.data;
};

// ==================== 文本扫描 ====================

export const scanText = async (text) => {
  const response = await apiClient.post('/scan/text', { text }, {
    timeout: 60000,
  });

  return response.data;
};

// ==================== DocScan 操作 API ====================

// 预处理 (X2MD 转换)
export const docscanPreprocess = async (filename, options = {}) => {
  const payload = {
    filename,
    chunk_strategy: options.chunkStrategy || 'parent-child',
    enable_llm: options.enableLlm || false,
    encoding: options.encoding || null,
    extract_tables: options.extractTables !== undefined ? options.extractTables : true,
    chunk_size: options.chunkSize || null,
    chunk_overlap: options.chunkOverlap || null,
    ocr_lang: options.ocrLang || null,
    enable_ocr: options.enableOcr !== undefined ? options.enableOcr : null,
  };

  const response = await apiClient.post('/docscan/preprocess', payload, {
    timeout: 300000,  // 5 分钟
  });

  return response.data;
};

// 生成向量 (后台任务: 立即返回 task_id, 前端轮询 /tasks/{id} 看进度)
export const docscanEmbed = async (filename) => {
  const response = await apiClient.post('/docscan/embed', null, {
    params: { filename },
  });

  return response.data;
};

// 比对 (层级比对: 摘要->父块->子块)
export const docscanCompare = async (filename, nResults = 10) => {
  const response = await apiClient.post('/docscan/compare', null, {
    params: { filename, n_results: nResults },
    timeout: 600000,  // 10 分钟
  });

  return response.data;
};

// 文件状态查询 (saved/preprocessed/embedded/compared)
export const docscanStatus = async (filename) => {
  const response = await apiClient.get('/docscan/status', {
    params: { filename },
  });

  return response.data;
};

// ==================== DocScan 文件管理 API ====================

export const listDocScanFiles = async () => {
  const response = await apiClient.get('/docscan/files');
  return response.data;
};

export const deleteDocScanFile = async (name) => {
  const response = await apiClient.delete('/docscan/files', {
    params: { name },
  });
  return response.data;
};

// 清空 DocScan 目录全部内容(源文件 + 切片 + 向量 + 图片)
export const clearAllDocScanFiles = async () => {
  const response = await apiClient.delete('/docscan/files/all');
  return response.data;
};

// DocScan 统计 API
export const getDocScanStats = async () => {
  const response = await apiClient.get('/docscan/stats');
  return response.data;
};

// ==================== 数据库操作 (DatabasePage 使用) ====================

export const ingestDocuments = async (classificationLevel = null) => {
  const data = classificationLevel ? { classification_level: classificationLevel } : {};
  const response = await apiClient.post('/ingest', data);
  return response.data;
};

export const getDatabaseStats = async () => {
  const response = await apiClient.get('/stats');
  return response.data;
};

export const resetDatabase = async () => {
  const response = await apiClient.delete('/reset');
  return response.data;
};

// ==================== 格式化工具 ====================

export const formatConfidence = (confidence) => {
  if (confidence === null || confidence === undefined || Number.isNaN(confidence)) {
    return '—';
  }
  return `${(confidence * 100).toFixed(1)}%`;
};

export const getLevelColor = (level) => {
  switch (level) {
    case 'restricted':
      return '#FA8C16';
    case 'confidential':
      return '#EC4F4F';
    default:
      return '#535353';
  }
};

export const getLevelLabel = (level) => {
  switch (level) {
    case 'public':
      return '公开';
    case 'restricted':
      return '受限';
    case 'confidential':
      return '机密';
    default:
      return '未知';
  }
};
