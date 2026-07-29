import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Button,
  message,
  Table,
  Progress,
  Space,
  Tag,
  Select,
  Switch,
  InputNumber,
  Card,
  Divider,
  Statistic,
  Row,
  Col,
  Badge,
  Spin,
  Popconfirm,
} from 'antd';
import {
  SettingOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  LoadingOutlined,
  DownOutlined,
  UpOutlined,
  CheckCircleOutlined,
  StopOutlined,
  RedoOutlined,
} from '@ant-design/icons';
import { startPreprocessTask, pollTaskUntilDone, resumeOrStartPolling, cancelTask } from '../utils/taskApi';
import { API_BASE_URL, apiHeaders } from '../utils/api';


const { Option } = Select;
const LEVEL_CONFIG = [
  { key: '0Public', name: '公开', color: 'green', sourceDir: 'DOC/0Public', outputDir: 'MD/0Public' },
  { key: '1Restricted', name: '受限', color: 'orange', sourceDir: 'DOC/1Restricted', outputDir: 'MD/1Restricted' },
  { key: '2Confidential', name: '机密', color: 'red', sourceDir: 'DOC/2Confidential', outputDir: 'MD/2Confidential' },
];

const PreprocessPage = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processProgress, setProcessProgress] = useState(0);
  const [activeTab, setActiveTab] = useState('overview');
  const [stats, setStats] = useState({
    '0Public': { total: 0, success: 0, failed: 0, skipped: 0 },
    '1Restricted': { total: 0, success: 0, failed: 0, skipped: 0 },
    '2Confidential': { total: 0, success: 0, failed: 0, skipped: 0 },
  });
  const [processingLevel, setProcessingLevel] = useState('');
  const [processingFile, setProcessingFile] = useState('');
  const [fileCounts, setFileCounts] = useState({
    '0Public': 0,
    '1Restricted': 0,
    '2Confidential': 0,
  });
  const [showSettings, setShowSettings] = useState(false);
  const [loadingCounts, setLoadingCounts] = useState({
    '0Public': false,
    '1Restricted': false,
    '2Confidential': false,
  });
  const [logs, setLogs] = useState([]);

  // 失败清单: [{level, file, reason}]
  const [failedFiles, setFailedFiles] = useState([]);
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryProgress, setRetryProgress] = useState(0);
  const [retryMessage, setRetryMessage] = useState('');

  // 扫描源目录获取实际文件数
  const scanDirectoryFileCount = async (levelKey) => {
    try {
      const response = await fetch(`${API_BASE_URL}/scan-source-dir?level=${levelKey}`, { headers: apiHeaders() });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return { count: data.file_count, files: data.files };
    } catch (error) {
      console.error('扫描目录失败:', error);
      return { count: 0, files: [] };
    }
  };

  const refreshFileCount = async (levelKey) => {
    setLoadingCounts(prev => ({ ...prev, [levelKey]: true }));
    try {
      const { count } = await scanDirectoryFileCount(levelKey);
      setFileCounts(prev => ({
        ...prev,
        [levelKey]: count,
      }));
      return count;
    } finally {
      setLoadingCounts(prev => ({ ...prev, [levelKey]: false }));
    }
  };

  const refreshAllFileCounts = async () => {
    // 并发刷新 3 个密级（之前 await 串行）
    await Promise.all(LEVEL_CONFIG.map(level => refreshFileCount(level.key)));
  };

  const pollTimerRef = useRef(null);
  const taskIdRef = useRef(null);
  // 同步 guard: 防止 Popconfirm 快速双击在 React 重渲染前触发两次启动
  const isStartingRef = useRef(false);
  // 组件挂载状态: 异步回调可在 unmount 后用 mountedRef.current === false 短路 setState
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // 挂载时: 扫描文件数 + 检查运行任务 + 加载上次失败记录 + 同步 Settings
  useEffect(() => {
    refreshAllFileCounts();
    checkRunningTask();
    loadPreviousFailedFiles();
    checkRunningRetryTask();
    loadPreprocessDefaults();
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      if (retryPollRef.current) clearInterval(retryPollRef.current);
    };
  }, []);

  // P1: 从 Settings 页保存的预处理默认值同步表单初值, 避免硬编码 500/50 与 Settings 脱节
  const loadPreprocessDefaults = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/settings/preprocess`, { headers: apiHeaders() });
      if (!res.ok) return;
      const cfg = await res.json();
      if (typeof cfg.chunkSize === 'number') setChunkSize(cfg.chunkSize);
      if (typeof cfg.chunkOverlap === 'number') setChunkOverlap(cfg.chunkOverlap);
      if (typeof cfg.extractTables === 'boolean') setEnableTableExtract(cfg.extractTables);
      if (typeof cfg.enableLLM === 'boolean') setEnableLLM(cfg.enableLLM);
      if (typeof cfg.ocrLang === 'string' && cfg.ocrLang) setOcrLang(cfg.ocrLang);
    } catch (err) {
      console.warn('[PreprocessPage] 加载 Settings 默认值失败:', err.message);
    }
  };

  // 从上一次预处理任务结果中加载失败文件清单
  const loadPreviousFailedFiles = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/tasks/latest/preprocess`, { headers: apiHeaders() });
      if (!res.ok) return;
      const task = await res.json();
      if (task && task.status === 'completed' && task.result) {
        const failed = extractFailedFiles(task.result);
        if (failed.length > 0) {
          setFailedFiles(failed);
        }
        // 同时恢复统计数据
        if (task.result.results) {
          setStats(prev => {
            const updated = { ...prev };
            for (const [level, data] of Object.entries(task.result.results)) {
              if (updated[level]) {
                updated[level] = {
                  total: data.processed || data.total || 0,
                  success: data.success || 0,
                  failed: data.failed || 0,
                  skipped: 0,
                };
              }
            }
            return updated;
          });
        }
      }
    } catch (err) {
      console.warn('[PreprocessPage] 加载上次失败记录失败:', err.message);
    }
  };

  // 检查是否有运行中的重试任务
  const checkRunningRetryTask = async () => {
    try {
      const { taskId, timer } = await resumeOrStartPolling(
        'preprocess_retry',
        (task) => {
          setIsRetrying(true);
          setRetryProgress(Math.round(task.progress * 100));
          setRetryMessage(task.progress_detail?.message || '正在重试...');
        },
        (task) => {
          setIsRetrying(false);
          setRetryProgress(100);
          retryTaskIdRef.current = null;
          const data = task.result;
          if (data) {
            if (data.success_list && data.success_list.length > 0) {
              const successNames = data.success_list.map(s => `${s.level}:${s.file}`);
              setFailedFiles(prev => prev.filter(f => !successNames.includes(`${f.level}:${f.file}`)));
              setStats(prev => {
                const updated = { ...prev };
                for (const s of data.success_list) {
                  if (updated[s.level]) {
                    updated[s.level] = {
                      ...updated[s.level],
                      success: updated[s.level].success + 1,
                      failed: updated[s.level].failed - 1,
                    };
                  }
                }
                return updated;
              });
              message.success(`重试成功: ${data.success_count} 个文件`);
            }
            if (data.still_failed_list && data.still_failed_list.length > 0) {
              setFailedFiles(prev => {
                const stillMap = {};
                data.still_failed_list.forEach(f => { stillMap[`${f.level}:${f.file}`] = f.reason; });
                return prev.map(f => {
                  const key = `${f.level}:${f.file}`;
                  if (stillMap[key]) {
                    return { ...f, reason: stillMap[key] };
                  }
                  return f;
                });
              });
              if (data.still_failed_count > 0) {
                message.warning(`${data.still_failed_count} 个文件仍然失败`);
              }
            }
          }
          refreshAllFileCounts();
          setTimeout(() => { setRetryProgress(0); setRetryMessage(''); }, 3000);
        },
        (task) => {
          message.error(`重试失败: ${task.error || '未知错误'}`);
          setIsRetrying(false);
          setRetryProgress(0);
          setRetryMessage('');
          retryTaskIdRef.current = null;
        },
        (task) => {
          setIsRetrying(false);
          setRetryProgress(0);
          setRetryMessage('');
          retryTaskIdRef.current = null;
          message.warning('重试已停止');
        },
      );
      if (taskId) {
        retryTaskIdRef.current = taskId;
        retryPollRef.current = timer;
        setIsRetrying(true);
      }
    } catch (err) {
      console.warn('[PreprocessPage] 检查重试任务失败:', err.message);
    }
  };

  // 从完成结果中提取失败文件清单
  const extractFailedFiles = (taskResult) => {
    const failed = [];
    if (taskResult && taskResult.results) {
      for (const [level, data] of Object.entries(taskResult.results)) {
        if (data.logs) {
          for (const log of data.logs) {
            if (log.status === 'error') {
              failed.push({
                level,
                file: log.file,
                reason: log.message,
              });
            }
          }
        }
      }
    }
    return failed;
  };

  const checkRunningTask = async () => {
    try {
      const { taskId, timer } = await resumeOrStartPolling(
        'preprocess',
        (task) => {
          setIsProcessing(true);
          setProcessProgress(Math.round(task.progress * 100));
          const detail = task.progress_detail || {};
          setProcessingLevel(detail.level || '');
          setProcessingFile(detail.file || '');
        },
        (task) => {
          setProcessProgress(100);
          setIsProcessing(false);
          taskIdRef.current = null;
          // 从任务结果中更新各密级统计
          if (task.result && task.result.results) {
            setStats(prev => {
              const updated = { ...prev };
              for (const [level, data] of Object.entries(task.result.results)) {
                if (updated[level]) {
                  updated[level] = {
                    total: data.processed || data.total || 0,
                    success: data.success || 0,
                    failed: data.failed || 0,
                    skipped: 0,
                  };
                }
              }
              return updated;
            });
          }
          // 提取失败文件清单 (去重: 用 level:file 作 key)
          const failed = extractFailedFiles(task.result);
          setFailedFiles(prev => {
            const seen = new Set(prev.map(f => `${f.level}:${f.file}`));
            const merged = [...prev];
            for (const f of failed) {
              const key = `${f.level}:${f.file}`;
              if (!seen.has(key)) {
                seen.add(key);
                merged.push(f);
              }
            }
            return merged;
          });
          message.success('预处理完成');
          refreshAllFileCounts();
          setTimeout(() => setProcessProgress(0), 3000);
        },
        (task) => {
          message.error(`预处理失败: ${task.error || '未知错误'}`);
          setIsProcessing(false);
          setProcessProgress(0);
          taskIdRef.current = null;
        },
        (task) => {
          // onCancel
          setIsProcessing(false);
          setProcessProgress(0);
          taskIdRef.current = null;
          message.warning('预处理已停止');
        },
      );
      // resumeOrStartPolling 只在 running/pending 时返回非空 taskId,
      // 终态任务由对应回调处理且返回 null, 不会误把"停止"按钮指向已完成任务。
      if (taskId) {
        taskIdRef.current = taskId;
        pollTimerRef.current = timer;
        setIsProcessing(true);
      }
    } catch (err) {
      console.warn('[PreprocessPage] 检查运行任务失败:', err.message);
    }
  };

  // 处理参数 - 默认使用父子块模式
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [enableChunk, setEnableChunk] = useState(true);
  const [chunkStrategy, setChunkStrategy] = useState('parent-child');
  const [enableTableExtract, setEnableTableExtract] = useState(true);
  const [encoding, setEncoding] = useState('utf-8');
  const [enableLLM, setEnableLLM] = useState(true);  // LLM摘要生成开关，默认开启
  const [ocrLang, setOcrLang] = useState('chi_sim+eng');  // OCR 语言, 由 Settings 同步

  // 调用 X2MD 处理单个密级
  const processLevel = async (level) => {
    try {
      // 先扫描获取文件列表
      const { count, files } = await scanDirectoryFileCount(level.key);

      if (count === 0) {
        setStats(prev => ({
          ...prev,
          [level.key]: { total: 0, success: 0, failed: 0, skipped: 0 }
        }));
        return { total: 0, success: 0, failed: 0 };
      }

      // 调用后端API进行X2MD处理
      const response = await fetch(`${API_BASE_URL}/preprocess`, {
        method: 'POST',
        headers: apiHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          level: level.key,
          chunk_strategy: chunkStrategy,
          enable_llm: enableLLM,
          // P2-5: 之前 UI 的 encoding/enableTableExtract 控件没传给后端, 形同摆设
          encoding: encoding,
          extract_tables: enableTableExtract,
          chunk_size: enableChunk ? chunkSize : null,
          chunk_overlap: enableChunk ? chunkOverlap : null,
          ocr_lang: ocrLang || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // 更新日志 — 批量写入,避免逐条 setLogs 触发 N 次 re-render
      if (result.logs) {
        const newLogEntries = [];
        const newFailedEntries = [];
        for (const log of result.logs) {
          newLogEntries.push({
            time: new Date().toLocaleTimeString(),
            level: level.key,
            file: log.file,
            status: log.status,
            message: log.message,
          });
          if (log.status === 'error') {
            newFailedEntries.push({
              level: level.key,
              file: log.file,
              reason: log.message,
            });
          }
        }
        if (newLogEntries.length > 0) {
          setLogs(prev => [...prev, ...newLogEntries]);
        }
        if (newFailedEntries.length > 0) {
          // 与现有失败列表去重(level:file 复用键)
          setFailedFiles(prev => {
            const seen = new Set(prev.map(f => `${f.level}:${f.file}`));
            const merged = [...prev];
            for (const f of newFailedEntries) {
              const k = `${f.level}:${f.file}`;
              if (!seen.has(k)) {
                seen.add(k);
                merged.push(f);
              }
            }
            return merged;
          });
        }
      }

      // 更新统计
      setStats(prev => ({
        ...prev,
        [level.key]: {
          total: result.processed || count,
          success: result.success || 0,
          failed: result.failed || 0,
          skipped: 0,
        }
      }));

      return result;

    } catch (error) {
      console.error(`处理 ${level.key} 失败:`, error);
      message.error(`${level.name}: 处理失败 - ${error.message}`);
      return { total: 0, success: 0, failed: 0 };
    }
  };

  const handleProcessAll = async () => {
    // 同步 guard: 双击 Popconfirm 时 React 还没渲染 disabled,这里直接拦
    if (isStartingRef.current) return;
    isStartingRef.current = true;
    setIsProcessing(true);
    setProcessProgress(0);
    setLogs([]);

    try {
      // 启动后台预处理任务(全部密级)
      const result = await startPreprocessTask({
        level: null,  // null = 全部密级
        chunk_strategy: chunkStrategy,
        enable_llm: enableLLM,
        encoding: encoding,
        extract_tables: enableTableExtract,
        chunk_size: enableChunk ? chunkSize : null,
        chunk_overlap: enableChunk ? chunkOverlap : null,
        ocr_lang: ocrLang || null,
      });

      if (result.message && result.task_id) {
        // 已有运行中的任务,自动切换到轮询
        message.info(result.message);
      }

      taskIdRef.current = result.task_id;

      // 开始轮询进度
      pollTimerRef.current = pollTaskUntilDone(
        result.task_id,
        (task) => {
          setProcessProgress(Math.round(task.progress * 100));
          const detail = task.progress_detail || {};
          setProcessingLevel(detail.level || '');
          setProcessingFile(detail.file || '');
          if (detail.message) {
            setProcessingFile(detail.message);
          }
        },
        (task) => {
          setProcessProgress(100);
          setIsProcessing(false);
          taskIdRef.current = null;
          // 从任务结果中更新各密级统计
          if (task.result && task.result.results) {
            setStats(prev => {
              const updated = { ...prev };
              for (const [level, data] of Object.entries(task.result.results)) {
                if (updated[level]) {
                  updated[level] = {
                    total: data.processed || data.total || 0,
                    success: data.success || 0,
                    failed: data.failed || 0,
                    skipped: 0,
                  };
                }
              }
              return updated;
            });
          }
          // 提取失败文件清单 (去重: 用 level:file 作 key)
          const failed = extractFailedFiles(task.result);
          setFailedFiles(prev => {
            const seen = new Set(prev.map(f => `${f.level}:${f.file}`));
            const merged = [...prev];
            for (const f of failed) {
              const key = `${f.level}:${f.file}`;
              if (!seen.has(key)) {
                seen.add(key);
                merged.push(f);
              }
            }
            return merged;
          });
          message.success('所有密级预处理完成');
          refreshAllFileCounts();
          setTimeout(() => setProcessProgress(0), 3000);
        },
        (task) => {
          message.error(`预处理失败: ${task.error || '未知错误'}`);
          setIsProcessing(false);
          setProcessProgress(0);
          taskIdRef.current = null;
        },
        (task) => {
          // onCancel
          setIsProcessing(false);
          setProcessProgress(0);
          taskIdRef.current = null;
          message.warning('预处理已停止');
        },
      );
    } catch (error) {
      message.error('启动预处理失败: ' + error.message);
      setIsProcessing(false);
      setProcessProgress(0);
    } finally {
      isStartingRef.current = false;
    }
  };

  // 停止预处理任务
  const handleStop = async () => {
    if (!taskIdRef.current) return;
    try {
      await cancelTask(taskIdRef.current);
      message.warning('预处理已停止');
    } catch (error) {
      message.error('停止任务失败: ' + error.message);
    } finally {
      // 无论 cancelTask 成败都清理 UI 状态, 否则失败时 UI 永久卡在"处理中"
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      setIsProcessing(false);
      setProcessProgress(0);
      taskIdRef.current = null;
    }
  };

  // 失败重试 (异步任务模式)
  const retryTaskIdRef = useRef(null);
  const retryPollRef = useRef(null);

  const handleRetryFailed = async () => {
    if (failedFiles.length === 0) {
      message.info('没有需要重试的文件');
      return;
    }
    // 同步 guard 防双击
    if (isStartingRef.current) return;
    isStartingRef.current = true;
    setIsRetrying(true);
    setRetryProgress(0);
    try {
      const res = await axios.post(`${API_BASE_URL}/preprocess/retry`, {
        files: failedFiles.map(f => ({ level: f.level, file: f.file })),
        chunk_strategy: chunkStrategy,
        enable_llm: enableLLM,
        encoding: encoding,
        extract_tables: enableTableExtract,
        chunk_size: enableChunk ? chunkSize : null,
        chunk_overlap: enableChunk ? chunkOverlap : null,
        ocr_lang: ocrLang || null,
      });

      // 如果已有运行中的重试任务
      if (res.data.message && res.data.task_id) {
        message.info(res.data.message);
      }

      retryTaskIdRef.current = res.data.task_id;

      retryPollRef.current = pollTaskUntilDone(
        res.data.task_id,
        (task) => {
          setRetryProgress(Math.round(task.progress * 100));
          const detail = task.progress_detail || {};
          setRetryMessage(detail.message || '正在重试...');
        },
        (task) => {
          setIsRetrying(false);
          setRetryProgress(100);
          retryTaskIdRef.current = null;
          const data = task.result;
          if (data) {
            // 成功重试的从失败清单中剔除
            if (data.success_list && data.success_list.length > 0) {
              const successNames = data.success_list.map(s => `${s.level}:${s.file}`);
              setFailedFiles(prev => prev.filter(f => !successNames.includes(`${f.level}:${f.file}`)));
              setStats(prev => {
                const updated = { ...prev };
                for (const s of data.success_list) {
                  if (updated[s.level]) {
                    updated[s.level] = {
                      ...updated[s.level],
                      success: updated[s.level].success + 1,
                      failed: updated[s.level].failed - 1,
                    };
                  }
                }
                return updated;
              });
              message.success(`重试成功: ${data.success_count} 个文件`);
            }
            // 仍失败的更新失败原因
            if (data.still_failed_list && data.still_failed_list.length > 0) {
              setFailedFiles(prev => {
                const stillMap = {};
                data.still_failed_list.forEach(f => { stillMap[`${f.level}:${f.file}`] = f.reason; });
                return prev.map(f => {
                  const key = `${f.level}:${f.file}`;
                  if (stillMap[key]) {
                    return { ...f, reason: stillMap[key] };
                  }
                  return f;
                });
              });
              if (data.still_failed_count > 0) {
                message.warning(`${data.still_failed_count} 个文件仍然失败`);
              }
            }
          }
          refreshAllFileCounts();
          setTimeout(() => { setRetryProgress(0); setRetryMessage(''); }, 3000);
        },
        (task) => {
          message.error(`重试失败: ${task.error || '未知错误'}`);
          setIsRetrying(false);
          setRetryProgress(0);
          setRetryMessage('');
          retryTaskIdRef.current = null;
        },
        (task) => {
          // onCancel
          setIsRetrying(false);
          setRetryProgress(0);
          setRetryMessage('');
          retryTaskIdRef.current = null;
          message.warning('重试已停止');
        },
      );
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || '未知错误';
      message.error(`启动重试失败: ${detail}`);
      setIsRetrying(false);
      setRetryProgress(0);
    } finally {
      isStartingRef.current = false;
    }
  };

  // 停止重试任务
  const handleStopRetry = async () => {
    if (!retryTaskIdRef.current) return;
    try {
      await cancelTask(retryTaskIdRef.current);
      message.warning('重试已停止');
    } catch (error) {
      message.error('停止重试失败: ' + error.message);
    } finally {
      // 无论 cancelTask 成败都清理 UI 状态, 否则失败时 UI 永久卡在"重试中"
      if (retryPollRef.current) {
        clearInterval(retryPollRef.current);
        retryPollRef.current = null;
      }
      setIsRetrying(false);
      setRetryProgress(0);
      setRetryMessage('');
      retryTaskIdRef.current = null;
    }
  };

  const getLevelTag = (levelKey) => {
    const level = LEVEL_CONFIG.find(l => l.key === levelKey);
    return level ? <Tag color={level.color}>{level.name}</Tag> : <Tag>{levelKey}</Tag>;
  };

  const logColumns = [
    { title: '时间', dataIndex: 'time', key: 'time', width: 100 },
    { title: '密级', dataIndex: 'level', key: 'level', width: 80, render: (l) => getLevelTag(l) },
    { title: '文件', dataIndex: 'file', key: 'file', ellipsis: true },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status) => (
        <Badge
          status={status === 'success' ? 'success' : status === 'error' ? 'error' : 'default'}
          text={status === 'success' ? '成功' : status === 'error' ? '失败' : '跳过'}
        />
      )
    },
    { title: '消息', dataIndex: 'message', key: 'message' },
  ];

  const failedColumns = [
    { title: '密级', dataIndex: 'level', key: 'level', width: 80, render: (l) => getLevelTag(l) },
    { title: '文件', dataIndex: 'file', key: 'file', ellipsis: true },
    { title: '失败原因', dataIndex: 'reason', key: 'reason', ellipsis: true },
  ];

  const totalStats = {
    total: Object.values(stats).reduce((sum, s) => sum + s.total, 0),
    success: Object.values(stats).reduce((sum, s) => sum + s.success, 0),
    failed: Object.values(stats).reduce((sum, s) => sum + s.failed, 0),
    skipped: Object.values(stats).reduce((sum, s) => sum + s.skipped, 0),
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">文件预处理</h1>
        <p className="page-subtitle">使用 X2MD 将 DOC 目录下各密级文档批量转换为 Markdown 格式</p>
      </div>

      {/* 全局统计 */}
      <Row gutter={[24, 24]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic title="总文件数" value={totalStats.total} valueStyle={{ color: '#2E2E2E' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="成功转换" value={totalStats.success} valueStyle={{ color: '#55A722' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="处理失败" value={totalStats.failed} valueStyle={{ color: '#EC4F4F' }} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="已跳过" value={totalStats.skipped} valueStyle={{ color: '#7E7E7E' }} />
          </Card>
        </Col>
      </Row>

      {/* 三个密级卡片并排显示 */}
      <Row gutter={[24, 24]} style={{ marginBottom: '24px' }}>
        {LEVEL_CONFIG.map(level => (
          <Col span={8} key={level.key}>
            <Card
              title={
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Space>
                    <Tag color={level.color}>{level.name}</Tag>
                  </Space>
                  <Button
                    type="link"
                    size="small"
                    icon={<ReloadOutlined spin={loadingCounts[level.key]} />}
                    onClick={async () => {
                      const count = await refreshFileCount(level.key);
                      message.success(`${level.name}: 扫描到 ${count} 个文件`);
                    }}
                    loading={loadingCounts[level.key]}
                  >
                    {loadingCounts[level.key] ? '扫描中...' : '刷新'}
                  </Button>
                </div>
              }
            >
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '12px', color: '#7E7E7E', marginBottom: '4px' }}>源目录</div>
                <div style={{ fontFamily: 'monospace', fontSize: '12px', wordBreak: 'break-all' }}>
                  {level.sourceDir}
                </div>
              </div>
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '12px', color: '#7E7E7E', marginBottom: '4px' }}>输出目录</div>
                <div style={{ fontFamily: 'monospace', fontSize: '12px', wordBreak: 'break-all' }}>
                  {level.outputDir}
                </div>
              </div>
              <Divider style={{ margin: '12px 0' }} />
              <div style={{ marginBottom: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: 500, color: '#2E2E2E' }}>{fileCounts[level.key]}</div>
                <div style={{ fontSize: '12px', color: '#7E7E7E' }}>源目录文件数</div>
              </div>
              <Row gutter={16}>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '18px', fontWeight: 500, color: '#55A722' }}>{stats[level.key].success}</div>
                    <div style={{ fontSize: '12px', color: '#7E7E7E' }}>成功</div>
                  </div>
                </Col>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '18px', fontWeight: 500, color: '#EC4F4F' }}>{stats[level.key].failed}</div>
                    <div style={{ fontSize: '12px', color: '#7E7E7E' }}>失败</div>
                  </div>
                </Col>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '18px', fontWeight: 500, color: '#7E7E7E' }}>{stats[level.key].skipped}</div>
                    <div style={{ fontSize: '12px', color: '#7E7E7E' }}>跳过</div>
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        ))}
      </Row>

      {/* 失败文件清单 */}
      {failedFiles.length > 0 && (
        <Card style={{ marginBottom: '24px' }}>
          <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Space>
              <Tag color="red">失败 {failedFiles.length}</Tag>
              失败文件清单
            </Space>
            <Space>
              {!isRetrying && (
                <Popconfirm
                  title={`确认重试 ${failedFiles.length} 个失败文件？`}
                  onConfirm={handleRetryFailed}
                  okText="确认重试"
                  cancelText="取消"
                >
                  <Button
                    danger
                    icon={<RedoOutlined />}
                    disabled={isProcessing}
                  >
                    失败重试
                  </Button>
                </Popconfirm>
              )}
              {isRetrying && (
                <Button
                  danger
                  icon={<StopOutlined />}
                  onClick={handleStopRetry}
                >
                  停止
                </Button>
              )}
            </Space>
          </div>
          {isRetrying && (
            <div style={{ marginBottom: '16px' }}>
              <Progress percent={retryProgress} status="active" />
              <div style={{ textAlign: 'center', color: '#7E7E7E', marginTop: '8px' }}>
                <Spin size="small" style={{ marginRight: '8px' }} />
                {retryMessage || '正在重试...'}
              </div>
            </div>
          )}
          <Table
            columns={failedColumns}
            dataSource={failedFiles.map((f, i) => ({ ...f, key: i }))}
            pagination={false}
            size="small"
          />
        </Card>
      )}

      {/* 转换参数配置 */}
      <Card style={{ marginBottom: '24px' }}>
        <div
          style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}
          onClick={() => setShowSettings(!showSettings)}
        >
          <SettingOutlined style={{ marginRight: '8px' }} />
          <span style={{ fontWeight: 500, flex: 1 }}>转换参数配置</span>
          {showSettings ? <UpOutlined /> : <DownOutlined />}
        </div>

        {showSettings && (
          <div style={{ marginTop: '24px' }}>
            <Row gutter={[24, 24]}>
              <Col span={8}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                  编码格式
                </label>
                <Select value={encoding} onChange={setEncoding} style={{ width: '100%' }}>
                  <Option value="utf-8">UTF-8</Option>
                  <Option value="gbk">GBK</Option>
                  <Option value="gb2312">GB2312</Option>
                  <Option value="auto">自动检测</Option>
                </Select>
              </Col>
              <Col span={8}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                  切片策略
                </label>
                <Select value={chunkStrategy} onChange={setChunkStrategy} style={{ width: '100%' }}>
                  <Option value="chunk">普通切片 (.chunks.json)</Option>
                  <Option value="parent-child">父子块切片 (.parents/.children.json)</Option>
                </Select>
              </Col>
              <Col span={8}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                  提取表格
                </label>
                <Switch checked={enableTableExtract} onChange={setEnableTableExtract} />
                <span style={{ marginLeft: '8px', color: '#7E7E7E', fontSize: '12px' }}>
                  PDF表格识别
                </span>
              </Col>
              <Col span={8}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                  LLM摘要生成
                </label>
                <Switch checked={enableLLM} onChange={setEnableLLM} />
                <span style={{ marginLeft: '8px', color: '#7E7E7E', fontSize: '12px' }}>
                  {enableLLM ? '开启（处理较慢）' : '关闭（默认）'}
                </span>
              </Col>
            </Row>

            {enableChunk && (
              <Row gutter={[24, 24]} style={{ marginTop: '16px' }}>
                <Col span={8}>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                    切片大小 (字符)
                  </label>
                  <InputNumber
                    value={chunkSize}
                    onChange={(v) => v != null && setChunkSize(v)}
                    min={100}
                    max={5000}
                    style={{ width: '100%' }}
                  />
                </Col>
                <Col span={8}>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                    切片重叠 (字符)
                  </label>
                  <InputNumber
                    value={chunkOverlap}
                    onChange={(v) => v != null && setChunkOverlap(v)}
                    min={0}
                    max={500}
                    style={{ width: '100%' }}
                  />
                </Col>
                <Col span={8}>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
                    启用切片
                  </label>
                  <Switch checked={enableChunk} onChange={setEnableChunk} />
                </Col>
              </Row>
            )}
          </div>
        )}
      </Card>

      {/* 处理进度 */}
      {isProcessing && (
        <div className="card" style={{ marginBottom: '24px' }}>
          <div className="card-title">
            <Space>
              <LoadingOutlined spin />
              处理进度
              {processingLevel && getLevelTag(processingLevel)}
            </Space>
          </div>
          <Progress percent={processProgress} status="active" />
          <div style={{ marginTop: '8px', textAlign: 'center', color: '#7E7E7E' }}>
            {processingFile ? `正在处理: ${processingFile}` : '准备处理...'}
          </div>
        </div>
      )}

      {/* 处理日志 */}
      {logs.length > 0 && (
        <div className="card">
          <div className="card-title">处理日志 ({logs.length})</div>
          <Table
            columns={logColumns}
            dataSource={logs}
            rowKey={(record, index) => `${record.time}-${index}`}
            pagination={{ pageSize: 50, showSizeChanger: true, pageSizeOptions: [20, 50, 100, 200] }}
            size="small"
            scroll={{ y: 400 }}
            virtual
          />
        </div>
      )}

      {/* 操作按钮 */}
      <div style={{ textAlign: 'center', marginTop: '24px' }}>
        <Space>
          <Button
            type="primary"
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={handleProcessAll}
            loading={isProcessing}
            disabled={isProcessing}
          >
            开始预处理全部密级
          </Button>
          {isProcessing && (
            <Button
              danger
              size="large"
              icon={<StopOutlined />}
              onClick={handleStop}
            >
              停止
            </Button>
          )}
          <Button
            size="large"
            icon={<ReloadOutlined />}
            onClick={() => {
              setLogs([]);
              setStats({
                '0Public': { total: 0, success: 0, failed: 0, skipped: 0 },
                '1Restricted': { total: 0, success: 0, failed: 0, skipped: 0 },
                '2Confidential': { total: 0, success: 0, failed: 0, skipped: 0 },
              });
            }}
            disabled={isProcessing}
          >
            清空日志
          </Button>
        </Space>
      </div>
    </div>
  );
};

export default PreprocessPage;
