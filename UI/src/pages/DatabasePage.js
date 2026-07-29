import React, { useState, useEffect, useRef } from 'react';
import { Button, message, Modal, Table, Space, Progress } from 'antd';
import { DatabaseOutlined, ReloadOutlined, DeleteOutlined, SyncOutlined, StopOutlined } from '@ant-design/icons';
import { getDatabaseStats, resetDatabase } from '../utils/api';
import { startIngestTask, pollTaskUntilDone, resumeOrStartPolling, cancelTask } from '../utils/taskApi';

// 密级元数据（统一 public/restricted/confidential 三级制）
const LEVEL_META = {
  public: { label: '公开', color: '#55A722' },
  restricted: { label: '受限', color: '#FA8C16' },
  confidential: { label: '机密', color: '#EC4F4F' },
};

const getLevelName = (level) => LEVEL_META[level]?.label || level;

const DatabasePage = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [importingLevel, setImportingLevel] = useState(null);
  const [resetting, setResetting] = useState(false);
  // 异步入库状态 (与 IngestPage 一致, 避免同步 /ingest 长任务 HTTP 超时)
  const [ingesting, setIngesting] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');

  const pollTimerRef = useRef(null);
  const taskIdRef = useRef(null);

  const loadStats = async () => {
    setLoading(true);
    try {
      const data = await getDatabaseStats();
      setStats(data);
    } catch (error) {
      const detail = error.response?.data?.detail || error.message || '未知错误';
      message.error('获取数据库状态失败: ' + detail);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
    // 挂载时恢复运行中的入库任务, 切页/刷新不丢进度
    checkRunningTask();
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const checkRunningTask = async () => {
    try {
      const { taskId, timer } = await resumeOrStartPolling(
        'ingest',
        (task) => {
          setIngesting(true);
          setImportingLevel(task.progress_detail?.level || 'all');
          setIngestProgress(Math.round(task.progress * 100));
          setProgressMessage(task.progress_detail?.message || '入库进行中...');
        },
        (task) => {
          setIngestProgress(100);
          setProgressMessage('入库完成');
          message.success('文档导入完成');
          setIngesting(false);
          setImportingLevel(null);
          taskIdRef.current = null;
          loadStats();
          setTimeout(() => { setIngestProgress(0); setProgressMessage(''); }, 3000);
        },
        (task) => {
          message.error(`入库失败: ${task.error || '未知错误'}`);
          setIngesting(false);
          setImportingLevel(null);
          setIngestProgress(0);
          setProgressMessage('');
          taskIdRef.current = null;
        },
        () => {
          setIngesting(false);
          setImportingLevel(null);
          setIngestProgress(0);
          setProgressMessage('');
          taskIdRef.current = null;
          message.warning('入库已停止');
        },
      );
      if (taskId) {
        taskIdRef.current = taskId;
        pollTimerRef.current = timer;
        setIngesting(true);
      }
    } catch (err) {
      console.warn('[DatabasePage] 检查运行任务失败:', err.message);
    }
  };

  // 通用异步入库: 启动后台任务 + 轮询 (与 IngestPage 行为一致, 取代同步 /ingest)
  const runIngest = async (classification) => {
    setIngesting(true);
    setImportingLevel(classification || 'all');
    setIngestProgress(0);
    setProgressMessage('启动入库任务...');
    try {
      const result = await startIngestTask({
        classification: classification,  // null = 全部密级
        strategy: 'auto',
        include_images: true,
        force: false,
        // 不传 embedding_model: 沿用当前 settings 的模型, 不触发模型切换/清库
      });
      // 后端返回"已有运行任务"时, 切到轮询模式而非直接报错
      if (result.message && result.message.includes('已在运行')) {
        message.warning(result.message);
        setIngesting(false);
        setImportingLevel(null);
        setProgressMessage('');
        return;
      }
      taskIdRef.current = result.task_id;
      setProgressMessage(result.message || '入库任务已启动');
      pollTimerRef.current = pollTaskUntilDone(
        result.task_id,
        (task) => {
          setImportingLevel(task.progress_detail?.level || classification || 'all');
          setIngestProgress(Math.round(task.progress * 100));
          setProgressMessage(task.progress_detail?.message || '入库进行中...');
        },
        () => {
          setIngestProgress(100);
          setProgressMessage('入库完成');
          message.success('文档导入完成');
          setIngesting(false);
          setImportingLevel(null);
          taskIdRef.current = null;
          loadStats();
          setTimeout(() => { setIngestProgress(0); setProgressMessage(''); }, 3000);
        },
        (task) => {
          message.error(`入库失败: ${task.error || '未知错误'}`);
          setIngesting(false);
          setImportingLevel(null);
          setIngestProgress(0);
          setProgressMessage('');
          taskIdRef.current = null;
        },
        () => {
          setIngesting(false);
          setImportingLevel(null);
          setIngestProgress(0);
          setProgressMessage('');
          taskIdRef.current = null;
          message.warning('入库已停止');
        },
      );
    } catch (error) {
      message.error('启动入库失败: ' + (error.message || '未知错误'));
      setIngesting(false);
      setImportingLevel(null);
      setIngestProgress(0);
      setProgressMessage('');
    }
  };

  const handleImportAll = () => runIngest(null);
  const handleImportLevel = (level) => runIngest(level);

  const handleStop = async () => {
    if (!taskIdRef.current) return;
    try {
      await cancelTask(taskIdRef.current);
      message.warning('入库已停止');
    } catch (error) {
      message.error('停止任务失败: ' + error.message);
    } finally {
      // 无论 cancelTask 成败都清理 UI 状态, 否则失败时永久卡在"处理中"
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      setIngesting(false);
      setImportingLevel(null);
      setIngestProgress(0);
      setProgressMessage('');
      taskIdRef.current = null;
    }
  };

  const handleReset = () => {
    Modal.confirm({
      title: '确认重置',
      content: '确定要清空所有向量数据吗？此操作不可恢复。',
      okText: '确认重置',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        setResetting(true);
        try {
          await resetDatabase();
          message.success('数据库已重置');
          await loadStats();
        } catch (error) {
          const detail = error.response?.data?.detail || error.message || '未知错误';
          message.error('重置失败: ' + detail);
        } finally {
          setResetting(false);
        }
      },
    });
  };

  const columns = [
    {
      title: '文档类型',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          <span style={{ fontSize: '16px' }}>{record.icon}</span>
          <span>{text}</span>
        </Space>
      ),
    },
    {
      title: '向量数量',
      dataIndex: 'count',
      key: 'count',
      render: (count, record) => (
        <span style={{ fontWeight: 500, color: LEVEL_META[record.key]?.color || '#55A722' }}>
          {count}
        </span>
      ),
    },
    {
      title: '说明',
      dataIndex: 'description',
      key: 'description',
      render: (text) => <span style={{ color: '#7E7E7E' }}>{text}</span>,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Button
          type="link"
          onClick={() => handleImportLevel(record.key)}
          loading={importingLevel === record.key}
          disabled={(importingLevel !== null && importingLevel !== record.key) || ingesting}
        >
          导入/更新
        </Button>
      ),
    },
  ];

  const dataSource = stats
    ? [
        {
          key: 'public',
          name: '公开文档',
          icon: '📄',
          count: stats.collections?.public_documents || 0,
          description: '用于建立基准向量的公开文档',
        },
        {
          key: 'restricted',
          name: '受限文档',
          icon: '⚠️',
          count: stats.collections?.restricted_documents || 0,
          description: '包含受限信息的文档库',
        },
        {
          key: 'confidential',
          name: '机密文档',
          icon: '🔒',
          count: stats.collections?.confidential_documents || 0,
          description: '包含机密信息的文档库',
        },
      ]
    : [];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">向量数据库管理</h1>
        <p className="page-subtitle">管理公开、受限、机密文档的向量数据</p>
      </div>

      <div className="card">
        <div className="card-title">
          <Space>
            <DatabaseOutlined />
            数据库统计
          </Space>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value" style={{ color: '#55A722' }}>
              {stats?.collections?.public_documents || 0}
            </div>
            <div className="stat-label">公开文档向量</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ color: '#FA8C16' }}>
              {stats?.collections?.restricted_documents || 0}
            </div>
            <div className="stat-label">受限文档向量</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ color: '#EC4F4F' }}>
              {stats?.collections?.confidential_documents || 0}
            </div>
            <div className="stat-label">机密文档向量</div>
          </div>
        </div>

        <div style={{ marginTop: '24px', textAlign: 'right' }}>
          <Space>
            <Button icon={<SyncOutlined />} onClick={loadStats} loading={loading}>
              刷新
            </Button>
            <Button
              type="primary"
              icon={<ReloadOutlined />}
              onClick={handleImportAll}
              loading={ingesting && importingLevel === 'all'}
              disabled={ingesting && importingLevel !== 'all'}
            >
              导入全部文档
            </Button>
            {ingesting && (
              <Button danger icon={<StopOutlined />} onClick={handleStop}>
                停止
              </Button>
            )}
            <Button
              danger
              icon={<DeleteOutlined />}
              onClick={handleReset}
              loading={resetting}
              disabled={ingesting}
            >
              重置数据库
            </Button>
          </Space>
        </div>
      </div>

      {ingesting && (
        <div className="card">
          <div className="card-title">入库进度</div>
          <Progress percent={ingestProgress} status="active" />
          <div style={{ marginTop: '8px', textAlign: 'center', color: '#7E7E7E' }}>
            {progressMessage || '正在向量化并入库...'}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">文档集合管理</div>
        <Table
          columns={columns}
          dataSource={dataSource}
          pagination={false}
          loading={loading}
        />
      </div>

      <div className="card">
        <div className="card-title">数据目录说明</div>
        <div style={{ color: '#535353', lineHeight: '1.8' }}>
          <p style={{ marginBottom: '12px' }}>
            <strong>数据目录结构：</strong>
          </p>
          <ul style={{ paddingLeft: '24px' }}>
            <li>
              <code style={{ background: '#F7F8FA', padding: '2px 6px' }}>MD/0Public/</code>
              {' '}- 公开文档目录
            </li>
            <li>
              <code style={{ background: '#F7F8FA', padding: '2px 6px' }}>MD/1Restricted/</code>
              {' '}- 受限文档目录
            </li>
            <li>
              <code style={{ background: '#F7F8FA', padding: '2px 6px' }}>MD/2Confidential/</code>
              {' '}- 机密文档目录
            </li>
          </ul>
          <p style={{ marginTop: '16px', color: '#7E7E7E' }}>
            将文档放入对应目录后，点击"导入/更新"按钮即可将文档向量化并存储到向量数据库中。
          </p>
        </div>
      </div>
    </div>
  );
};

export default DatabasePage;
