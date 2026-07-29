import React, { useState, useEffect, useRef } from 'react';
import {
  Button,
  message,
  Table,
  Space,
  Tag,
  Progress,
  Select,
  Card,
  Statistic,
  Row,
  Col,
  Modal,
  Badge,
} from 'antd';
import {
  DatabaseOutlined,
  ReloadOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  FileTextOutlined,
  ClockCircleOutlined,
  StopOutlined,
} from '@ant-design/icons';
import { startIngestTask, pollTaskUntilDone, resumeOrStartPolling, cancelTask } from '../utils/taskApi';
import { apiClient } from '../utils/api';

const { Option } = Select;


const IngestPage = () => {
  const [stats, setStats] = useState({
    public: 0, restricted: 0, confidential: 0, total: 0,
  });
  const [pendingStats, setPendingStats] = useState({
    public: 0, restricted: 0, confidential: 0, total: 0,
  });
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [selectedClassification, setSelectedClassification] = useState('all');
  const [embeddingModel, setEmbeddingModel] = useState('ollama-bge-m3');
  const [chunkStrategy, setChunkStrategy] = useState('auto');
  const [fileList, setFileList] = useState([]);
  const [clearing, setClearing] = useState(false);
  const [modelChangeModal, setModelChangeModal] = useState(false);
  const [pendingModel, setPendingModel] = useState(null);

  const pollTimerRef = useRef(null);
  const taskIdRef = useRef(null);
  const autoRefreshTimerRef = useRef(null);

  // 挂载时: 加载统计 + 检查是否有运行中的入库任务 + 启动文档集合自动刷新
  useEffect(() => {
    loadStatus();
    checkRunningTask();
    // 文档集合每 5s 静默刷新一次 — 仅在页签可见时刷新,避免后台标签页空转。
    // Page Visibility API 触发 visibilitychange 时按需启停定时器。
    const startAutoRefresh = () => {
      if (autoRefreshTimerRef.current) return;
      autoRefreshTimerRef.current = setInterval(() => loadStatus(true), 5000);
    };
    const stopAutoRefresh = () => {
      if (autoRefreshTimerRef.current) {
        clearInterval(autoRefreshTimerRef.current);
        autoRefreshTimerRef.current = null;
      }
    };
    const handleVisibilityChange = () => {
      if (document.hidden) stopAutoRefresh();
      else { loadStatus(true); startAutoRefresh(); }
    };
    if (!document.hidden) startAutoRefresh();
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      // 卸载时停止轮询,但后台任务不受影响
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      stopAutoRefresh();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const checkRunningTask = async () => {
    try {
      const { taskId, timer } = await resumeOrStartPolling(
        'ingest',
        (task) => {
          setIngesting(true);
          setIngestProgress(Math.round(task.progress * 100));
          setProgressMessage(task.progress_detail?.message || '入库进行中...');
        },
        (task) => {
          setIngestProgress(100);
          setProgressMessage('入库完成');
          message.success(`数据入库完成`);
          setIngesting(false);
          taskIdRef.current = null;
          loadStatus();
          setTimeout(() => { setIngestProgress(0); setProgressMessage(''); }, 3000);
        },
        (task) => {
          message.error(`入库失败: ${task.error || '未知错误'}`);
          setIngesting(false);
          setIngestProgress(0);
          setProgressMessage('');
          taskIdRef.current = null;
        },
        (task) => {
          // onCancel
          setIngesting(false);
          setIngestProgress(0);
          setProgressMessage('');
          taskIdRef.current = null;
          message.warning('入库已停止');
        },
      );
      // resumeOrStartPolling 只在 running/pending 时返回非空 taskId,
      // 终态任务由对应回调处理且返回 null, 不会误把"停止"按钮指向已完成任务。
      if (taskId) {
        taskIdRef.current = taskId;
        pollTimerRef.current = timer;
        setIngesting(true);
      }
    } catch (err) {
      // 检查失败不影响页面正常使用
      console.warn('[IngestPage] 检查运行任务失败:', err.message);
    }
  };

  // silent=true 时不切换 loading 状态、不弹错误提示,
  // 供 5s 自动刷新使用, 避免按钮闪烁/网络抖动反复弹 toast。
  const loadStatus = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const response = await apiClient.get(`/status`);
      const data = response.data;
      const data2status = (cls) => (data.indexed?.[cls] || 0) > 0 ? 'indexed' : 'pending';
      setPendingStats({
        public: data.pending?.public || 0,
        restricted: data.pending?.restricted || 0,
        confidential: data.pending?.confidential || 0,
        total: data.pending_total || 0,
      });
      setStats({
        public: data.indexed?.public || 0,
        restricted: data.indexed?.restricted || 0,
        confidential: data.indexed?.confidential || 0,
        total: data.indexed_total || 0,
      });
      if (data.embedding_model) {
        setEmbeddingModel(data.embedding_model);
      }
      setFileList([
        { key: '1', name: '公开文档集', classification: 'public',
          pending: data.pending?.public || 0, indexed: data.indexed?.public || 0,
          status: data2status('public') },
        { key: '2', name: '受限文档集', classification: 'restricted',
          pending: data.pending?.restricted || 0, indexed: data.indexed?.restricted || 0,
          status: data2status('restricted') },
        { key: '3', name: '机密文档集', classification: 'confidential',
          pending: data.pending?.confidential || 0, indexed: data.indexed?.confidential || 0,
          status: data2status('confidential') },
      ]);
    } catch (error) {
      if (!silent) message.error('获取统计失败: ' + (error.response?.data?.detail || error.message || '未知错误'));
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const handleModelChange = (value) => {
    if (value === embeddingModel) return;
    setPendingModel(value);
    setModelChangeModal(true);
  };

  const confirmModelChange = async () => {
    setModelChangeModal(false);
    if (!pendingModel) return;
    try {
      // 先把新模型持久化到 /settings/model (settings_service 会校验白名单并应用到运行时),
      // 否则后续 loadStatus (含 5s 自动刷新) 会从服务器读回旧值, 把本地状态覆盖回去,
      // 造成"看似已切换、实际仍用旧模型入库"的假切换。
      await apiClient.put('/settings/model', { values: { embeddingModel: pendingModel } });
      setEmbeddingModel(pendingModel);
      await apiClient.delete(`/reset`);
      message.success('向量数据库已清空，嵌入模型已切换');
      await loadStatus();
      setFileList([]);
    } catch (error) {
      message.error('切换模型失败: ' + (error.response?.data?.detail || error.message || '未知错误'));
      // 持久化或清库失败时, 静默回读服务器实际模型, 避免本地与服务器不一致
      await loadStatus(true);
    }
    setPendingModel(null);
  };

  const cancelModelChange = () => {
    setModelChangeModal(false);
    setPendingModel(null);
  };

  // 入库: 启动后台任务 + 轮询进度
  const handleIngest = async () => {
    setIngesting(true);
    setIngestProgress(0);
    setProgressMessage('启动入库任务...');

    try {
      const result = await startIngestTask({
        classification: selectedClassification === 'all' ? null : selectedClassification,
        strategy: chunkStrategy,
        include_images: true,
        force: false,
        embedding_model: embeddingModel,
      });

      // 如果返回已有运行任务的消息,自动切换到轮询
      taskIdRef.current = result.task_id;
      setProgressMessage(result.message || '入库任务已启动');

      // 开始轮询
      pollTimerRef.current = pollTaskUntilDone(
        result.task_id,
        (task) => {
          setIngestProgress(Math.round(task.progress * 100));
          setProgressMessage(task.progress_detail?.message || '入库进行中...');
        },
        (task) => {
          setIngestProgress(100);
          setProgressMessage('入库完成');
          const detail = task.result || {};
          message.success(`数据入库完成: ${detail.chunks_added || detail.parents_added + detail.children_added || 0} 个切片`);
          setIngesting(false);
          taskIdRef.current = null;
          loadStatus();
          setTimeout(() => { setIngestProgress(0); setProgressMessage(''); }, 3000);
        },
        (task) => {
          message.error(`入库失败: ${task.error || '未知错误'}`);
          setIngesting(false);
          setIngestProgress(0);
          setProgressMessage('');
          taskIdRef.current = null;
        },
        (task) => {
          // onCancel
          setIngesting(false);
          setIngestProgress(0);
          setProgressMessage('');
          taskIdRef.current = null;
          message.warning('入库已停止');
        },
      );
    } catch (error) {
      message.error('启动入库失败: ' + error.message);
      setIngesting(false);
      setIngestProgress(0);
      setProgressMessage('');
    }
  };

  // 停止入库任务
  const handleStop = async () => {
    if (!taskIdRef.current) return;
    try {
      await cancelTask(taskIdRef.current);
      message.warning('入库已停止');
    } catch (error) {
      message.error('停止任务失败: ' + error.message);
    } finally {
      // 无论 cancelTask 成败都清理 UI 状态, 否则失败时 UI 永久卡在"处理中"
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      setIngesting(false);
      setIngestProgress(0);
      setProgressMessage('');
      taskIdRef.current = null;
    }
  };

  const handleClear = () => {
    Modal.confirm({
      title: '确认清空数据库',
      content: '确定要清空所有向量数据吗？此操作不可恢复。',
      okText: '确认清空',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        setClearing(true);
        try {
          await apiClient.delete(`/reset`);
          message.success('向量数据库已清空');
          await loadStatus();
          setFileList([]);
        } catch (error) {
          message.error('清空数据库失败: ' + (error.response?.data?.detail || error.message || '未知错误'));
        } finally {
          setClearing(false);
        }
      },
    });
  };

  const getClassificationTag = (classification) => {
    switch (classification) {
      case 'public': return <Tag color="success">公开</Tag>;
      case 'restricted': return <Tag color="warning">受限</Tag>;
      case 'confidential': return <Tag color="error">机密</Tag>;
      default: return <Tag>未知</Tag>;
    }
  };

  const columns = [
    {
      title: '文档集名称', dataIndex: 'name', key: 'name',
      render: (text) => (<Space><FileTextOutlined /><span>{text}</span></Space>),
    },
    {
      title: '密级', dataIndex: 'classification', key: 'classification',
      render: (classification) => getClassificationTag(classification),
    },
    {
      title: '待入库', dataIndex: 'pending', key: 'pending',
      render: (pending) => (<Space><ClockCircleOutlined style={{ color: '#1890ff' }} /><span style={{ color: '#1890ff', fontWeight: 500 }}>{pending}</span></Space>),
    },
    {
      title: '已入库', dataIndex: 'indexed', key: 'indexed',
      render: (indexed) => (<Space><CheckCircleOutlined style={{ color: '#55A722' }} /><span style={{ color: '#55A722', fontWeight: 500 }}>{indexed}</span></Space>),
    },
    {
      title: '状态', dataIndex: 'status', key: 'status',
      render: (status) => {
        if (status === 'indexed') return <Badge status="success" text="已入库" />;
        if (status === 'indexing') return <Badge status="processing" text="入库中" />;
        return <Badge status="default" text="待入库" />;
      },
    },
    {
      title: '操作', key: 'action',
      render: (_, record) => (<Space><Button type="link" onClick={() => message.warning('重新索引功能尚未实现')}>重新索引</Button></Space>),
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">数据入库</h1>
        <p className="page-subtitle">将预处理后的 Markdown 切片向量化并存入向量数据库</p>
      </div>

      <Row gutter={[24, 24]} style={{ marginBottom: '12px' }}>
        <Col span={8}>
          <Card size="small">
            <Statistic title="待入库文件 - 公开" value={pendingStats.public} valueStyle={{ color: '#1890ff' }} prefix={<ClockCircleOutlined />} />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic title="待入库文件 - 受限" value={pendingStats.restricted} valueStyle={{ color: '#1890ff' }} prefix={<ClockCircleOutlined />} />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic title="待入库文件 - 机密" value={pendingStats.confidential} valueStyle={{ color: '#1890ff' }} prefix={<ClockCircleOutlined />} />
          </Card>
        </Col>
      </Row>

      <Row gutter={[24, 24]} style={{ marginBottom: '24px' }}>
        <Col span={8}>
          <Card size="small">
            <Statistic title="已入库条目 - 公开" value={stats.public} valueStyle={{ color: '#55A722' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic title="已入库条目 - 受限" value={stats.restricted} valueStyle={{ color: '#FA8C16' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic title="已入库条目 - 机密" value={stats.confidential} valueStyle={{ color: '#EC4F4F' }} prefix={<CheckCircleOutlined />} />
          </Card>
        </Col>
      </Row>

      <div className="card">
        <div className="card-title">
          <Space><DatabaseOutlined />入库配置</Space>
          <span style={{ float: 'right', color: '#7E7E7E', fontSize: '12px' }}>
            待入库: {pendingStats.total} | 已入库: {stats.total} | 总计: {pendingStats.total + stats.total}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '24px', marginBottom: '24px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>目标密级</label>
            <Select value={selectedClassification} onChange={setSelectedClassification} style={{ width: '100%' }}>
              <Option value="all">全部密级</Option>
              <Option value="public">公开</Option>
              <Option value="restricted">受限</Option>
              <Option value="confidential">机密</Option>
            </Select>
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>嵌入模型</label>
            <Select value={embeddingModel} onChange={handleModelChange} style={{ width: '100%' }}>
              <Option value="ollama-bge-m3">Ollama (bge-m3, 1024维) — 推荐</Option>
              <Option value="mps-bge-m3">MPS (bge-m3, 1024维)</Option>
            </Select>
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>切片策略</label>
            <Select value={chunkStrategy} onChange={setChunkStrategy} style={{ width: '100%' }}>
              <Option value="auto">自动检测</Option>
              <Option value="parent-child">父子块切片</Option>
              <Option value="chunk">普通切片</Option>
            </Select>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
          <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleIngest} loading={ingesting} disabled={ingesting} size="large">
            开始入库
          </Button>
          {ingesting && (
            <Button danger icon={<StopOutlined />} onClick={handleStop} size="large">
              停止
            </Button>
          )}
          <Button icon={<ReloadOutlined />} onClick={loadStatus} loading={loading}>
            刷新统计
          </Button>
          <Button danger icon={<DeleteOutlined />} onClick={handleClear} loading={clearing}>
            清空数据库
          </Button>
        </div>
      </div>

      {/* 进度条: 有任务运行时始终显示,切换页面回来后也能看到 */}
      {ingesting && (
        <div className="card">
          <div className="card-title">入库进度</div>
          <Progress percent={ingestProgress} status="active" />
          <div style={{ marginTop: '8px', textAlign: 'center', color: '#7E7E7E' }}>
            <LoadingOutlined style={{ marginRight: '8px' }} />
            {progressMessage || '正在向量化并入库...'}
          </div>
        </div>
      )}

      <Modal
        title="⚠️ 切换嵌入模型"
        open={modelChangeModal}
        onOk={confirmModelChange}
        onCancel={cancelModelChange}
        okText="确认切换并清空"
        okType="danger"
        cancelText="取消"
        closable={false}
        maskClosable={false}
      >
        <div style={{ marginBottom: '16px' }}>
          <p><strong>当前模型:</strong> {embeddingModel === 'mps-bge-m3' ? 'MPS (bge-m3, 1024维)' : embeddingModel === 'ollama-bge-m3' ? 'Ollama (bge-m3, 1024维)' : embeddingModel}</p>
          <p><strong>目标模型:</strong> {pendingModel === 'mps-bge-m3' ? 'MPS (bge-m3, 1024维)' : pendingModel === 'ollama-bge-m3' ? 'Ollama (bge-m3, 1024维)' : pendingModel}</p>
        </div>
        <div style={{ background: '#fff2f0', border: '1px solid #ffccc7', borderRadius: '4px', padding: '12px 16px', marginBottom: '8px' }}>
          <p style={{ color: '#cf1322', margin: 0, fontWeight: 500 }}>⚠️ 警告：此操作将清空当前向量数据库</p>
          <p style={{ color: '#595959', margin: '8px 0 0 0', fontSize: '13px' }}>
            不同嵌入模型产生的向量维度不同，混用会导致数据不兼容。<br/>
            切换模型后，所有已入库的向量数据将被清空，需要重新执行数据入库。
          </p>
        </div>
        <p style={{ color: '#8c8c8c', fontSize: '12px', marginTop: '12px' }}>
          确认后将立即清空向量数据库，此操作不可恢复。
        </p>
      </Modal>

      <div className="card">
        <div className="card-title">文档集合</div>
        <Table columns={columns} dataSource={fileList} pagination={false} loading={loading} />
      </div>
    </div>
  );
};

export default IngestPage;