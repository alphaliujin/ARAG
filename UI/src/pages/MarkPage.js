import React, { useState, useEffect, useRef } from 'react';
import {
  Button,
  message,
  Table,
  Space,
  Tag,
  Card,
  Spin,
  Statistic,
  Row,
  Col,
  Progress,
  Divider,
  Popconfirm,
  Select,
  Checkbox,
} from 'antd';
import {
  EyeOutlined,
  DownloadOutlined,
  DeleteOutlined,
  BarChartOutlined,
  StopOutlined,
  ClearOutlined,
} from '@ant-design/icons';
import { startDedupTask, pollTaskUntilDone, resumeOrStartPolling, cancelTask } from '../utils/taskApi';
import { apiClient } from '../utils/api';


const MarkPage = () => {
  const [dbStats, setDbStats] = useState({
    public_documents: 0,
    confidential_documents: 0,
    restricted_documents: 0,
  });

  // 数据去重状态
  // 结果按相似度自动分为三档: ≥0.8 高度相似 / 0.65-0.8 中度相似 / 0.5-0.65 弱相关
  const [isDeduping, setIsDeduping] = useState(false);
  const [dedupProgress, setDedupProgress] = useState(0);
  const [dedupMessage, setDedupMessage] = useState('');
  const [dedupCompleted, setDedupCompleted] = useState(0);
  const [dedupTotal, setDedupTotal] = useState(0);
  const [dedupResults, setDedupResults] = useState(null);
  const [dedupResultFiles, setDedupResultFiles] = useState([]);
  const dedupPollRef = useRef(null);
  const taskIdRef = useRef(null);

  // 去重参数
  const [dedupThreshold, setDedupThreshold] = useState(0.75);  // 下拉选项：相似度阈值
  const [autoDedup, setAutoDedup] = useState(false);           // 勾选框：是否自动去重
  const [isApplyingDedup, setIsApplyingDedup] = useState(false);  // 手动去重按钮状态

  // 加载数据库统计
  const loadStats = async () => {
    try {
      const res = await apiClient.get('/stats');
      if (res.data && res.data.collections) {
        setDbStats(res.data.collections);
      }
    } catch (err) {
      console.error('加载统计失败:', err);
    }
  };

  // 加载去重结果文件列表
  const loadDedupFiles = async () => {
    try {
      const res = await apiClient.get('/dedup/results');
      if (res.data) {
        setDedupResultFiles(res.data.files || []);
      }
    } catch (err) {
      console.error('加载去重结果文件失败:', err);
    }
  };

  // 删除去重结果文件
  const handleDeleteDedupFile = async (fileName) => {
    try {
      await apiClient.delete('/dedup/results/file', { params: { name: fileName } });
      message.success(`已删除: ${fileName}`);
      loadDedupFiles();
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || '未知错误';
      message.error(`删除失败: ${detail}`);
    }
  };

  // 查望去重结果文件: window.open 无法附加 X-API-Key 自定义头, 改为用 apiClient
  // (带认证拦截器) 拉取。用 responseType:'text' 让 axios 按响应的 charset=utf-8
  // 解码成 JS 字符串, 再包进 <meta charset=utf-8> 的 HTML blob 展示——
  // 直接对 text/markdown 用 blob 会被浏览器按 Latin-1 解码导致中文乱码。
  const handleViewDedupFile = async (fileName) => {
    try {
      const res = await apiClient.get('/dedup/results/file', {
        params: { name: fileName },
        responseType: 'text',
        transformResponse: [(d) => d], // 不让 axios 把 markdown 当 JSON 解析
      });
      const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      const html =
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>' + esc(fileName) + '</title>' +
        '<style>body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;' +
        'white-space:pre-wrap;padding:16px;word-break:break-word;}</style></head><body>' +
        esc(res.data) + '</body></html>';
      const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
      const url = window.URL.createObjectURL(blob);
      window.open(url, '_blank');
      // 给浏览器时间打开后释放 blob 引用
      setTimeout(() => window.URL.revokeObjectURL(url), 60000);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || '未知错误';
      message.error(`查看失败: ${detail}`);
    }
  };

  useEffect(() => {
    loadStats();
    loadDedupFiles();
    checkRunningDedupTask();
    return () => {
      if (dedupPollRef.current) clearInterval(dedupPollRef.current);
    };
  }, []);

  const checkRunningDedupTask = async () => {
    try {
      const { taskId, timer } = await resumeOrStartPolling(
        'dedup',
        (task) => {
          setIsDeduping(true);
          setDedupProgress(Math.round(task.progress * 100));
          setDedupMessage(task.progress_detail?.message || '去重比对进行中...');
          if (task.progress_detail?.completed !== undefined) {
            setDedupCompleted(task.progress_detail.completed);
          }
          if (task.progress_detail?.total !== undefined) {
            setDedupTotal(task.progress_detail.total);
          }
        },
        (task) => {
          setDedupProgress(100);
          setIsDeduping(false);
          taskIdRef.current = null;
          if (task.result) setDedupResults(task.result);
          message.success('数据去重比对完成');
          loadDedupFiles();
          loadStats();
          setTimeout(() => { setDedupProgress(0); setDedupMessage(''); setDedupCompleted(0); setDedupTotal(0); }, 3000);
        },
        (task) => {
          message.error(`去重失败: ${task.error || '未知错误'}`);
          setIsDeduping(false);
          setDedupProgress(0);
          setDedupMessage('');
          setDedupCompleted(0);
          setDedupTotal(0);
          taskIdRef.current = null;
        },
        (task) => {
          // onCancel
          setIsDeduping(false);
          setDedupProgress(0);
          setDedupMessage('');
          setDedupCompleted(0);
          setDedupTotal(0);
          taskIdRef.current = null;
          message.warning('去重已停止');
        },
      );
      if (taskId) {
        taskIdRef.current = taskId;
        dedupPollRef.current = timer;
        setIsDeduping(true);
      }
    } catch (err) {
      console.warn('[MarkPage] 检查运行任务失败:', err.message);
    }
  };

  // 执行数据去重
  const handleDedup = async () => {
    setIsDeduping(true);
    setDedupProgress(0);
    setDedupMessage('启动去重任务...');
    setDedupCompleted(0);
    setDedupTotal(0);
    setDedupResults(null);

    try {
      const params = {
        dedup_threshold: dedupThreshold,
        auto_dedup: autoDedup,
      };
      const result = await startDedupTask(params);

      if (result.message && result.task_id) {
        message.info(result.message);
      }

      taskIdRef.current = result.task_id;

      dedupPollRef.current = pollTaskUntilDone(
        result.task_id,
        (task) => {
          setDedupProgress(Math.round(task.progress * 100));
          setDedupMessage(task.progress_detail?.message || '去重比对进行中...');
          if (task.progress_detail?.completed !== undefined) {
            setDedupCompleted(task.progress_detail.completed);
          }
          if (task.progress_detail?.total !== undefined) {
            setDedupTotal(task.progress_detail.total);
          }
        },
        (task) => {
          setDedupProgress(100);
          setIsDeduping(false);
          taskIdRef.current = null;
          if (task.result) setDedupResults(task.result);
          message.success('数据去重比对完成');
          loadDedupFiles();
          loadStats();
          setTimeout(() => { setDedupProgress(0); setDedupMessage(''); setDedupCompleted(0); setDedupTotal(0); }, 3000);
        },
        (task) => {
          message.error(`去重失败: ${task.error || '未知错误'}`);
          setIsDeduping(false);
          setDedupProgress(0);
          setDedupMessage('');
          setDedupCompleted(0);
          setDedupTotal(0);
          taskIdRef.current = null;
        },
        (task) => {
          // onCancel
          setIsDeduping(false);
          setDedupProgress(0);
          setDedupMessage('');
          setDedupCompleted(0);
          setDedupTotal(0);
          taskIdRef.current = null;
          message.warning('去重已停止');
        },
      );
    } catch (error) {
      message.error('启动去重失败: ' + error.message);
      setIsDeduping(false);
      setDedupProgress(0);
      setDedupMessage('');
      setDedupCompleted(0);
      setDedupTotal(0);
    }
  };

  // 停止去重任务
  const handleStop = async () => {
    if (!taskIdRef.current) return;
    try {
      await cancelTask(taskIdRef.current);
      message.warning('去重已停止');
    } catch (error) {
      message.error('停止任务失败: ' + error.message);
    } finally {
      // 无论 cancelTask 成败都清理 UI 状态, 否则失败时 UI 永久卡在"处理中"
      if (dedupPollRef.current) {
        clearInterval(dedupPollRef.current);
        dedupPollRef.current = null;
      }
      setIsDeduping(false);
      setDedupProgress(0);
      setDedupMessage('');
      setDedupCompleted(0);
      setDedupTotal(0);
      taskIdRef.current = null;
    }
  };

  // 手动去重: 按照比对结果中超过阈值的记录，从高密级库删除
  const handleApplyDedup = async () => {
    setIsApplyingDedup(true);
    try {
      const res = await apiClient.post('/dedup/apply', {
        dedup_threshold: dedupThreshold,
      });
      if (res.data.status === 'success') {
        message.success(res.data.message);
        loadStats();
      } else if (res.data.status === 'skipped') {
        message.info(res.data.message);
      }
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || '未知错误';
      message.error(`去重失败: ${detail}`);
    } finally {
      setIsApplyingDedup(false);
    }
  };

  // 去重结果表格列
  const dedupColumns = [
    {
      title: '比对对',
      dataIndex: 'pair_name',
      key: 'pair_name',
      render: (text) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: '高度相似(≥0.8)',
      dataIndex: 'high_similarity',
      key: 'high_similarity',
      render: (v) => <span style={{ color: '#EC4F4F', fontWeight: 600 }}>{v}</span>,
    },
    {
      title: '中度相似(0.65-0.8)',
      dataIndex: 'medium_similarity',
      key: 'medium_similarity',
      render: (v) => <span style={{ color: '#FA8C16', fontWeight: 600 }}>{v}</span>,
    },
    {
      title: '弱相关(0.5-0.65)',
      dataIndex: 'low_similarity',
      key: 'low_similarity',
      render: (v) => <span style={{ color: '#8c8c8c' }}>{v}</span>,
    },
    {
      title: '匹配对数',
      dataIndex: 'total_matches',
      key: 'total_matches',
      render: (v) => <span style={{ fontWeight: 600 }}>{v}</span>,
    },
    {
      title: '比对总数',
      dataIndex: 'total_checked',
      key: 'total_checked',
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">数据优化</h1>
        <p className="page-subtitle">对入库文档进行去重比对与数据优化</p>
      </div>

      <div className="card">
        <div className="card-title">
          <Space>
            <BarChartOutlined />
            跨密级数据比对
          </Space>
        </div>

        <div style={{ padding: '16px 0' }}>
          <p style={{ color: '#8c8c8c', fontSize: '13px', marginBottom: '24px', textAlign: 'center' }}>
            比对受限-公开 / 机密-公开 / 机密-受限 三组数据，结果按相似度分为
            <span style={{ color: '#EC4F4F' }}> 高度相似 </span>/
            <span style={{ color: '#FA8C16' }}> 中度相似 </span>/
            <span style={{ color: '#8c8c8c' }}> 弱相关 </span>三档。
          </p>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap' }}>
            <Button
              type="primary"
              icon={<BarChartOutlined />}
              onClick={handleDedup}
              loading={isDeduping}
              size="large"
              disabled={isDeduping}
            >
              开始数据比对
            </Button>
            {isDeduping && (
              <Button
                danger
                size="large"
                icon={<StopOutlined />}
                onClick={handleStop}
              >
                停止
              </Button>
            )}
            <Select
              value={dedupThreshold}
              onChange={setDedupThreshold}
              style={{ width: 160 }}
              disabled={isDeduping}
              options={[
                ...Array.from({ length: 8 }, (_, i) => {
                  const v = 0.6 + i * 0.05;
                  return { value: v, label: `相似度 ≥ ${v.toFixed(2)}` };
                }),
              ]}
            />
            <Checkbox
              checked={autoDedup}
              onChange={(e) => setAutoDedup(e.target.checked)}
              disabled={isDeduping}
            >
              自动去重
            </Checkbox>
            <Popconfirm
              title={`确认按照比对结果去重？将删除相似度 ≥ ${dedupThreshold} 的数据，此操作不可恢复。`}
              onConfirm={handleApplyDedup}
              okText="确认去重"
              cancelText="取消"
              okButtonProps={{ danger: true }}
            >
              <Button
                danger
                icon={<ClearOutlined />}
                loading={isApplyingDedup}
                disabled={isDeduping || isApplyingDedup}
              >
                去重
              </Button>
            </Popconfirm>
            <Button
              icon={<DownloadOutlined />}
              onClick={loadDedupFiles}
              disabled={isDeduping}
            >
              刷新结果
            </Button>
          </div>

          {isDeduping && (
            <div style={{ marginBottom: '24px' }}>
              <Progress percent={dedupProgress} status="active" />
              <div style={{ textAlign: 'center', color: '#7E7E7E', marginTop: '8px' }}>
                <Spin size="small" style={{ marginRight: '8px' }} />
                {dedupTotal > 0
                  ? `已完成比对 ${dedupCompleted}/${dedupTotal} 次`
                  : (dedupMessage || '正在启动比对任务...')}
              </div>
            </div>
          )}
        </div>

        <Divider />

        {/* 比对结果统计 */}
        {dedupResults && dedupResults.comparisons && (
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ marginBottom: '16px' }}>比对结果统计</h3>
            <Table
              columns={dedupColumns}
              dataSource={Object.entries(dedupResults.comparisons).map(([label, data]) => ({
                key: label,
                pair_name: label,
                ...data,
              }))}
              pagination={false}
              size="small"
            />
            <div style={{ textAlign: 'right', marginTop: '8px', color: '#8c8c8c' }}>
              总耗时: {dedupResults.elapsed_seconds?.toFixed(1)} 秒
            </div>
          </div>
        )}

        {/* 结果文件列表 */}
        {dedupResultFiles.length > 0 && (
          <div>
            <Divider />
            <h3 style={{ marginBottom: '16px' }}>结果文件</h3>
            <Table
              columns={[
                {
                  title: '文件名',
                  dataIndex: 'name',
                  key: 'name',
                },
                {
                  title: '大小',
                  dataIndex: 'size',
                  key: 'size',
                  render: (size) => `${(size / 1024).toFixed(1)} KB`,
                },
                {
                  title: '操作',
                  key: 'action',
                  render: (_, record) => (
                    <Space>
                      <Button
                        type="link"
                        icon={<EyeOutlined />}
                        onClick={() => handleViewDedupFile(record.name)}
                      >
                        查看
                      </Button>
                      <Popconfirm
                        title="确定删除该结果文件？"
                        onConfirm={() => handleDeleteDedupFile(record.name)}
                        okText="删除"
                        cancelText="取消"
                        okButtonProps={{ danger: true }}
                      >
                        <Button type="link" danger icon={<DeleteOutlined />}>
                          删除
                        </Button>
                      </Popconfirm>
                    </Space>
                  ),
                },
              ]}
              dataSource={dedupResultFiles.map((f, i) => ({ ...f, key: i }))}
              pagination={false}
              size="small"
            />
          </div>
        )}
      </div>

      {/* 数据优化统计 */}
      <div className="card">
        <div className="card-title">数据优化统计</div>
        <Row gutter={24}>
          <Col span={8}>
            <Card bordered={false} style={{ textAlign: 'center', background: '#f6ffed' }}>
              <Statistic
                title="公开文档条目"
                value={dbStats.public_documents}
                valueStyle={{ color: '#55A722' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card bordered={false} style={{ textAlign: 'center', background: '#fff7e6' }}>
              <Statistic
                title="受限文档条目"
                value={dbStats.restricted_documents}
                valueStyle={{ color: '#FA8C16' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card bordered={false} style={{ textAlign: 'center', background: '#fff1f0' }}>
              <Statistic
                title="机密文档条目"
                value={dbStats.confidential_documents}
                valueStyle={{ color: '#EC4F4F' }}
              />
            </Card>
          </Col>
        </Row>
        <div style={{ textAlign: 'center', marginTop: 16 }}>
          <Button onClick={loadStats}>刷新统计</Button>
        </div>
      </div>
    </div>
  );
};

export default MarkPage;
