import React, { useState, useEffect, useRef } from 'react';
import { Upload, Button, Tabs, Input, message, Table, Popconfirm, Tag, Space, Spin, Steps, Tooltip, Progress } from 'antd';
import {
  InboxOutlined, FileTextOutlined, EditOutlined, DeleteOutlined, FolderOutlined,
  ReloadOutlined, FileSearchOutlined, ApartmentOutlined, ThunderboltOutlined, StopOutlined,
} from '@ant-design/icons';
import {
  scanDocument, scanText, formatConfidence,
  listDocScanFiles, deleteDocScanFile, clearAllDocScanFiles, getDocScanStats,
  docscanPreprocess, docscanEmbed, docscanCompare, docscanStatus,
  getLevelLabel, getLevelColor,
} from '../utils/api';
import { pollTaskUntilDone, cancelTask, resumeOrStartPolling } from '../utils/taskApi';

const { TextArea } = Input;
const { Dragger } = Upload;

// 上传前端预校验: Dragger 的 accept 只是选择器提示, customRequest 不强制, 需显式检查
const ALLOWED_UPLOAD_EXTS = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.html', '.md'];
const MAX_UPLOAD_MB = 50;

const ScanPage = () => {
  // 文本扫描
  const [textValue, setTextValue] = useState('');
  const [textScanning, setTextScanning] = useState(false);
  const [textResult, setTextResult] = useState(null);

  // 文件管理
  const [docScanFiles, setDocScanFiles] = useState([]);
  const [docScanLoading, setDocScanLoading] = useState(false);
  const [clearingAll, setClearingAll] = useState(false);
  const [docScanStats, setDocScanStats] = useState({ total_files: 0, preprocessed: 0, embedded: 0, compared: 0 });

  // 选中文件 + 状态 + 操作 loading
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileStatus, setFileStatus] = useState({ saved: false, preprocessed: false, embedded: false, compared: false });
  const [opLoading, setOpLoading] = useState({ preprocess: false, embed: false, compare: false });
  const [lastResult, setLastResult] = useState(null);  // 最近一次操作结果

  // 生成向量后台任务(分批嵌入 + 进度轮询, 避免长文档 HTTP 超时)
  const [embedProgress, setEmbedProgress] = useState(0);
  const [embedMessage, setEmbedMessage] = useState('');
  const embedTaskIdRef = useRef(null);
  const embedPollRef = useRef(null);

  // ============ 数据获取 ============

  const fetchDocScanFiles = async () => {
    setDocScanLoading(true);
    try {
      const result = await listDocScanFiles();
      setDocScanFiles(result.files || []);
    } catch (error) {
      message.error('获取文件列表失败');
    } finally {
      setDocScanLoading(false);
    }
  };

  const fetchDocScanStats = async () => {
    try {
      const result = await getDocScanStats();
      setDocScanStats(result);
    } catch (e) {}
  };

  const refreshSelectedStatus = async (filename) => {
    if (!filename) return;
    try {
      const status = await docscanStatus(filename);
      setFileStatus(status);
    } catch (e) {}
  };

  useEffect(() => {
    fetchDocScanFiles();
    fetchDocScanStats();
    // 恢复运行中的"生成向量"任务(页面刷新/重开时继续轮询)
    resumeOrStartPolling(
      'docscan_embed',
      (t) => {
        setOpLoading(prev => ({ ...prev, embed: true }));
        setEmbedProgress(Math.round((t.progress || 0) * 100));
        setEmbedMessage(t.progress_detail?.message || '嵌入中...');
      },
      (t) => {
        message.success(t.result?.message || '向量生成完成');
        setOpLoading(prev => ({ ...prev, embed: false }));
        fetchDocScanStats();
      },
      (t) => {
        message.error(`生成向量失败: ${t.error || '未知错误'}`);
        setOpLoading(prev => ({ ...prev, embed: false }));
      },
      () => {
        message.warning('生成向量已停止');
        setOpLoading(prev => ({ ...prev, embed: false }));
      },
    ).then(({ taskId, timer }) => {
      if (taskId) {
        embedTaskIdRef.current = taskId;
        embedPollRef.current = timer;
      }
    });
    return () => {
      if (embedPollRef.current) clearInterval(embedPollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ============ 文件上传 (只保存) ============

  // 用 customRequest 接管上传 — Dragger 的 onChange 会因 status 转移触发多次,
  // 此处直接接管整个上传流程,从根上避免重复调用。
  // antd 会传入 { file, onSuccess, onError }; 我们用 file 直接调 scanDocument。
  const handleCustomUpload = async ({ file, onSuccess, onError }) => {
    // 前端预校验: 扩展名 + 大小, 避免非法/超大文件白传一遍再被后端拒
    const ext = (file.name.match(/\.[^.]+$/) || [''])[0].toLowerCase();
    if (!ALLOWED_UPLOAD_EXTS.includes(ext)) {
      message.error(`不支持的文件类型: ${ext || '(无扩展名)'}`);
      onError && onError(new Error('Unsupported file type'));
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      message.error(`文件过大 (>${MAX_UPLOAD_MB}MB)`);
      onError && onError(new Error('File too large'));
      return;
    }
    try {
      const result = await scanDocument(file);
      message.success(result.message || '文件已保存');
      setSelectedFile(result.saved_filename);
      setFileStatus({ saved: true, preprocessed: false, embedded: false, compared: false });
      setLastResult(null);
      fetchDocScanFiles();
      fetchDocScanStats();
      onSuccess && onSuccess({}, file);
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      message.error(`文件上传失败: ${detail}`);
      onError && onError(error);
    }
  };

  // ============ 三个操作按钮 ============

  const handlePreprocess = async () => {
    if (!selectedFile) {
      message.warning('请先选择文件');
      return;
    }
    setOpLoading(prev => ({ ...prev, preprocess: true }));
    try {
      const result = await docscanPreprocess(selectedFile);
      if (result.success) {
        message.success('预处理完成');
        setLastResult({ type: 'preprocess', data: result });
        await refreshSelectedStatus(selectedFile);
        fetchDocScanStats();
      } else {
        message.error(`预处理失败: ${result.message || '未知错误'}`);
      }
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      message.error(`预处理失败: ${detail}`);
    } finally {
      setOpLoading(prev => ({ ...prev, preprocess: false }));
    }
  };

  const handleEmbed = async () => {
    if (!selectedFile) {
      message.warning('请先选择文件');
      return;
    }
    if (!fileStatus.preprocessed) {
      message.warning('请先执行预处理');
      return;
    }
    setOpLoading(prev => ({ ...prev, embed: true }));
    setEmbedProgress(0);
    setEmbedMessage('启动生成向量任务...');
    try {
      // 后台任务: 立即返回 task_id, 前端轮询进度(无 HTTP 超时, 长文档也不卡死)
      const res = await docscanEmbed(selectedFile);
      const taskId = res?.task_id;
      if (!taskId) {
        message.error('启动生成向量失败');
        setOpLoading(prev => ({ ...prev, embed: false }));
        return;
      }
      embedTaskIdRef.current = taskId;
      embedPollRef.current = pollTaskUntilDone(
        taskId,
        (t) => {
          setEmbedProgress(Math.round((t.progress || 0) * 100));
          setEmbedMessage(t.progress_detail?.message || '嵌入中...');
        },
        (t) => {
          message.success(t.result?.message || '向量生成完成');
          setOpLoading(prev => ({ ...prev, embed: false }));
          setLastResult({ type: 'embed', data: t.result });
          refreshSelectedStatus(selectedFile);
          fetchDocScanStats();
          embedTaskIdRef.current = null;
        },
        (t) => {
          message.error(`生成向量失败: ${t.error || '未知错误'}`);
          setOpLoading(prev => ({ ...prev, embed: false }));
          embedTaskIdRef.current = null;
        },
        () => {
          message.warning('生成向量已停止');
          setOpLoading(prev => ({ ...prev, embed: false }));
          embedTaskIdRef.current = null;
        },
      );
    } catch (error) {
      const detail = error.response?.data?.detail || error.message || '未知错误';
      message.error(`启动生成向量失败: ${detail}`);
      setOpLoading(prev => ({ ...prev, embed: false }));
    }
  };

  const handleStopEmbed = async () => {
    if (!embedTaskIdRef.current) return;
    try {
      await cancelTask(embedTaskIdRef.current);
    } catch (error) {
      message.error('停止失败: ' + (error.message || '未知错误'));
    } finally {
      // 轮询的 onCancel 会清状态; 这里兜底清 timer, 防网络异常时卡住
      if (embedPollRef.current) {
        clearInterval(embedPollRef.current);
        embedPollRef.current = null;
      }
    }
  };

  const handleCompare = async () => {
    if (!selectedFile) {
      message.warning('请先选择文件');
      return;
    }
    if (!fileStatus.embedded) {
      message.warning('请先生成向量');
      return;
    }
    setOpLoading(prev => ({ ...prev, compare: true }));
    try {
      const result = await docscanCompare(selectedFile, 10);
      if (result.status === 'success') {
        message.success(result.message);
        setLastResult({ type: 'compare', data: result });
        await refreshSelectedStatus(selectedFile);
        fetchDocScanStats();
      } else {
        message.error(`比对失败: ${result.message || '未知错误'}`);
      }
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      message.error(`比对失败: ${detail}`);
    } finally {
      setOpLoading(prev => ({ ...prev, compare: false }));
    }
  };

  // ============ 文件管理 ============

  const handleDeleteDocScanFile = async (name) => {
    try {
      await deleteDocScanFile(name);
      message.success(`已删除: ${name}`);
      if (selectedFile === name) {
        setSelectedFile(null);
        setFileStatus({ saved: false, preprocessed: false, embedded: false, compared: false });
        setLastResult(null);
      }
      fetchDocScanFiles();
      fetchDocScanStats();
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      message.error(`删除失败: ${detail}`);
    }
  };

  const handleClearAllDocScanFiles = async () => {
    setClearingAll(true);
    try {
      const res = await clearAllDocScanFiles();
      message.success(res.message || '已清空 DocScan');
      setSelectedFile(null);
      setFileStatus({ saved: false, preprocessed: false, embedded: false, compared: false });
      setLastResult(null);
      fetchDocScanFiles();
      fetchDocScanStats();
    } catch (error) {
      const detail = error.response?.data?.detail || error.message || '未知错误';
      message.error(`清空失败: ${detail}`);
    } finally {
      setClearingAll(false);
    }
  };

  const handleSelectFile = async (name) => {
    setSelectedFile(name);
    setLastResult(null);
    await refreshSelectedStatus(name);
  };

  // ============ 文本扫描 ============

  const handleTextScan = async () => {
    if (!textValue || textValue.trim().length < 10) {
      message.warning('请输入至少 10 个字符的文本内容');
      return;
    }
    setTextScanning(true);
    try {
      const result = await scanText(textValue);
      setTextResult(result);
      message.success('文本扫描完成');
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      message.error(`文本扫描失败: ${detail}`);
    } finally {
      setTextScanning(false);
    }
  };

  // ============ 渲染辅助 ============

  const renderSegments = (segments) => {
    if (!segments || segments.length === 0) {
      return <div className="empty-state">未检测到敏感信息</div>;
    }
    return (
      <div className="segment-list">
        {segments.map((segment, index) => {
          const level = segment.level || 'unknown';
          return (
            <div key={index} className="segment-item">
              <div className="segment-header">
                <span className={`segment-level ${level}`}>{getLevelLabel(level)}</span>
                <span className="segment-confidence">置信度: {formatConfidence(segment.confidence)}</span>
              </div>
              <div className="segment-content">{segment.content}</div>
              {segment.matched_source && (
                <div className="segment-source">匹配来源: {segment.matched_source}</div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const renderResultSummary = (summary) => {
    if (!summary) return null;
    const confidentialCount = summary.confidential || 0;
    const restrictedCount = summary.restricted || 0;
    const total = confidentialCount + restrictedCount;
    return (
      <div className="result-summary">
        <div className="summary-item warning">
          <div className="summary-count" style={{ color: '#FA8C16' }}>{restrictedCount}</div>
          <div className="summary-label">受限信息</div>
        </div>
        <div className="summary-item error">
          <div className="summary-count" style={{ color: '#EC4F4F' }}>{confidentialCount}</div>
          <div className="summary-label">机密信息</div>
        </div>
        <div className="summary-item">
          <div className="summary-count" style={{ color: '#1890ff' }}>{total}</div>
          <div className="summary-label">总计敏感信息</div>
        </div>
      </div>
    );
  };

  // 渲染流程进度 (已选文件的状态)
  const renderPipelineSteps = () => {
    if (!selectedFile) return null;
    const stepItems = [
      {
        title: '已保存',
        description: selectedFile,
        status: fileStatus.saved ? 'finish' : 'wait',
      },
      {
        title: '预处理',
        description: fileStatus.preprocessed ? '已完成 X2MD 转换' : '点击右上角"预处理"按钮',
        status: fileStatus.preprocessed ? 'finish' : (fileStatus.saved ? 'process' : 'wait'),
      },
      {
        title: '生成向量',
        description: fileStatus.embedded ? '向量已写入 JSON' : '点击右上角"生成向量"按钮',
        status: fileStatus.embedded ? 'finish' : (fileStatus.preprocessed ? 'process' : 'wait'),
      },
      {
        title: '比对',
        description: fileStatus.compared ? '已与三级库比对' : '点击右上角"比对"按钮',
        status: fileStatus.compared ? 'finish' : (fileStatus.embedded ? 'process' : 'wait'),
      },
    ];
    return (
      <div className="card" style={{ marginTop: '16px' }}>
        <div className="card-title">处理流程</div>
        <Steps size="small" items={stepItems} />
      </div>
    );
  };

  // 渲染最近一次操作结果
  const renderLastResult = () => {
    if (!lastResult) return null;
    const { type, data } = lastResult;

    if (type === 'preprocess') {
      return (
        <div className="card" style={{ marginTop: '16px' }}>
          <div className="card-title">预处理结果</div>
          <p style={{ color: data.success ? '#55A722' : '#EC4F4F' }}>
            {data.success ? '✓ X2MD 转换成功' : `✗ ${data.message}`}
          </p>
          {data.md_path && <p style={{ color: 'var(--font-secondary)', fontSize: '13px' }}>MD 文件: {data.md_path}</p>}
        </div>
      );
    }

    if (type === 'embed') {
      return (
        <div className="card" style={{ marginTop: '16px' }}>
          <div className="card-title">生成向量结果</div>
          <Space>
            <Tag color="blue">父块: {data.parents_embedded}</Tag>
            <Tag color="cyan">子块: {data.children_embedded}</Tag>
            <Tag color={data.abstract_embedded ? 'green' : 'default'}>
              摘要: {data.abstract_embedded ? '已嵌入' : '无摘要'}
            </Tag>
          </Space>
          <p style={{ color: 'var(--font-secondary)', fontSize: '13px', marginTop: '8px' }}>
            向量值已写入切片 JSON 文件，使用 /ARAG-begin/ 和 /ARAG-end/ 标识首尾
          </p>
        </div>
      );
    }

    if (type === 'compare') {
      const stats = data.stats || {};
      const skipReason = data.skip_reason;
      return (
        <div className="card" style={{ marginTop: '16px' }}>
          <div className="card-title">层级比对结果</div>
          <p style={{ color: 'var(--font-secondary)', fontSize: '13px', marginBottom: '12px' }}>
            策略: 摘要 → 父块 → 子块, 高度相似(≥80%)的父块跳过其子块比对
          </p>

          {skipReason === 'abstract' ? (
            <div style={{ padding: '12px', background: '#FFF1F0', borderRadius: '6px', marginBottom: '12px' }}>
              <Tag color="red">摘要超阈值 → 整篇文章标记为高度相似</Tag>
              <div style={{ marginTop: '8px' }}>
                {Object.entries(stats.abstract_max_sim || {}).map(([level, sim]) => (
                  <Tag key={level} style={{ margin: '2px' }}>
                    {getLevelLabel(level)}: {(sim * 100).toFixed(1)}%
                  </Tag>
                ))}
              </div>
            </div>
          ) : (
            <>
              <div className="result-summary">
                <div className="summary-item">
                  <div className="summary-count">{stats.parents_total || 0}</div>
                  <div className="summary-label">父块总数</div>
                </div>
                <div className="summary-item warning">
                  <div className="summary-count" style={{ color: '#FA8C16' }}>{stats.parents_skipped || 0}</div>
                  <div className="summary-label">父块跳过(高度相似)</div>
                </div>
                <div className="summary-item">
                  <div className="summary-count" style={{ color: '#1890ff' }}>{stats.children_compared || 0}</div>
                  <div className="summary-label">子块逐一比对</div>
                </div>
                <div className="summary-item warning">
                  <div className="summary-count" style={{ color: '#FA8C16' }}>{stats.children_skipped_by_parent || 0}</div>
                  <div className="summary-label">子块跳过(父块超阈值)</div>
                </div>
              </div>

              <div style={{ marginTop: '12px' }}>
                <strong>各密级最高相似度: </strong>
                {Object.entries(stats.level_max_sim || {}).map(([level, sim]) => (
                  <Tag key={level} color={getLevelColor(level)} style={{ margin: '2px' }}>
                    {getLevelLabel(level)}: {(sim * 100).toFixed(1)}%
                  </Tag>
                ))}
              </div>
            </>
          )}

          {/* 命中片段双栏对比 (按相似度降序, 过滤 < 60%) */}
          {Array.isArray(data.matches) && data.matches.length > 0 && (
            <div style={{ marginTop: '16px' }}>
              <div style={{ marginBottom: '8px' }}>
                <strong>命中片段对比</strong>
                <span style={{ color: 'var(--font-secondary)', fontSize: '12px', marginLeft: '8px' }}>
                  按相似度从高到低 · 仅显示 ≥ 60% · 共 {data.matches.length} 条
                </span>
              </div>
              <Table
                dataSource={data.matches.map((m, i) => ({ ...m, key: `${m.scanned_chunk_type}-${m.scanned_chunk_index}-${i}` }))}
                size="small"
                pagination={data.matches.length > 10 ? { pageSize: 10, showSizeChanger: false } : false}
                columns={[
                  {
                    title: '相似度',
                    dataIndex: 'similarity',
                    key: 'similarity',
                    width: 90,
                    render: (s) => {
                      const pct = (s * 100).toFixed(1);
                      const color = s >= 0.8 ? '#cf1322' : s >= 0.7 ? '#fa8c16' : '#1890ff';
                      return <Tag color={color}>{pct}%</Tag>;
                    },
                  },
                  {
                    title: '密级',
                    dataIndex: 'matched_level',
                    key: 'matched_level',
                    width: 90,
                    render: (lv) => <Tag color={getLevelColor(lv)}>{getLevelLabel(lv)}</Tag>,
                  },
                  {
                    title: '类型',
                    dataIndex: 'scanned_chunk_type',
                    key: 'scanned_chunk_type',
                    width: 70,
                    render: (t) => {
                      const label = t === 'abstract' ? '摘要' : t === 'parent' ? '父块' : '子块';
                      return <Tag>{label}</Tag>;
                    },
                  },
                  {
                    title: '原文档',
                    dataIndex: 'matched_doc',
                    key: 'matched_doc',
                    width: 160,
                    ellipsis: { showTitle: false },
                    render: (d) => (
                      <Tooltip placement="topLeft" title={d || '-'}>
                        {d || <span style={{ color: 'var(--font-secondary)' }}>-</span>}
                      </Tooltip>
                    ),
                  },
                  {
                    title: '被扫描内容',
                    dataIndex: 'scanned_text',
                    key: 'scanned_text',
                    ellipsis: { showTitle: false },
                    render: (t) => (
                      <Tooltip placement="topLeft" overlayStyle={{ maxWidth: 600 }} title={<div style={{ whiteSpace: 'pre-wrap' }}>{t}</div>}>
                        <span style={{ display: 'inline-block', maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {t || ''}
                        </span>
                      </Tooltip>
                    ),
                  },
                  {
                    title: '库中匹配内容',
                    dataIndex: 'matched_text',
                    key: 'matched_text',
                    ellipsis: { showTitle: false },
                    render: (t) => (
                      <Tooltip placement="topLeft" overlayStyle={{ maxWidth: 600 }} title={<div style={{ whiteSpace: 'pre-wrap' }}>{t}</div>}>
                        <span style={{ display: 'inline-block', maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {t || ''}
                        </span>
                      </Tooltip>
                    ),
                  },
                ]}
              />
            </div>
          )}

          <p style={{ color: 'var(--font-secondary)', fontSize: '12px', marginTop: '12px' }}>
            详细比对结果已写入切片 JSON 文件，紧跟在 /ARAG-end/ 标识后
          </p>
        </div>
      );
    }

    return null;
  };

  // ============ 文件扫描 Tab ============

  const fileScanTab = {
    key: 'file',
    label: <span><FileTextOutlined /> 文件扫描</span>,
    children: (
      <div>
        {/* 上传区 */}
        <Dragger
          name="file"
          multiple={false}
          showUploadList={false}
          customRequest={handleCustomUpload}
          accept=".pdf,.docx,.xlsx,.pptx,.txt,.html,.md"
        >
          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
          <p className="ant-upload-text">点击或拖拽文件到此区域</p>
          <p className="ant-upload-hint">
            文件将立即保存到 DocScan 目录，请使用下方按钮逐步处理
          </p>
        </Dragger>

        {/* 操作按钮 */}
        {selectedFile && (
          <div className="card" style={{ marginTop: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong>当前文件: </strong>
                <Tag color="blue">{selectedFile}</Tag>
              </div>
              <Space>
                <Button
                  icon={<FileSearchOutlined />}
                  onClick={handlePreprocess}
                  loading={opLoading.preprocess}
                  disabled={!fileStatus.saved}
                  type={!fileStatus.preprocessed && fileStatus.saved ? 'primary' : 'default'}
                >
                  预处理
                </Button>
                <Button
                  icon={<ApartmentOutlined />}
                  onClick={handleEmbed}
                  loading={opLoading.embed}
                  disabled={!fileStatus.preprocessed}
                  type={!fileStatus.embedded && fileStatus.preprocessed ? 'primary' : 'default'}
                >
                  生成向量
                </Button>
                <Button
                  icon={<ThunderboltOutlined />}
                  onClick={handleCompare}
                  loading={opLoading.compare}
                  disabled={!fileStatus.embedded}
                  type={!fileStatus.compared && fileStatus.embedded ? 'primary' : 'default'}
                >
                  比对
                </Button>
              </Space>
              {opLoading.embed && (
                <div style={{ marginTop: 12 }}>
                  <Progress percent={embedProgress} size="small" status="active" />
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
                    <span style={{ color: '#666', fontSize: 12 }}>{embedMessage}</span>
                    <Button size="small" danger icon={<StopOutlined />} onClick={handleStopEmbed}>
                      停止
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 流程进度 */}
        {renderPipelineSteps()}

        {/* 最近一次操作结果 */}
        {renderLastResult()}
      </div>
    ),
  };

  // ============ 文本扫描 Tab ============

  const textScanTab = {
    key: 'text',
    label: <span><EditOutlined /> 文本扫描</span>,
    children: (
      <div>
        <TextArea
          className="text-scan-area"
          placeholder="请输入需要扫描的文本内容（至少 10 个字符）..."
          value={textValue}
          onChange={(e) => setTextValue(e.target.value)}
          rows={8}
        />
        <div style={{ marginTop: '16px', textAlign: 'right' }}>
          <Button
            type="primary"
            onClick={handleTextScan}
            loading={textScanning}
            disabled={!textValue || textValue.trim().length < 10}
          >
            开始扫描
          </Button>
        </div>
        {textResult && (
          <div className="card" style={{ marginTop: '24px' }}>
            <div className="card-title">扫描结果</div>
            {renderResultSummary(textResult.summary)}
            <Tabs items={[{
              key: 'segments',
              label: '敏感信息列表',
              children: renderSegments(textResult.segments),
            }]} />
          </div>
        )}
      </div>
    ),
  };

  // ============ 文件管理 Tab ============

  const fileManageTab = {
    key: 'docscan',
    label: <span><FolderOutlined /> 文件管理</span>,
    children: (
      <div>
        <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ color: 'var(--font-secondary)', fontSize: '14px' }}>
            DocScan 目录 · 共 {docScanStats.total_files} 个文件
            (已预处理 {docScanStats.preprocessed} · 已生成向量 {docScanStats.embedded} · 已比对 {docScanStats.compared})
          </span>
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => { fetchDocScanFiles(); fetchDocScanStats(); }}
              loading={docScanLoading}
              size="small"
            >
              刷新
            </Button>
            <Popconfirm
              title="确定清空 DocScan 目录全部内容？"
              description="将删除所有源文件、切片、向量与图片, 不可恢复。"
              okText="清空"
              cancelText="取消"
              okButtonProps={{ danger: true }}
              onConfirm={handleClearAllDocScanFiles}
              disabled={docScanFiles.length === 0}
            >
              <Button
                danger
                icon={<DeleteOutlined />}
                loading={clearingAll}
                size="small"
                disabled={docScanFiles.length === 0}
              >
                全部删除
              </Button>
            </Popconfirm>
          </Space>
        </div>
        <Table
          dataSource={docScanFiles}
          rowKey="name"
          loading={docScanLoading}
          size="middle"
          pagination={{ pageSize: 10, showSizeChanger: false }}
          locale={{ emptyText: '暂无文件，上传后将自动保存到此目录' }}
          rowClassName={(record) => record.name === selectedFile ? 'ant-table-row-selected' : ''}
          onRow={(record) => ({
            onClick: () => handleSelectFile(record.name),
            style: { cursor: 'pointer' },
          })}
          columns={[
            {
              title: '文件名',
              dataIndex: 'name',
              key: 'name',
              ellipsis: true,
              render: (name) => (
                <span style={{ color: name === selectedFile ? 'var(--primary-color)' : 'var(--font-title)' }}>
                  <FileTextOutlined style={{ marginRight: '8px', color: 'var(--primary-color)' }} />
                  {name}
                </span>
              ),
            },
            { title: '大小', dataIndex: 'size_display', key: 'size_display', width: 120 },
            {
              title: '修改时间',
              dataIndex: 'modified',
              key: 'modified',
              width: 180,
              render: (ts) => new Date(ts * 1000).toLocaleString('zh-CN'),
            },
            {
              title: '操作',
              key: 'action',
              width: 120,
              render: (_, record) => (
                <Space>
                  <Button
                    type="link"
                    size="small"
                    onClick={(e) => { e.stopPropagation(); handleSelectFile(record.name); }}
                  >
                    选择
                  </Button>
                  <Popconfirm
                    title="确定删除此文件？"
                    onConfirm={(e) => { e?.stopPropagation?.(); handleDeleteDocScanFile(record.name); }}
                    onCancel={(e) => e?.stopPropagation?.()}
                  >
                    <Button
                      type="text"
                      danger
                      icon={<DeleteOutlined />}
                      size="small"
                      onClick={(e) => e.stopPropagation()}
                    />
                  </Popconfirm>
                </Space>
              ),
            },
          ]}
        />
      </div>
    ),
  };

  const items = [fileScanTab, textScanTab, fileManageTab];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">文档敏感信息扫描</h1>
        <p className="page-subtitle">
          上传文件后通过"预处理 → 生成向量 → 比对"三个按钮分步处理；或输入文本检测敏感信息
        </p>
      </div>
      <div className="card">
        <Tabs items={items} defaultActiveKey="file" />
      </div>
    </div>
  );
};

export default ScanPage;
