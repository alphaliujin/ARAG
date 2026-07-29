import React, { useState, useEffect } from 'react';
import { Button, message, Modal, Table, Space } from 'antd';
import { DatabaseOutlined, ReloadOutlined, DeleteOutlined, SyncOutlined } from '@ant-design/icons';
import { ingestDocuments, getDatabaseStats, resetDatabase } from '../utils/api';

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
  }, []);

  const handleImportAll = async () => {
    setImportingLevel('all');
    try {
      const result = await ingestDocuments();
      if (result.status === 'success' || result.status === 'completed') {
        message.success('文档导入成功');
        await loadStats();
      } else {
        message.warning('部分文档导入完成，请检查数据目录');
      }
    } catch (error) {
      const detail = error.response?.data?.detail || error.message || '未知错误';
      message.error('文档导入失败: ' + detail);
    } finally {
      setImportingLevel(null);
    }
  };

  const handleImportLevel = async (level) => {
    setImportingLevel(level);
    try {
      const result = await ingestDocuments(level);
      if (result.status === 'success' || result.status === 'completed') {
        message.success(`${getLevelName(level)}文档导入成功`);
        await loadStats();
      } else {
        message.warning(result.message || '导入遇到问题');
      }
    } catch (error) {
      const detail = error.response?.data?.detail || error.message || '未知错误';
      message.error('文档导入失败: ' + detail);
    } finally {
      setImportingLevel(null);
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
          disabled={importingLevel !== null && importingLevel !== record.key}
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
              loading={importingLevel === 'all'}
              disabled={importingLevel !== null}
            >
              导入全部文档
            </Button>
            <Button
              danger
              icon={<DeleteOutlined />}
              onClick={handleReset}
              loading={resetting}
            >
              重置数据库
            </Button>
          </Space>
        </div>
      </div>

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
