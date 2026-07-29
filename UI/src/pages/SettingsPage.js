import React, { useState, useEffect, useRef } from 'react';
import {
  Form,
  Input,
  InputNumber,
  Slider,
  Switch,
  Button,
  Space,
  message,
  Tabs,
  Select,
  Divider,
  Tag,
  Spin,
} from 'antd';
import { apiClient } from '../utils/api';
import {
  SettingOutlined,
  SaveOutlined,
  ReloadOutlined,
  DatabaseOutlined,
  RobotOutlined,
  FileTextOutlined,
} from '@ant-design/icons';

const { Option } = Select;


const SettingsPage = () => {
  const [activeTab, setActiveTab] = useState('general');
  const [loading, setLoading] = useState(true);

  // 为每个 tab 独立 form 实例，避免共享 form 实例导致字段串扰
  const [generalForm] = Form.useForm();
  const [modelForm] = Form.useForm();
  const [preprocessForm] = Form.useForm();
  const [databaseForm] = Form.useForm();

  // 每个类别的 saving 状态独立
  const [saving, setSaving] = useState({
    general: false,
    model: false,
    preprocess: false,
    database: false,
  });

  const formMap = {
    general: generalForm,
    model: modelForm,
    preprocess: preprocessForm,
    database: databaseForm,
  };

  // 记录是否已经做过首次加载;之后不再覆盖用户的编辑
  const initialLoadedRef = useRef(false);

  // 加载所有设置
  const loadSettings = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get(`/settings`);
      const data = res.data || {};
      // 只在首次加载时全量 setFieldsValue;后续 reload 仅填充用户尚未编辑(empty/dirty=false)的字段。
      // 旧实现总是无条件 setFieldsValue,慢网络下用户已开始编辑会被覆盖。
      const isFirstLoad = !initialLoadedRef.current;
      const safeSetFields = (form, serverValues) => {
        if (!serverValues) return;
        if (isFirstLoad) {
          form.setFieldsValue(serverValues);
          return;
        }
        // 仅填充当前 form 中为 undefined 的字段,避免覆盖 dirty 编辑
        const current = form.getFieldsValue(true);
        const patch = {};
        for (const k of Object.keys(serverValues)) {
          if (current[k] === undefined || current[k] === null || current[k] === '') {
            patch[k] = serverValues[k];
          }
        }
        if (Object.keys(patch).length > 0) form.setFieldsValue(patch);
      };
      safeSetFields(generalForm, data.general);
      safeSetFields(modelForm, data.model);
      safeSetFields(preprocessForm, data.preprocess);
      safeSetFields(databaseForm, data.database);
      initialLoadedRef.current = true;
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      message.error(`加载设置失败: ${detail}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 保存指定类别的设置
  const handleSave = async (category, values) => {
    setSaving((prev) => ({ ...prev, [category]: true }));
    try {
      const res = await apiClient.put(`/settings/${category}`, { values });
      if (res.data && res.data.status === 'success') {
        message.success('设置保存成功');
        // 用后端返回值刷新表单（确保 UI 与持久化值同步）
        if (res.data.values) {
          formMap[category].setFieldsValue(res.data.values);
        }
      } else {
        message.warning('设置可能未完全保存');
      }
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      message.error(`保存设置失败: ${detail}`);
    } finally {
      setSaving((prev) => ({ ...prev, [category]: false }));
    }
  };

  // 重置指定类别为默认值
  const handleReset = async (category) => {
    setSaving((prev) => ({ ...prev, [category]: true }));
    try {
      const res = await apiClient.post(`/settings/${category}/reset`);
      if (res.data && res.data.status === 'success' && res.data.values) {
        formMap[category].setFieldsValue(res.data.values);
        message.success('设置已重置为默认值');
      }
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      message.error(`重置失败: ${detail}`);
    } finally {
      setSaving((prev) => ({ ...prev, [category]: false }));
    }
  };

  // 渲染保存/重置按钮组
  const renderFormActions = (category) => (
    <Form.Item>
      <Space>
        <Button
          type="primary"
          htmlType="submit"
          loading={saving[category]}
          icon={<SaveOutlined />}
        >
          保存设置
        </Button>
        <Button
          onClick={() => handleReset(category)}
          loading={saving[category]}
          icon={<ReloadOutlined />}
        >
          重置为默认值
        </Button>
      </Space>
    </Form.Item>
  );

  // 使用 antd v5 推荐的 items 写法
  const tabItems = [
    {
      key: 'general',
      label: (
        <span>
          <SettingOutlined />
          通用设置
        </span>
      ),
      children: (
        <div className="card">
          <div className="card-title">通用参数</div>
          <Form
            form={generalForm}
            layout="vertical"
            onFinish={(values) => handleSave('general', values)}
          >
            <Form.Item
              label="机密信息相似度阈值"
              name="similarityThresholdConfidential"
              help="文本与机密文档的相似度达到此值时标记为机密信息"
            >
              <Slider
                min={0}
                max={1}
                step={0.05}
                marks={{ 0: '0%', 0.5: '50%', 1: '100%' }}
              />
            </Form.Item>

            <Form.Item
              label="受限信息相似度阈值"
              name="similarityThresholdRestricted"
              help="文本与受限文档的相似度达到此值时标记为受限信息"
            >
              <Slider
                min={0}
                max={1}
                step={0.05}
                marks={{ 0: '0%', 0.5: '50%', 1: '100%' }}
              />
            </Form.Item>

            <Form.Item label="最大上传文件大小 (MB)" name="maxUploadSize">
              <InputNumber min={1} max={200} style={{ width: '100%' }} />
            </Form.Item>

            <Form.Item label="启用日志记录" name="enableLogging" valuePropName="checked">
              <Switch />
            </Form.Item>

            <Form.Item label="日志级别" name="logLevel">
              <Select>
                <Option value="debug">Debug</Option>
                <Option value="info">Info</Option>
                <Option value="warning">Warning</Option>
                <Option value="error">Error</Option>
              </Select>
            </Form.Item>

            {renderFormActions('general')}
          </Form>
        </div>
      ),
    },
    {
      key: 'model',
      label: (
        <span>
          <RobotOutlined />
          模型设置
        </span>
      ),
      children: (
        <div className="card">
          <div className="card-title">嵌入模型配置</div>
          <Form
            form={modelForm}
            layout="vertical"
            onFinish={(values) => handleSave('model', values)}
          >
            <Form.Item label="默认嵌入模型" name="embeddingModel">
              <Select>
                <Option value="ollama-bge-m3">Ollama (bge-m3, 1024维)</Option>
                <Option value="chromadb-default">ChromaDB 默认 (all-MiniLM-L6-v2, 384维)</Option>
                <Option value="sentence-transformers">Sentence-Transformers (384维)</Option>
              </Select>
            </Form.Item>

            <Divider />

            <div style={{ marginBottom: '16px' }}>
              <Tag color="blue">Ollama 配置</Tag>
            </div>

            <Form.Item label="Ollama 服务地址" name="ollamaUrl">
              <Input placeholder="http://localhost:11434" />
            </Form.Item>

            <Form.Item label="Ollama 模型" name="ollamaModel">
              <Input placeholder="bge-m3:latest" />
            </Form.Item>

            <Divider />

            <div style={{ marginBottom: '16px' }}>
              <Tag color="green">Sentence-Transformers 配置</Tag>
            </div>

            <Form.Item label="模型名称" name="stModel">
              <Input placeholder="all-MiniLM-L6-v2" />
            </Form.Item>

            <Form.Item label="批量大小" name="batchSize">
              <InputNumber min={1} max={128} style={{ width: '100%' }} />
            </Form.Item>

            {renderFormActions('model')}
          </Form>
        </div>
      ),
    },
    {
      key: 'preprocess',
      label: (
        <span>
          <FileTextOutlined />
          预处理设置
        </span>
      ),
      children: (
        <div className="card">
          <div className="card-title">X2MD 预处理配置</div>
          <Form
            form={preprocessForm}
            layout="vertical"
            onFinish={(values) => handleSave('preprocess', values)}
          >
            <Form.Item label="默认切片大小" name="chunkSize" help="每个切片的最大字符数">
              <InputNumber min={100} max={5000} style={{ width: '100%' }} />
            </Form.Item>

            <Form.Item label="切片重叠" name="chunkOverlap" help="相邻切片之间的重叠字符数">
              <InputNumber min={0} max={500} style={{ width: '100%' }} />
            </Form.Item>

            <Form.Item label="启用 OCR" name="enableOCR" valuePropName="checked">
              <Switch />
            </Form.Item>

            <Form.Item label="OCR 语言" name="ocrLang">
              <Select>
                <Option value="chi_sim+eng">中文+英文</Option>
                <Option value="chi_sim">中文</Option>
                <Option value="eng">英文</Option>
              </Select>
            </Form.Item>

            <Form.Item label="提取表格" name="extractTables" valuePropName="checked">
              <Switch />
            </Form.Item>

            <Form.Item
              label="启用 LLM 增强"
              name="enableLLM"
              valuePropName="checked"
              help="使用大模型生成摘要和优化内容"
            >
              <Switch />
            </Form.Item>

            {renderFormActions('preprocess')}
          </Form>
        </div>
      ),
    },
    {
      key: 'database',
      label: (
        <span>
          <DatabaseOutlined />
          数据库设置
        </span>
      ),
      children: (
        <div className="card">
          <div className="card-title">向量数据库配置</div>
          <Form
            form={databaseForm}
            layout="vertical"
            onFinish={(values) => handleSave('database', values)}
          >
            <Form.Item label="向量数据库目录" name="vectorDbDir">
              <Input />
            </Form.Item>

            <Form.Item label="集合前缀" name="collectionPrefix">
              <Input />
            </Form.Item>

            <Form.Item label="匿名化遥测" name="anonymizedTelemetry" valuePropName="checked">
              <Switch />
            </Form.Item>

            {renderFormActions('database')}
          </Form>
        </div>
      ),
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">系统设置</h1>
        <p className="page-subtitle">配置系统参数、模型连接和预处理选项</p>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 80 }}>
          <Spin size="large" tip="加载设置中..." />
        </div>
      ) : (
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          type="card"
          items={tabItems}
        />
      )}
    </div>
  );
};

export default SettingsPage;
