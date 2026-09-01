import React, { useState } from 'react';
import {
  FileTextOutlined,
  DatabaseOutlined,
  TagsOutlined,
  ScanOutlined,
  AppstoreOutlined,
  SettingOutlined,
} from '@ant-design/icons';

const TopNav = ({ activeMenu, onMenuChange }) => {
  const menuItems = [
    { key: 'preprocess', label: '文件预处理', icon: <FileTextOutlined /> },
    { key: 'ingest', label: '数据入库', icon: <DatabaseOutlined /> },
    { key: 'mark', label: '数据优化', icon: <TagsOutlined /> },
    { key: 'scan', label: '文档扫描', icon: <ScanOutlined /> },
    { key: 'database', label: '数据库管理', icon: <AppstoreOutlined /> },
    { key: 'settings', label: '系统设置', icon: <SettingOutlined /> },
  ];

  return (
    <nav className="top-nav">
      <div className="nav-logo">
        <span style={{ color: '#fff', fontSize: '20px', marginRight: '8px' }}>🛡️</span>
        <span className="nav-title">敏感信息识别系统</span>
      </div>

      <div className="nav-menu">
        {menuItems.map((item) => (
          <button
            key={item.key}
            className={`nav-menu-item ${activeMenu === item.key ? 'active' : ''}`}
            onClick={() => onMenuChange(item.key)}
          >
            {item.icon}
            <span style={{ marginLeft: '8px' }}>{item.label}</span>
          </button>
        ))}
      </div>

      <div className="nav-toolbar">
        <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '12px' }}>v0.2.7</span>
      </div>
    </nav>
  );
};

export default TopNav;
