import React, { Suspense, useState, useEffect, useCallback } from 'react';
import TopNav from './components/TopNav';
import { Spin, Result, Button } from 'antd';
import './styles/App.css';

// 路由级代码分割：每个页面独立 chunk，首屏只需下载当前页
const PreprocessPage = React.lazy(() => import('./pages/PreprocessPage'));
const IngestPage = React.lazy(() => import('./pages/IngestPage'));
const MarkPage = React.lazy(() => import('./pages/MarkPage'));
const ScanPage = React.lazy(() => import('./pages/ScanPage'));
const DatabasePage = React.lazy(() => import('./pages/DatabasePage'));
const SettingsPage = React.lazy(() => import('./pages/SettingsPage'));

const PAGES = {
  preprocess: PreprocessPage,
  ingest: IngestPage,
  mark: MarkPage,
  scan: ScanPage,
  database: DatabasePage,
  settings: SettingsPage,
};
const VALID_KEYS = Object.keys(PAGES);
const DEFAULT_KEY = 'preprocess';

const PageFallback = () => (
  <div style={{ display: 'flex', justifyContent: 'center', padding: 80 }}>
    <Spin size="large" tip="加载中..." />
  </div>
);

// Hash 路由 — 不引入新依赖,通过 window.location.hash 反映当前页签:
//   优势:刷新仍在原页面、URL 可分享、浏览器 back/forward 工作。
//   #/preprocess, #/ingest, ...
const readHashKey = () => {
  const h = (window.location.hash || '').replace(/^#\/?/, '').trim();
  return h || DEFAULT_KEY;
};

function App() {
  const [activeMenu, setActiveMenu] = useState(readHashKey);

  // hashchange: 浏览器 back/forward 或外部脚本改 hash 时同步 state
  useEffect(() => {
    const onHash = () => setActiveMenu(readHashKey());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  // 用户点 TopNav 时写入 hash,触发上面 onHash 同步 state(单一事实源)
  const handleMenuChange = useCallback((key) => {
    if (window.location.hash !== `#/${key}`) {
      window.location.hash = `/${key}`;
    } else {
      // 点同一个 menu 也强制刷一次 state(防止 fallback 卡住)
      setActiveMenu(key);
    }
  }, []);

  const renderPage = () => {
    const isValidKey = VALID_KEYS.includes(activeMenu);
    if (!isValidKey) {
      // 显式 404 提示而非静默 fallback,便于发现脏 hash / 链接错误
      return (
        <Result
          status="404"
          title="页面不存在"
          subTitle={`未知页签: "${activeMenu}"`}
          extra={
            <Button type="primary" onClick={() => handleMenuChange(DEFAULT_KEY)}>
              返回首页
            </Button>
          }
        />
      );
    }
    const PageComponent = PAGES[activeMenu];
    return (
      <Suspense fallback={<PageFallback />}>
        <PageComponent />
      </Suspense>
    );
  };

  return (
    <div className="app-container">
      <TopNav activeMenu={activeMenu} onMenuChange={handleMenuChange} />
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  );
}

export default App;
