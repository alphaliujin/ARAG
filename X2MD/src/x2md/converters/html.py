from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import urllib.parse
import urllib.request as _ureq
from pathlib import Path

from bs4 import BeautifulSoup

from x2md.converters.base import BaseConverter
from x2md.utils import (
    clean_markdown,
    get_image_output_dir,
    image_marker,
    split_contact_info,
    strip_toc_markers,
    table_to_lines,
)


# --------------------------------------------------------------------------
# href 协议白名单: 阻止 javascript:/data:/vbscript: 等 XSS scheme 注入到
# Markdown 输出, 同时放行文档站最常见的相对路径链接 (chapter2.html 等)。
# 用"是否含 scheme"判断而非"是否以 http:// 开头", 故相对路径不被误杀;
# scheme 大小写不敏感 (HTTPS:// 合法)。
# --------------------------------------------------------------------------

_ALLOWED_HREF_SCHEMES = ("http", "https", "mailto", "ftp")
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):")


def _is_safe_href(href: str) -> bool:
    """href 是否可安全写入 Markdown 链接 (True=可写, False=丢弃 href 仅留文本)."""
    if not href:
        return False
    m = _SCHEME_RE.match(href)
    if m is None:
        return True  # 无 scheme = 相对路径, 不可能携带危险 scheme
    return m.group(1).lower() in _ALLOWED_HREF_SCHEMES


# --------------------------------------------------------------------------
# SSRF 防护: 抓取 <img src=http(s)://...> 时, 解析+校验+固定 IP 后直连,
# 杜绝 DNS rebinding TOCTOU (校验用的 getaddrinfo 与 urllib 内部二次解析之间,
# 攻击者可借短 TTL DNS 把域名翻到内网 IP)。也顺带覆盖 302 跳内网的向量:
# 内网目标在 _resolve_safe_ip 即抛错, 连接被拒绝, 图片跳过。
# --------------------------------------------------------------------------

def _resolve_safe_ip(hostname: str) -> str:
    """解析 hostname, 返回一个非内网 IP (字符串) 用于固定连接; 不安全则抛 RuntimeError.

    返回的 IP 由调用方直接建连, 不再二次 DNS, 故无 rebinding 窗口。
    """
    if not hostname:
        raise RuntimeError("no hostname")
    # 字面 IP: 直接校验
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise RuntimeError(f"blocked internal IP {hostname}")
        return str(ip)
    except ValueError:
        pass
    # 域名: 解析后挑一个公网 IP (跳过内网地址)
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except OSError as e:
        raise RuntimeError(f"resolve failed: {e}")
    for _f, _t, _p, _c, sockaddr in addrs:
        try:
            r = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if r.is_private or r.is_loopback or r.is_link_local or r.is_reserved or r.is_multicast:
            continue
        return sockaddr[0]
    raise RuntimeError(f"no safe IP for {hostname}")


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """连接到 pinned_ip; Host 头由 handler 设为原域名 (避免按 IP 寻址到错误 vhost)。"""

    def __init__(self, pinned_ip, port, timeout, original_host):
        super().__init__(pinned_ip, port if port else 80, timeout=timeout)
        self._original_host = original_host


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """连接到 pinned_ip, 但 TLS SNI/证书校验仍用原域名 (按 IP 校验会失败)。"""

    def __init__(self, pinned_ip, port, timeout, original_host):
        super().__init__(pinned_ip, port if port else 443, timeout=timeout)
        self._original_host = original_host

    def connect(self):
        # 镜像 http.client.HTTPSConnection.connect, 但 server_hostname 用原域名,
        # 而非 self.host(=pinned_ip)。先走 HTTPConnection.connect 建立 TCP 到 pinned_ip。
        http.client.HTTPConnection.connect(self)
        if self._tunnel_host:
            server_hostname = self._tunnel_host
        else:
            server_hostname = self._original_host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


def _pinning_open(handler, req, conn_class):
    """解析+校验+固定 IP 后交给 do_open; Host 头设为原域名 (触发 skip_host)。"""
    u = urllib.parse.urlparse(req.full_url)
    host = u.hostname
    port = u.port
    pinned_ip = _resolve_safe_ip(host)  # 不安全则 raise -> 上层 except 捕获, 图片跳过
    req.add_unredirected_header("Host", host if port is None else f"{host}:{port}")
    return handler.do_open(
        lambda h, timeout=None, **kw: conn_class(pinned_ip, port, timeout, host),
        req,
    )


class _PinningHTTPHandler(_ureq.HTTPHandler):
    """接管 http:// 连接: 每个请求(含重定向目标)解析+校验+固定 IP。"""

    def http_open(self, req):
        return _pinning_open(self, req, _PinnedHTTPConnection)


class _PinningHTTPSHandler(_ureq.HTTPSHandler):
    """接管 https:// 连接: 解析+校验+固定 IP, TLS 仍按原域名校验证书。"""

    def https_open(self, req):
        return _pinning_open(self, req, _PinnedHTTPSConnection)


class HtmlConverter(BaseConverter):
    extensions = [".html", ".htm"]

    HEADING_TAGS = {
        "h1": "#", "h2": "##", "h3": "###",
        "h4": "####", "h5": "#####", "h6": "######",
    }

    def convert(self, file_path: Path, **kwargs) -> str:
        encoding: str = kwargs.get("encoding", "utf-8")
        md_output_path: Path | None = kwargs.get("md_output_path")

        self._image_dir = get_image_output_dir(md_output_path, file_path)
        self._image_counter = 0
        self._source_stem = file_path.stem

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        # 只 strip `<body>` 的直接子级的 <header>/<footer>/<nav> (站点级框架),
        # 保留文章内 <article><header> (标题/日期/作者) 等语义标签。
        body = soup.find("body")
        site_level_tags = ("nav",)
        if body:
            for child in list(body.children):
                if hasattr(child, "name") and child.name in ("header", "footer"):
                    child.decompose()
        # 非 body 子级的 <header>/<footer> 仍剥离 (如无关的全局 header)
        for tag in soup(["script", "style", "noscript", "nav"]):
            tag.decompose()

        parts: list[str] = []
        self._walk(soup, parts)

        result = "\n\n".join(parts)
        result = strip_toc_markers(result)
        result = split_contact_info(result)
        return clean_markdown(result)

    def _walk(self, element, parts: list[str]):
        if isinstance(element, str):
            text = element.strip()
            if text:
                parts.append(text)
            return

        tag_name = element.name

        if tag_name in self.HEADING_TAGS:
            prefix = self.HEADING_TAGS[tag_name]
            text = element.get_text(strip=True)
            if text:
                parts.append(f"{prefix} {text}")
            return

        if tag_name == "p":
            # 用 _inline_walk 递归收集内联格式 (bold/italic/links/code),
            # 而非 get_text() -- 后者把 <strong>/<a>/<em> 全部展平为纯文本,
            # 丢失加粗/斜体/链接, 且内联 <a> 永不触发 href 白名单 (XSS 风险)。
            text = "".join(self._inline_walk(c) for c in element.children).strip()
            if text:
                parts.append(text)
            return

        if tag_name == "li":
            text = "".join(self._inline_walk(c) for c in element.children).strip()
            if text:
                parts.append(f"- {text}")
            return

        if tag_name == "a":
            href = element.get("href", "")
            text = element.get_text(strip=True)
            if text and href:
                # 协议白名单: 阻止 javascript:/data: 等 XSS 注入到 Markdown 输出;
                # 相对路径 (无 scheme) 放行。不安全协议丢弃 href, 仅保留链接文本。
                if _is_safe_href(href):
                    parts.append(f"[{text}]({href})")
                else:
                    parts.append(text)
            elif text:
                parts.append(text)
            return

        if tag_name == "img":
            src = element.get("src", "")
            alt = element.get("alt", "")
            saved_name = self._save_image(src)
            if saved_name:
                parts.append(image_marker(saved_name))
            else:
                parts.append(f"![{alt}]({src})")
            return

        if tag_name == "br":
            parts.append("\n")
            return

        if tag_name == "strong" or tag_name == "b":
            text = element.get_text(strip=True)
            if text:
                parts.append(f"**{text}**")
            return

        if tag_name == "em" or tag_name == "i":
            text = element.get_text(strip=True)
            if text:
                parts.append(f"*{text}*")
            return

        if tag_name == "code":
            text = element.get_text(strip=True)
            if text:
                parts.append(f"`{text}`")
            return

        if tag_name == "pre":
            text = element.get_text()
            parts.append(f"```\n{text}\n```")
            return

        if tag_name == "table":
            self._convert_table(element, parts)
            return

        for child in element.children:
            self._walk(child, parts)

    def _inline_walk(self, element) -> str:
        """递归收集内联元素为带 Markdown 格式的字符串。

        供 p/li 等容器使用, 替代 get_text() (后者展平所有子标签, 丢失
        bold/italic/links/code)。内联 <a> 在此走 href 协议白名单。
        """
        # NavigableString (bs4 文本节点) 是 str 子类
        if isinstance(element, str):
            return str(element)
        tag = element.name
        if tag in ("strong", "b"):
            inner = "".join(self._inline_walk(c) for c in element.children)
            return f"**{inner}**" if inner.strip() else ""
        if tag in ("em", "i"):
            inner = "".join(self._inline_walk(c) for c in element.children)
            return f"*{inner}*" if inner.strip() else ""
        if tag == "code":
            inner = element.get_text()
            return f"`{inner}`" if inner else ""
        if tag == "a":
            href = element.get("href", "")
            text = "".join(self._inline_walk(c) for c in element.children).strip()
            if text and href:
                # 协议白名单: 与 _walk 中 <a> 处理一致, 阻止 javascript:/data: 等;
                # 相对路径 (无 scheme) 放行。
                if _is_safe_href(href):
                    return f"[{text}]({href})"
                return text  # 不安全协议仅保留链接文本
            return text
        if tag == "br":
            return "\n"
        # 其他内联标签 (span/sub/sup/u 等): 递归子节点
        return "".join(self._inline_walk(c) for c in element.children)

    def _convert_table(self, table, parts: list[str]):
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            rows.append([cell.get_text(strip=True) for cell in cells])
        if rows:
            md = table_to_lines(rows)
            if md:
                parts.append(md)

    def _save_image(self, src: str) -> str | None:
        if not src:
            return None

        self._image_counter += 1
        ext = ".png"

        if src.startswith("data:"):
            try:
                import base64
                header, data = src.split(",", 1)
                if "image/png" in header:
                    ext = ".png"
                elif "image/jpeg" in header or "image/jpg" in header:
                    ext = ".jpg"
                elif "image/webp" in header:
                    ext = ".webp"
                filename = f"{self._source_stem}_{self._image_counter:03d}{ext}"
                self._image_dir.mkdir(parents=True, exist_ok=True)
                filepath = self._image_dir / filename
                filepath.write_bytes(base64.b64decode(data))
                return filename
            except Exception:
                return None

        if src.startswith(("http://", "https://")):
            # SSRF 防护: 拒绝私有网络地址和可疑 URL
            import ipaddress
            import socket
            parsed = urllib.parse.urlparse(src)
            hostname = parsed.hostname
            if not hostname:
                return None
            # 1) hostname 是字面 IP → 直接判定
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                    return None
            except ValueError:
                # 2) hostname 是域名 → DNS 解析后逐个 IP 校验,防止 evil.com → 169.254.169.254
                try:
                    addrs = socket.getaddrinfo(hostname, None)
                except OSError:
                    return None  # 解析失败直接拒绝
                for family, _type, _proto, _canon, sockaddr in addrs:
                    addr_str = sockaddr[0]
                    try:
                        resolved = ipaddress.ip_address(addr_str)
                    except ValueError:
                        return None
                    if (
                        resolved.is_private
                        or resolved.is_loopback
                        or resolved.is_link_local
                        or resolved.is_reserved
                        or resolved.is_multicast
                    ):
                        return None
            # 拒绝常见内网域名
            blocked_hostnames = ("localhost", "local", "internal", "intranet", "127.0.0.1")
            if hostname.lower() in blocked_hostnames or hostname.lower().endswith(".local") or hostname.lower().endswith(".internal"):
                return None
            # 后缀白名单: 只允许常见图片格式,避免 .svg / .html / .php 落盘
            url_suffix = Path(parsed.path).suffix.lower()
            allowed_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
            ext = url_suffix if url_suffix in allowed_suffixes else ".png"
            filename = f"{self._source_stem}_{self._image_counter:03d}{ext}"
            self._image_dir.mkdir(parents=True, exist_ok=True)
            filepath = self._image_dir / filename
            try:
                import urllib.request as _ureq

                # SSRF: 默认 urlopen 跟随 3xx 重定向且不重新校验目标, 攻击者可用
                # 302 跳到 169.254.169.254 云元数据服务。自定义 redirect handler 对
                # 每个重定向目标重跑内网校验, 拒绝则不跟随 (urlopen 抛 HTTPError)。
                # 注: ipaddress/socket 已在本方法上方 import, 这里直接复用。
                def _redirect_host_is_safe(url: str) -> bool:
                    h = urllib.parse.urlparse(url).hostname
                    if not h:
                        return False
                    try:
                        ip = ipaddress.ip_address(h)
                        return not (ip.is_private or ip.is_loopback or ip.is_link_local
                                    or ip.is_reserved or ip.is_multicast)
                    except ValueError:
                        try:
                            addrs = socket.getaddrinfo(h, None)
                        except OSError:
                            return False
                        for _f, _t, _p, _c, sockaddr in addrs:
                            try:
                                r = ipaddress.ip_address(sockaddr[0])
                            except ValueError:
                                return False
                            if r.is_private or r.is_loopback or r.is_link_local or r.is_reserved or r.is_multicast:
                                return False
                        return True

                class _SafeRedirectHandler(_ureq.HTTPRedirectHandler):
                    def redirect_request(self, req, fp, code, msg, headers, newurl):
                        if not _redirect_host_is_safe(newurl):
                            return None
                        return super().redirect_request(req, fp, code, msg, headers, newurl)

                # _PinningHTTPHandler/_PinningHTTPSHandler: 每个连接(含重定向目标)
                # 解析+校验+固定 IP 后直连, 杜绝 DNS rebinding TOCTOU, 也阻止 302 跳内网。
                # _SafeRedirectHandler 留作纵深防御 (内网重定向目标在连接前即拒绝)。
                opener = _ureq.build_opener(_PinningHTTPHandler(), _PinningHTTPSHandler(), _SafeRedirectHandler())
                req = _ureq.Request(src, headers={"User-Agent": "Mozilla/5.0"})
                with opener.open(req, timeout=30) as response:
                    # 检查内容大小（限制 10MB）
                    content_length = response.getheader('Content-Length')
                    if content_length and int(content_length) > 10 * 1024 * 1024:
                        return None
                    data = response.read(10 * 1024 * 1024)  # 最多读取 10MB
                    if len(data) >= 10 * 1024 * 1024:
                        return None
                    filepath.write_bytes(data)
                return filename
            except Exception:
                return None

        return None
