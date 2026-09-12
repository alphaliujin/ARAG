#!/bin/bash

# =============================================================================
# 敏感信息识别系统 - 一键启动脚本
# =============================================================================
# ★ 本文件仅用于 macOS/本地【开发环境】双进程 (backend:8000 + frontend:3000)。
#   Linux 生产部署逻辑在 deploy/start.sh (单进程同源), 打包由
#   scripts/make_deploy_package.sh 自动从 deploy/ 拷贝, 与根目录此文件无关。
#   要改生产部署行为, 请改 deploy/start.sh, 不要改这里 (两边互不同步)。
# =============================================================================
# 用法: ./start.sh [frontend|backend|all]
#   ./start.sh        - 启动所有服务 (默认)
#   ./start.sh front  - 仅启动前端
#   ./start.sh back   - 仅启动后端
#   ./start.sh status - 查看服务状态
#   ./start.sh stop   - 停止所有服务
# =============================================================================

# 严格模式: 未定义变量 / pipe 中任一阶段失败 都视为错误
# 注意: 不开 -e, 因为脚本里有大量 || true 容错; 但 pipefail + nounset 仍能捕到大多数 typo
set -uo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 项目路径
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/UI"

# 端口配置
BACKEND_PORT=8000
FRONTEND_PORT=3000

# PID 文件
BACKEND_PID_FILE="$PROJECT_DIR/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_DIR/.frontend.pid"

# =============================================================================
# 工具函数
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           敏感信息识别系统 - 一键启动脚本                       ║"
    echo "║           Sensitive Information Detection System              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 检查端口是否被占用
# -n -P 跳过 DNS/端口名解析,避免 macOS 上 lsof 慢查询
check_port() {
    local port=$1
    lsof -nP -i :"$port" -sTCP:LISTEN 2>/dev/null | grep -v COMMAND | awk '{print $2}' | head -1
}

# 等待服务启动
wait_for_service() {
    local url=$1
    local max_attempts=${2:-30}
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -n "."
    done
    return 1
}

# =============================================================================
# 进程清理函数 - 增强版
# =============================================================================

# 强制终止进程（带重试）
kill_process_force() {
    local pid=$1
    local name=$2
    local max_attempts=5
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        # 检查进程是否还存在
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0  # 进程已退出
        fi

        # 先尝试优雅终止
        kill "$pid" 2>/dev/null || true
        sleep 0.5

        # 检查是否还在
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi

        # 强制终止
        kill -9 "$pid" 2>/dev/null || true
        sleep 0.5

        attempt=$((attempt + 1))
    done

    # 最终检查
    if kill -0 "$pid" 2>/dev/null; then
        log_error "无法终止进程 $name (PID: $pid)，请手动检查"
        return 1
    fi
    return 0
}

# 清理后端相关进程
cleanup_backend_processes() {
    log_step "清理后端相关进程..."
    local count=0

    # 1. 通过端口查找并终止
    local port_pid
    port_pid=$(check_port "$BACKEND_PORT")
    if [ -n "$port_pid" ]; then
        log_warn "发现占用端口 $BACKEND_PORT 的进程 (PID: $port_pid)"
        if kill_process_force "$port_pid" "后端端口进程"; then
            log_info "已终止端口进程 (PID: $port_pid)"
            count=$((count + 1))
        fi
    fi

    # 2. 通过 PID 文件终止
    if [ -f "$BACKEND_PID_FILE" ]; then
        local file_pid
        file_pid=$(cat "$BACKEND_PID_FILE" 2>/dev/null)
        if [ -n "$file_pid" ] && kill -0 "$file_pid" 2>/dev/null; then
            log_warn "发现 PID 文件中的后端进程 (PID: $file_pid)"
            if kill_process_force "$file_pid" "后端PID文件进程"; then
                log_info "已终止 PID 文件进程 (PID: $file_pid)"
                count=$((count + 1))
            fi
        fi
        rm -f "$BACKEND_PID_FILE"
    fi

    # 3. 查找并终止 uvicorn 进程（在项目目录下运行的）
    # 不再用 lsof -p PID 查 cwd (macOS 上极慢,易卡死);
    # 改为直接在 ps 命令行里匹配项目路径,效果等价且毫秒级返回
    local uvicorn_pids
    uvicorn_pids=$(ps -eo pid,command | grep -E "uvicorn.*app\.main:app" | grep -v grep | grep -F "$PROJECT_DIR" | awk '{print $1}')
    if [ -n "$uvicorn_pids" ]; then
        for uv_pid in $uvicorn_pids; do
            log_warn "发现遗留的 uvicorn 进程 (PID: $uv_pid)"
            if kill_process_force "$uv_pid" "uvicorn进程"; then
                log_info "已终止 uvicorn 进程 (PID: $uv_pid)"
                count=$((count + 1))
            fi
        done
    fi

    # 4. 查找并终止 python 进程（在项目目录下运行的 uvicorn）
    local python_pids
    python_pids=$(ps -eo pid,command | grep -E "python.*uvicorn" | grep -v grep | grep -F "$PROJECT_DIR" | awk '{print $1}')
    if [ -n "$python_pids" ]; then
        for py_pid in $python_pids; do
            log_warn "发现遗留的 python+uvicorn 进程 (PID: $py_pid)"
            if kill_process_force "$py_pid" "python+uvicorn进程"; then
                log_info "已终止 python+uvicorn 进程 (PID: $py_pid)"
                count=$((count + 1))
            fi
        done
    fi

    # 5. Zombie processes are automatically reaped when their parent exits,
    #    so no explicit cleanup is needed here.

    # 等待端口释放
    local wait_count=0
    while [ -n "$(check_port "$BACKEND_PORT")" ] && [ $wait_count -lt 10 ]; do
        sleep 0.5
        wait_count=$((wait_count + 1))
    done

    if [ $count -gt 0 ]; then
        log_info "共清理 $count 个后端进程"
    fi

    return 0
}

# 清理前端相关进程
cleanup_frontend_processes() {
    log_step "清理前端相关进程..."
    local count=0

    # 1. 通过端口查找并终止
    local port_pid
    port_pid=$(check_port "$FRONTEND_PORT")
    if [ -n "$port_pid" ]; then
        log_warn "发现占用端口 $FRONTEND_PORT 的进程 (PID: $port_pid)"
        if kill_process_force "$port_pid" "前端端口进程"; then
            log_info "已终止端口进程 (PID: $port_pid)"
            count=$((count + 1))
        fi
    fi

    # 2. 通过 PID 文件终止
    if [ -f "$FRONTEND_PID_FILE" ]; then
        local file_pid
        file_pid=$(cat "$FRONTEND_PID_FILE" 2>/dev/null)
        if [ -n "$file_pid" ] && kill -0 "$file_pid" 2>/dev/null; then
            log_warn "发现 PID 文件中的前端进程 (PID: $file_pid)"
            if kill_process_force "$file_pid" "前端PID文件进程"; then
                log_info "已终止 PID 文件进程 (PID: $file_pid)"
                count=$((count + 1))
            fi
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi

    # 3. 查找并终止 react-scripts 进程（在项目目录下运行的）
    # 同样改用 ps 命令行匹配,避免 lsof -p PID 在 macOS 卡顿
    local react_pids
    react_pids=$(ps -eo pid,command | grep -E "react-scripts.*start" | grep -v grep | grep -F "$PROJECT_DIR" | awk '{print $1}')
    if [ -n "$react_pids" ]; then
        for r_pid in $react_pids; do
            log_warn "发现遗留的 react-scripts 进程 (PID: $r_pid)"
            if kill_process_force "$r_pid" "react-scripts进程"; then
                log_info "已终止 react-scripts 进程 (PID: $r_pid)"
                count=$((count + 1))
            fi
        done
    fi

    # 4. 查找并终止 node 进程（在项目目录下运行 npm start 的）
    local node_pids
    node_pids=$(ps -eo pid,command | grep -E "node.*start\.js" | grep -v grep | grep -F "$PROJECT_DIR" | awk '{print $1}')
    if [ -n "$node_pids" ]; then
        for n_pid in $node_pids; do
            log_warn "发现遗留的 node 进程 (PID: $n_pid)"
            if kill_process_force "$n_pid" "node进程"; then
                log_info "已终止 node 进程 (PID: $n_pid)"
                count=$((count + 1))
            fi
        done
    fi

    # 等待端口释放
    local wait_count=0
    while [ -n "$(check_port "$FRONTEND_PORT")" ] && [ $wait_count -lt 10 ]; do
        sleep 0.5
        wait_count=$((wait_count + 1))
    done

    if [ $count -gt 0 ]; then
        log_info "共清理 $count 个前端进程"
    fi

    return 0
}

# 依赖预检 - 启动前检查关键命令是否可用
preflight_checks() {
    local missing_required=()
    local missing_optional=()

    # 必需依赖
    for cmd in python3 node npm curl lsof; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_required+=("$cmd")
        fi
    done

    # 可选依赖 (运行时间接使用,缺失只警告不阻塞)
    # 注意: LibreOffice 在 macOS 上 CLI 叫 soffice (Homebrew Cask 安装),
    #       在 Linux 上常叫 libreoffice; 项目代码 (X2MD/converters/legacy_office.py)
    #       实际调用的是 soffice,故任一存在即视为满足
    for cmd in ollama antiword; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_optional+=("$cmd")
        fi
    done
    if ! command -v soffice >/dev/null 2>&1 && ! command -v libreoffice >/dev/null 2>&1; then
        missing_optional+=("libreoffice")
    fi

    if [ ${#missing_required[@]} -gt 0 ]; then
        log_error "缺少必需的命令: ${missing_required[*]}"
        log_error "请先安装这些工具再启动"
        return 1
    fi

    if [ ${#missing_optional[@]} -gt 0 ]; then
        log_warn "缺少可选命令: ${missing_optional[*]}"
        # 只打印实际缺失项的说明,避免误导 (例如 ollama 在运行但被列为缺失)
        for cmd in "${missing_optional[@]}"; do
            case "$cmd" in
                ollama)       log_warn "  - ollama: 嵌入/LLM 摘要功能需要" ;;
                libreoffice)  log_warn "  - libreoffice: 旧版 Office (.ppt/.xls) 转换需要" ;;
                antiword)     log_warn "  - antiword: .doc 转换需要" ;;
            esac
        done
    fi

    # 清理 macOS Finder/iCloud 同步产生的孤儿 PID 文件 (.backend 2.pid 等)
    # 这些文件含有过期 PID,可能误杀无关进程
    # 用 -Fx 做字面整行匹配,避免路径中的 '.' 被当作正则元字符
    local orphans
    orphans=$(ls "$PROJECT_DIR"/.backend*.pid "$PROJECT_DIR"/.frontend*.pid 2>/dev/null | grep -Fxv "$BACKEND_PID_FILE" | grep -Fxv "$FRONTEND_PID_FILE" || true)
    if [ -n "$orphans" ]; then
        log_warn "发现孤儿 PID 文件,正在清理:"
        echo "$orphans" | while read -r f; do
            log_warn "  rm $f"
            rm -f "$f"
        done
    fi

    return 0
}

# =============================================================================
# 服务管理函数
# =============================================================================

# 启动后端服务
start_backend() {
    log_step "启动后端服务..."

    # 检查后端目录
    if [ ! -d "$BACKEND_DIR" ]; then
        log_error "后端目录不存在: $BACKEND_DIR"
        return 1
    fi

    # 先清理可能存在的旧进程
    cleanup_backend_processes

    # 检查虚拟环境
    if [ ! -d "$BACKEND_DIR/.venv" ]; then
        log_warn "虚拟环境不存在，尝试创建..."
        cd "$BACKEND_DIR" || { log_error "无法进入后端目录: $BACKEND_DIR"; return 1; }
        python3 -m venv .venv
        if [ $? -ne 0 ]; then
            log_error "创建虚拟环境失败，请确保已安装 Python 3"
            return 1
        fi
    fi

    # 激活虚拟环境 (先确认 activate 脚本存在,避免污染系统 Python)
    if [ ! -f "$BACKEND_DIR/.venv/bin/activate" ]; then
        log_error "虚拟环境损坏 (.venv/bin/activate 缺失): $BACKEND_DIR/.venv"
        log_error "请删除 .venv 目录后重试: rm -rf $BACKEND_DIR/.venv"
        return 1
    fi
    # shellcheck disable=SC1091
    source "$BACKEND_DIR/.venv/bin/activate"

    # 检查依赖 (仅在 venv 新建时安装,避免每次启动都 pip install)
    if [ ! -f "$BACKEND_DIR/.venv/.deps_installed" ]; then
        log_info "首次安装后端依赖..."
        python3 -m pip install -q -e "$BACKEND_DIR" 2>/dev/null || {
            log_warn "依赖安装可能不完整，继续尝试启动..."
        }
        touch "$BACKEND_DIR/.venv/.deps_installed"
    else
        log_info "依赖已安装,跳过 pip install"
    fi

    # 创建必要目录
    mkdir -p "$PROJECT_DIR/MD/0Public" "$PROJECT_DIR/MD/2Confidential" "$PROJECT_DIR/MD/1Restricted"
    mkdir -p "$PROJECT_DIR/vector_db"

    # 启动后端服务 — 日志轮转: 保留上一份为 .prev,新启动用追加,避免覆盖崩溃证据
    log_info "启动 FastAPI 后端服务 (端口: $BACKEND_PORT)..."
    cd "$BACKEND_DIR" || { log_error "无法进入后端目录: $BACKEND_DIR"; return 1; }
    if [ -f "$PROJECT_DIR/backend.log" ]; then
        mv "$PROJECT_DIR/backend.log" "$PROJECT_DIR/backend.log.prev" 2>/dev/null || true
    fi
    nohup python3 -m uvicorn app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" >> "$PROJECT_DIR/backend.log" 2>&1 &
    local backend_pid=$!
    echo "$backend_pid" > "$BACKEND_PID_FILE"

    # 等待服务就绪
    log_info "等待后端服务就绪..."
    if wait_for_service "http://localhost:$BACKEND_PORT/health" 30; then
        log_info "后端服务启动成功! PID: $backend_pid"
        log_info "API 地址: http://localhost:$BACKEND_PORT"
        log_info "API 文档: http://localhost:$BACKEND_PORT/docs"
    else
        log_error "后端服务启动超时，请检查日志: $PROJECT_DIR/backend.log"
        return 1
    fi
}

# 启动前端服务
start_frontend() {
    log_step "启动前端服务..."

    # 检查前端目录
    if [ ! -d "$FRONTEND_DIR" ]; then
        log_error "前端目录不存在: $FRONTEND_DIR"
        return 1
    fi

    # 检查 Node.js
    if ! command -v node &> /dev/null; then
        log_error "未找到 Node.js，请先安装 Node.js"
        return 1
    fi

    # 先清理可能存在的旧进程
    cleanup_frontend_processes

    # 检查 node_modules
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        log_warn "node_modules 不存在，正在安装依赖..."
        cd "$FRONTEND_DIR" || { log_error "无法进入前端目录: $FRONTEND_DIR"; return 1; }
        npm install || { log_error "npm install 失败"; return 1; }
    fi

    # 启动前端服务 — 日志轮转同后端
    log_info "启动 React 前端服务 (端口: $FRONTEND_PORT)..."
    cd "$FRONTEND_DIR" || { log_error "无法进入前端目录: $FRONTEND_DIR"; return 1; }
    if [ -f "$PROJECT_DIR/frontend.log" ]; then
        mv "$PROJECT_DIR/frontend.log" "$PROJECT_DIR/frontend.log.prev" 2>/dev/null || true
    fi
    nohup npm start >> "$PROJECT_DIR/frontend.log" 2>&1 &
    local frontend_pid=$!
    echo "$frontend_pid" > "$FRONTEND_PID_FILE"

    # 等待服务就绪
    log_info "等待前端服务就绪..."
    if wait_for_service "http://localhost:$FRONTEND_PORT" 60; then
        log_info "前端服务启动成功! PID: $frontend_pid"
        log_info "访问地址: http://localhost:$FRONTEND_PORT"
    else
        log_error "前端服务启动超时，请检查日志: $PROJECT_DIR/frontend.log"
        return 1
    fi
}

# 停止所有服务
stop_services() {
    log_step "停止所有服务..."

    # 停止后端
    cleanup_backend_processes

    # 停止前端
    cleanup_frontend_processes

    # 清理 PID 文件
    rm -f "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"

    log_info "所有服务已停止"
}

# 查看服务状态
show_status() {
    log_step "服务状态检查..."

    echo -e "\n${CYAN}┌────────────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│${NC} 服务状态                                               ${CYAN}│${NC}"
    echo -e "${CYAN}├────────────────────────────────────────────────────────┤${NC}"

    # 后端状态
    local backend_pid
    backend_pid=$(check_port "$BACKEND_PORT")
    if [ -n "$backend_pid" ]; then
        echo -e "${CYAN}│${NC} 后端服务  ${GREEN}● 运行中${NC}  端口: $BACKEND_PORT  PID: $backend_pid       ${CYAN}│${NC}"
        echo -e "${CYAN}│${NC}           API: http://localhost:$BACKEND_PORT            ${CYAN}│${NC}"
        echo -e "${CYAN}│${NC}           文档: http://localhost:$BACKEND_PORT/docs     ${CYAN}│${NC}"
    else
        echo -e "${CYAN}│${NC} 后端服务  ${RED}● 已停止${NC}  端口: $BACKEND_PORT                     ${CYAN}│${NC}"
    fi

    echo -e "${CYAN}├────────────────────────────────────────────────────────┤${NC}"

    # 前端状态
    local frontend_pid
    frontend_pid=$(check_port "$FRONTEND_PORT")
    if [ -n "$frontend_pid" ]; then
        echo -e "${CYAN}│${NC} 前端服务  ${GREEN}● 运行中${NC}  端口: $FRONTEND_PORT  PID: $frontend_pid      ${CYAN}│${NC}"
        echo -e "${CYAN}│${NC}           地址: http://localhost:$FRONTEND_PORT           ${CYAN}│${NC}"
    else
        echo -e "${CYAN}│${NC} 前端服务  ${RED}● 已停止${NC}  端口: $FRONTEND_PORT                     ${CYAN}│${NC}"
    fi

    echo -e "${CYAN}└────────────────────────────────────────────────────────┘${NC}"
}

# =============================================================================
# 主程序
# =============================================================================

main() {
    print_banner

    # 依赖预检 + 孤儿 PID 清理
    if ! preflight_checks; then
        log_error "预检失败,请修正后重试"
        exit 1
    fi

    local command=${1:-all}

    case "$command" in
        front|frontend|f)
            start_frontend
            ;;
        back|backend|b)
            start_backend
            ;;
        stop|kill|s)
            stop_services
            ;;
        status|st)
            show_status
            ;;
        all|start|"")
            start_backend
            echo ""
            start_frontend
            echo ""
            log_info "所有服务已启动!"
            echo -e "${GREEN}"
            echo "╔═══════════════════════════════════════════════════════════════╗"
            echo "║  系统访问地址:                                                 ║"
            echo "║    前端界面: http://localhost:$FRONTEND_PORT                     ║"
            echo "║    API 文档: http://localhost:$BACKEND_PORT/docs                 ║"
            echo "╚═══════════════════════════════════════════════════════════════╝"
            echo -e "${NC}"
            ;;
        *)
            echo "用法: $0 [front|back|all|stop|status]"
            echo ""
            echo "命令:"
            echo "  front, frontend, f    仅启动前端服务"
            echo "  back, backend, b      仅启动后端服务"
            echo "  all, start            启动所有服务 (默认)"
            echo "  stop, kill, s         停止所有服务"
            echo "  status, st            查看服务状态"
            exit 1
            ;;
    esac
}

# 捕获 Ctrl+C 信号
trap 'log_warn "收到中断信号，正在退出..."; exit 0' INT

# 运行主程序
main "$@"
