记录如何在linux下设置代理，首先去flowercloud购买一个vpn，里面会导出一个clash的配置文件。
# 1 安装clash
```shell
#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v1.19.24}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_DIR="$ROOT_DIR/tools/clash/bin"
TMP_GZ="$BIN_DIR/mihomo.gz"
TARGET_BIN="$BIN_DIR/mihomo"
TARGET_LINK="$BIN_DIR/clash"
ASSET="mihomo-linux-amd64-compatible-${VERSION}.gz"
URL="https://github.com/MetaCubeX/mihomo/releases/download/${VERSION}/${ASSET}"

mkdir -p "$BIN_DIR"

echo "下载: $URL"
# curl -fL "$URL" -o "$TMP_GZ"
mv "$ASSET" "$TMP_GZ"
gunzip -f "$TMP_GZ"
chmod +x "$TARGET_BIN"
ln -sf mihomo "$TARGET_LINK"

echo
"$TARGET_BIN" -v

echo
echo "安装完成: $TARGET_BIN"
echo "本地 clash 入口: $TARGET_LINK"
```
# 2 启动clash
```shell
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.local.yaml"
LOG_FILE="$SCRIPT_DIR/clash.log"
PID_FILE="$SCRIPT_DIR/clash.pid"
HTTP_PORT=17890
SOCKS_PORT=17891
REDIR_PORT=17892
MIXED_PORT=17893
CONTROLLER_PORT=19090

if ! command -v clash >/dev/null 2>&1; then
  echo "未找到 clash，请先安装 clash 或将其加入 PATH" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
用法: start_clash.sh [start|stop|restart|status|test]

  start   启动 clash
  stop    停止 clash
  restart 重启 clash
  status  查看状态
  test    校验配置文件
EOF
}

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

port_in_use() {
  local port="$1"
  ss -ltn | awk '{print $4}' | grep -Eq "(^|[.:])${port}$"
}

ports_ready() {
  for port in "$HTTP_PORT" "$SOCKS_PORT" "$REDIR_PORT" "$MIXED_PORT" "$CONTROLLER_PORT"; do
    if ! port_in_use "$port"; then
      return 1
    fi
  done
  return 0
}

ensure_ports_free() {
  for port in "$HTTP_PORT" "$SOCKS_PORT" "$REDIR_PORT" "$MIXED_PORT" "$CONTROLLER_PORT"; do
    if port_in_use "$port"; then
      echo "端口 $port 已被占用，请先释放或修改 $CONFIG_FILE" >&2
      exit 1
    fi
  done
}

test_config() {
  clash -t -f "$CONFIG_FILE"
}

start_clash() {
  [[ -f "$CONFIG_FILE" ]] || { echo "配置文件不存在: $CONFIG_FILE" >&2; exit 1; }
  test_config >/dev/null

  if is_running; then
    echo "clash 已在运行，PID=$(cat "$PID_FILE")"
    return 0
  fi

  ensure_ports_free
  nohup clash -f "$CONFIG_FILE" >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  sleep 3

  if is_running && ports_ready; then
    echo "clash 已启动，PID=$(cat "$PID_FILE")"
    echo "HTTP 代理: 127.0.0.1:$HTTP_PORT"
    echo "SOCKS5 代理: 127.0.0.1:$SOCKS_PORT"
    echo "Mixed 代理: 127.0.0.1:$MIXED_PORT"
    echo "控制接口: http://127.0.0.1:$CONTROLLER_PORT"
  else
    echo "clash 启动失败，请查看日志: $LOG_FILE" >&2
    exit 1
  fi
}

stop_clash() {
  if ! is_running; then
    rm -f "$PID_FILE"
    echo "clash 未运行"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  rm -f "$PID_FILE"
  echo "clash 已停止"
}

status_clash() {
  if is_running; then
    echo "clash 运行中，PID=$(cat "$PID_FILE")"
    echo "配置文件: $CONFIG_FILE"
    echo "日志文件: $LOG_FILE"
    echo "HTTP 代理: 127.0.0.1:$HTTP_PORT"
    echo "SOCKS5 代理: 127.0.0.1:$SOCKS_PORT"
    echo "Mixed 代理: 127.0.0.1:$MIXED_PORT"
    echo "控制接口: http://127.0.0.1:$CONTROLLER_PORT"
  else
    echo "clash 未运行"
    return 1
  fi
}

ACTION="${1:-start}"
case "$ACTION" in
  start)
    start_clash
    ;;
  stop)
    stop_clash
    ;;
  restart)
    stop_clash || true
    start_clash
    ;;
  status)
    status_clash
    ;;
  test)
    test_config
    ;;
  *)
    usage
    exit 1
    ;;
esac


```

# 3 停止clash
```shell
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start_clash.sh" stop

```

# 4 选择代理
```shell
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="$SCRIPT_DIR/start_clash.sh"
CONTROLLER_URL="${CLASH_CONTROLLER:-http://127.0.0.1:19090}"
GROUP="Proxies"

usage() {
  cat <<'EOF'
用法:
  switch_proxy.sh groups
  switch_proxy.sh current [-g GROUP]
  switch_proxy.sh list [-g GROUP]
  switch_proxy.sh set [-g GROUP] NODE_NAME

示例:
  ./switch_proxy.sh groups
  ./switch_proxy.sh current
  ./switch_proxy.sh list
  ./switch_proxy.sh set "🇭🇰 香港高级 IEPL 专线 1"
  ./switch_proxy.sh -g HK set "🇭🇰 香港标准 IEPL 专线 2"
EOF
}

require_tools() {
  command -v curl >/dev/null 2>&1 || { echo "未找到 curl" >&2; exit 1; }
  command -v python3 >/dev/null 2>&1 || { echo "未找到 python3" >&2; exit 1; }
}

ensure_clash_running() {
  "$START_SCRIPT" start >/dev/null
}

api_get() {
  curl -fsS "$CONTROLLER_URL/proxies"
}

urlencode() {
  python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"
}

json_name_body() {
  python3 -c 'import json, sys; print(json.dumps({"name": sys.argv[1]}, ensure_ascii=False))' "$1"
}

print_groups() {
  local raw
  raw="$(api_get)"
  python3 -c 'import json,sys
obj=json.loads(sys.stdin.read())
proxies=obj.get("proxies", {})
names=[]
for name, item in proxies.items():
    if item.get("type") == "Selector":
        names.append(name)
for name in sorted(names):
    print(name)' <<<"$raw"
}

print_current() {
  local group="$1"
  local raw
  raw="$(api_get)"
  python3 -c 'import json,sys
obj=json.loads(sys.stdin.read())
group=sys.argv[1]
item=obj.get("proxies", {}).get(group)
if not item:
    print(f"未找到代理组: {group}", file=sys.stderr)
    raise SystemExit(1)
print(f"组: {group}")
print("当前节点:", item.get("now", "<unknown>"))' "$group" <<<"$raw"
}

print_list() {
  local group="$1"
  local raw
  raw="$(api_get)"
  python3 -c 'import json,sys
obj=json.loads(sys.stdin.read())
group=sys.argv[1]
item=obj.get("proxies", {}).get(group)
if not item:
    print(f"未找到代理组: {group}", file=sys.stderr)
    raise SystemExit(1)
current=item.get("now")
print(f"组: {group}")
print("可选节点:")
for name in item.get("all", []):
    prefix="*" if name == current else "-"
    print(f"{prefix} {name}")' "$group" <<<"$raw"
}

set_node() {
  local group="$1"
  local node="$2"
  local encoded_group
  local body
  encoded_group="$(urlencode "$group")"
  body="$(json_name_body "$node")"
  curl -fsS -X PUT "$CONTROLLER_URL/proxies/$encoded_group" \
    -H 'Content-Type: application/json' \
    --data "$body" >/dev/null
  echo "已切换组 $group 到节点: $node"
  print_current "$group"
}

require_tools
ensure_clash_running

while getopts ':g:h' opt; do
  case "$opt" in
    g) GROUP="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

ACTION="${1:-}"
case "$ACTION" in
  groups)
    print_groups
    ;;
  current)
    print_current "$GROUP"
    ;;
  list)
    print_list "$GROUP"
    ;;
  set)
    shift
    [[ $# -ge 1 ]] || { echo "缺少节点名称" >&2; usage; exit 1; }
    set_node "$GROUP" "$*"
    ;;
  *)
    usage
    exit 1
    ;;
esac

```

# 启动codex
```shell
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HTTP_PROXY_URL="http://127.0.0.1:7890"
SOCKS_PROXY_URL="socks5://127.0.0.1:7891"
START_SCRIPT="$ROOT_DIR/tools/clash/start_clash.sh"
PID_FILE="$ROOT_DIR/tools/clash/run/clash.pid"

if ! command -v codex >/dev/null 2>&1; then
  echo "未找到 codex 命令，请先安装 Codex CLI" >&2
  exit 1
fi

if [[ ! -f "$PID_FILE" ]] || ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  "$START_SCRIPT"
fi

export HTTP_PROXY="$HTTP_PROXY_URL"
export HTTPS_PROXY="$HTTP_PROXY_URL"
export ALL_PROXY="$SOCKS_PROXY_URL"
export NO_PROXY="localhost,127.0.0.1,::1"

echo "Codex 将通过 HTTP_PROXY=$HTTP_PROXY 和 ALL_PROXY=$ALL_PROXY 访问网络"
exec codex "$@"
~                   
```
