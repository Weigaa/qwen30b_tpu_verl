# VS Code Codex 会话列表恢复方法

## 结论

VS Code Codex 插件的会话列表不只读取 `~/.codex/sessions/*.jsonl`，还依赖 `~/.codex/state_5.sqlite` 里的 `threads` 表。账号登录和 API 登录切换后，旧会话可能因为 `model_provider` 字段不一致而被列表隐藏。

这份文档在 2026-06-11 更新后的关键变化是：

- 不要再把 `model_provider` 固定统一成 `openai`。
- 也不要只依赖“最新 VS Code 会话”的 `model_provider`。
- 正确做法是先读取 `~/.codex/auth.json` 的 `auth_mode`。
- 当 `auth_mode = "apikey"` 时，把本地历史 VS Code 会话统一到 `~/.codex/config.toml` 顶层的 `model_provider`，例如本机 API provider 是 `OpenAI`。
- 当 `auth_mode` 是账号登录模式时，把本地历史 VS Code 会话统一到插件账号 provider `openai`。
- 如果 `auth_mode` 或 `config.toml` 不可用，才退回探测“当前登录模式刚生成的最新 VS Code 会话”使用的 `model_provider`。

原因是本机当前版本的 VS Code Codex 扩展在切换登录方式后，列表可见性仍然会受到当前 provider 的影响。固定写死成 `openai` 只适用于账号登录；只看最新会话也不够，因为从账号切到 API 时，app-server 启动前可能还没有新的 API 模式 VS Code 会话，旧逻辑会继续把历史会话同步到账号模式 provider。

## 2026-05-31 这次的实际状态

本机这次看到的状态是：

- `~/.codex/session_index.jsonl` 有 9 条会话。
- `~/.codex/state_5.sqlite` 的 `threads` 表中，当前 workspace 有 9 条未归档 VS Code 会话。
- 切回 API 后，最新会话的 `model_provider` 是 `OpenAI`。
- 旧的 8 条会话还是 `openai`。
- 统一为当前活跃 provider `OpenAI` 后，数据库校验通过，9 条会话都处于可显示状态。

## 本次已执行

备份目录：

```bash
/root/.codex/backups/session_provider_fix_20260531T104523Z
```

本次实际修复步骤：

```bash
# 1. 备份 state_5.sqlite、session_index.jsonl 和所有 live jsonl 会话文件
# 2. 从 state_5.sqlite 中读取“当前最新未归档 VS Code 会话”的 model_provider
# 3. 将 ~/.codex/sessions/**/*.jsonl 中的 session_meta.model_provider 统一为这个值
# 4. 将 ~/.codex/state_5.sqlite threads.model_provider 统一为这个值
# 5. 执行 sqlite integrity_check 和 WAL checkpoint
```

修复后的校验结果：

```text
target_provider=OpenAI
state_5.sqlite integrity_check: ok
threads.model_provider: OpenAI = 9
当前 workspace 未归档 threads: 9
所有 live JSONL session_meta.model_provider: OpenAI
```

## 以后切换登录后的正确同步命令

如果账号登录和 API 登录之间切换后，会话列表再次丢失，先确认 `~/.codex/auth.json` 里的 `auth_mode` 已经是当前模式。如果是 API 模式，再确认 `~/.codex/config.toml` 顶层 `model_provider` 已经是当前要使用的 API provider，然后执行下面这段。

它会先备份，再把数据库和 JSONL 元数据统一到“当前认证模式对应的 provider”。如果认证模式或配置不可用，才会退回使用最新未归档 VS Code 会话的 provider。

```bash
set -euo pipefail

CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
CODEX_CONFIG="$CODEX_HOME_DIR/config.toml"
CODEX_AUTH_JSON="$CODEX_HOME_DIR/auth.json"
CODEX_STATE_DB="$CODEX_HOME_DIR/state_5.sqlite"
CODEX_SESSIONS_DIR="$CODEX_HOME_DIR/sessions"
CODEX_ARCHIVED_DIR="$CODEX_HOME_DIR/archived_sessions"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup_dir="$CODEX_HOME_DIR/backups/session_provider_fix_$stamp"
mkdir -p "$backup_dir/sessions"

target_provider=""
config_provider=""
if [ -f "$CODEX_CONFIG" ]; then
  config_provider=$(awk '
    BEGIN { in_top = 1 }
    /^[[:space:]]*\[/ { in_top = 0 }
    in_top && /^[[:space:]]*model_provider[[:space:]]*=/ {
      sub(/^[^=]*=[[:space:]]*/, "")
      sub(/[[:space:]]*#.*/, "")
      gsub(/^[[:space:]]+|[[:space:]]+$/, "")
      gsub(/^"|"$/, "")
      gsub(/^'\''|'\''$/, "")
      print
      exit
    }
  ' "$CODEX_CONFIG")
fi

auth_mode=""
if [ -f "$CODEX_AUTH_JSON" ]; then
  auth_mode=$(sed -n 's/.*"auth_mode"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$CODEX_AUTH_JSON" | head -n 1)
fi

case "$auth_mode" in
  apikey|api_key|apiKey)
    target_provider="$config_provider"
    ;;
  "")
    target_provider="$config_provider"
    ;;
  *)
    target_provider="openai"
    ;;
esac

if [ -z "$target_provider" ]; then
  target_provider=$(sqlite3 "$CODEX_STATE_DB" \
    "select model_provider
     from threads
     where source = 'vscode' and archived = 0
     order by updated_at desc
     limit 1;")
fi

if [ -z "$target_provider" ]; then
  echo "No target provider found. Set auth_mode/model_provider or create one new VS Code Codex session first."
  exit 1
fi

sqlite3 "$CODEX_STATE_DB" ".backup '$backup_dir/state_5.sqlite'"

if [ -f "$CODEX_HOME_DIR/session_index.jsonl" ]; then
  cp --preserve=timestamps "$CODEX_HOME_DIR/session_index.jsonl" "$backup_dir/session_index.jsonl"
fi

find "$CODEX_SESSIONS_DIR" "$CODEX_ARCHIVED_DIR" -type f -name '*.jsonl' -print0 2>/dev/null \
  | xargs -0 -r cp --parents --preserve=timestamps --reflink=auto -t "$backup_dir/sessions"

while IFS= read -r -d '' f; do
  perl -i -pe 's/"model_provider":"[^"]+"/"model_provider":"'"$target_provider"'"/ if /"type":"session_meta"/' "$f"
  backup_file="$backup_dir/sessions${f}"
  if [ -f "$backup_file" ]; then
    touch -r "$backup_file" "$f"
  fi
done < <(find "$CODEX_SESSIONS_DIR" "$CODEX_ARCHIVED_DIR" -type f -name '*.jsonl' -print0 2>/dev/null)

sql_provider=$(printf '%s' "$target_provider" | sed "s/'/''/g")
sqlite3 "$CODEX_STATE_DB" \
  "update threads
   set model_provider = '$sql_provider'
   where source = 'vscode';
   pragma wal_checkpoint(TRUNCATE);"

echo "target_provider=$target_provider"
sqlite3 "$HOME/.codex/state_5.sqlite" "pragma integrity_check;"
sqlite3 -header -column "$HOME/.codex/state_5.sqlite" \
  "select model_provider, count(*) as n
   from threads
   where source = 'vscode'
   group by model_provider
   order by model_provider;"
```

执行后，在 VS Code 里运行 `Developer: Reload Window`，或重启 VS Code 远端窗口，让 Codex 插件重新读取本地状态库。

## 验证命令

```bash
target_provider=$(sqlite3 ~/.codex/state_5.sqlite \
  "select model_provider
   from threads
   where source = 'vscode' and archived = 0
   order by updated_at desc
   limit 1;")

echo "target_provider=$target_provider"

sqlite3 -header -column ~/.codex/state_5.sqlite \
  "select model_provider, count(*) as n
   from threads
   where source = 'vscode'
   group by model_provider
   order by model_provider;"

sqlite3 -header -column ~/.codex/state_5.sqlite \
  "select count(*) as visible_threads
   from threads
   where archived = 0 and cwd = '/workspace/cann-recipes-train/llm_rl/qwen3';"

find ~/.codex/sessions -type f -name '*.jsonl' -print0 \
  | xargs -0 grep -L "\"model_provider\":\"$target_provider\""
```

预期结果：

- 第一条应输出当前最新会话的 provider，例如这次是 `OpenAI`。
- 第二条应只剩一个 provider 分组。
- 第三条应显示当前 workspace 下的未归档会话数量。
- 第四条如果没有输出，说明 live JSONL 文件都已经统一到当前 provider。

## 何时需要重新检查扩展侧行为

如果你已经完成上面的 provider 统一，重载 VS Code 后列表仍然不完整，再去检查当前 VS Code Codex 扩展版本的 `thread/list` 请求行为。

也就是说：

- 先做本地 provider 统一，这是当前默认首选修法。
- 只有在 provider 已统一但列表仍然异常时，才考虑再次检查扩展 bundle 或 app-server 的过滤逻辑。

不要直接照搬更早版本里“修改 `extension.js` 把 `modelProviders` 从 `null` 改成 `[]`”的补丁，因为扩展版本升级后实现细节可能已经变化。

## 回滚方法

如果修复后插件表现异常，可以回滚到本次备份：

```bash
backup_dir=/root/.codex/backups/session_provider_fix_20260531T104523Z

cp "$backup_dir/state_5.sqlite" ~/.codex/state_5.sqlite

if [ -f "$backup_dir/session_index.jsonl" ]; then
  cp "$backup_dir/session_index.jsonl" ~/.codex/session_index.jsonl
fi

rsync -a "$backup_dir/sessions/root/.codex/sessions/" ~/.codex/sessions/
```

回滚后同样需要重载 VS Code 窗口。

## 2026-06-10 账号登录恢复记录

这次从 API 切回账号登录后，最新 VS Code 会话的 `model_provider` 变成了 `openai`，但历史 VS Code 会话大多还是 `OpenAI`。修复前当前 workspace 的未归档 VS Code 会话分布是：

```text
OpenAI = 17
openai = 1
```

已执行一次手动同步，备份目录为：

```bash
/root/.codex/backups/session_provider_fix_20260610T051930Z
```

修复后校验结果：

```text
target_provider=openai
state_5.sqlite integrity_check: ok
threads.model_provider: openai = 18 active vscode, 8 archived vscode
当前 workspace 未归档 threads: 18
```

此外，已在 VS Code Codex 扩展的本地 wrapper 里加入 app-server 启动前自动同步：

```bash
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64/bin/linux-aarch64/codex
```

wrapper 现在只在 `codex app-server` 启动时执行同步逻辑：

- 读取 `~/.codex/state_5.sqlite` 中最新未归档 VS Code thread 的 `model_provider`。
- 将 `threads` 表里所有 `source = 'vscode'` 的 provider 统一到这个值。
- 将 `~/.codex/sessions/**/*.jsonl` 里的 `session_meta.model_provider` 统一到这个值。
- 每次自动同步前会备份数据库到 `~/.codex/backups/session_provider_autosync_<timestamp>/state_5.sqlite`。
- 同步失败不会阻断 Codex 启动。

wrapper 修改前的备份文件：

```bash
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64/bin/linux-aarch64/codex.bak_session_provider_sync_20260610T052247Z
```

本次 wrapper 校验：

```text
sh -n codex: ok
codex --version: codex-cli 0.137.0-alpha.4
state_5.sqlite integrity_check: ok
```

注意：2026-06-11 后这个自动同步方案已优化，不再依赖“当前登录方式下已经出现过至少一条最新 VS Code 会话”。API key 模式下只要 `~/.codex/config.toml` 顶层 `model_provider` 已经切到当前 provider，重载 VS Code 窗口触发 app-server 启动时就会自动同步；账号登录模式下会自动同步到 `openai`。

## 2026-06-11 API 登录恢复记录

这次从账号登录切回 API 后，`~/.codex/auth.json` 里 `auth_mode` 是 `apikey`，`~/.codex/config.toml` 顶层 `model_provider` 是 `OpenAI`，但 `state_5.sqlite` 里当前 workspace 的 20 条未归档 VS Code 会话只有 1 条是 `OpenAI`，其余 19 条仍是 `openai`。这说明 2026-06-10 的 wrapper 仍然可能在切换后选错目标 provider。

已执行一次手动同步，备份目录为：

```bash
/root/.codex/backups/session_provider_fix_20260611T022158Z
```

修复后校验结果：

```text
target_provider=OpenAI
state_5.sqlite integrity_check: ok
threads.model_provider: OpenAI = 20 active vscode, 9 archived vscode
当前 workspace 未归档 threads: 20
```

同时已优化 VS Code Codex 扩展 wrapper：

```bash
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64/bin/linux-aarch64/codex
```

新的自动同步优先级：

- 先读取 `~/.codex/auth.json` 的 `auth_mode`。
- 如果是 `apikey`，使用 `~/.codex/config.toml` 顶层 `model_provider`。
- 如果是账号登录模式，使用插件账号 provider `openai`。
- 如果 `auth_mode` 或配置不可用，再读取最新未归档 VS Code thread 的 `model_provider`。
- 将 `threads` 表里所有 `source = 'vscode'` 的 provider 统一到目标 provider。
- 将 `~/.codex/sessions/**/*.jsonl` 和 `~/.codex/archived_sessions/**/*.jsonl` 里的 `session_meta.model_provider` 统一到目标 provider。
- 每次自动同步前会备份数据库到 `~/.codex/backups/session_provider_autosync_<timestamp>/state_5.sqlite`。
- 同步失败不会阻断 Codex 启动。

wrapper 修改前的备份文件：

```bash
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64/bin/linux-aarch64/codex.bak_config_provider_sync_20260611T022336Z
```

本次 wrapper 校验：

```text
sh -n codex: ok
codex --version: codex-cli 0.137.0-alpha.4
state_5.sqlite integrity_check: ok
```
