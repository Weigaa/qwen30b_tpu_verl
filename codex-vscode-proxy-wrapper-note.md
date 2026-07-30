# Codex VS Code Extension Proxy Fix

## Quick Recovery Runbook

Use this when the VS Code Codex extension is upgraded and starts failing again with errors like:

```text
unexpected status 502 Bad Gateway
url: http://100.94.44.45:8080/responses
```

### 1. Try Reload First

In VS Code, run:

```text
Developer: Reload Window
```

If the persisted settings below are still in effect, this is often enough.

### 2. Restore Persistent VS Code Settings

Run this on the remote machine:

```sh
python3 - <<'PY'
import json
from pathlib import Path

settings = {
    "chatgpt.cliExecutable": "/workspace/cann-recipes-train/llm_rl/qwen3/codex-vscode-proxy-wrapper.sh",
    "http.proxy": "http://127.0.0.1:1056",
    "http.proxyStrictSSL": False,
}

for path in [
    Path("/root/.vscode-server/data/Machine/settings.json"),
    Path("/root/.vscode-server/data/User/settings.json"),
]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text().strip():
        data = json.loads(path.read_text())
    else:
        data = {}
    data.update(settings)
    path.write_text(json.dumps(data, ensure_ascii=False, indent="\t") + "\n")
PY
```

Then run `Developer: Reload Window` again.

### 3. Patch The Latest Extension Bundle If Needed

If reload still starts the bundled extension path directly, patch the latest installed bundle:

```sh
EXT=$(ls -td /root/.vscode-server/extensions/openai.chatgpt-*-linux-arm64 | head -1)
BIN="$EXT/bin/linux-aarch64"

if [ ! -x "$BIN/codex-real" ]; then
  mv "$BIN/codex" "$BIN/codex-real"
fi

cat > "$BIN/codex" <<'SH'
#!/bin/sh

if [ "${1:-}" = "app-server" ]; then
  export HTTP_PROXY="http://127.0.0.1:1056"
  export http_proxy="http://127.0.0.1:1056"
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="127.0.0.1,localhost"
  unset HTTPS_PROXY ALL_PROXY https_proxy all_proxy
fi

SELF_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$SELF_DIR/codex-real" "$@"
SH

chmod +x "$BIN/codex"
"$BIN/codex" --version
```

Then run:

```text
Developer: Reload Window
```

### 4. Verify

Check that the proxy is listening:

```sh
ss -ltnpe | grep -E '127\.0\.0\.1:1056|127\.0\.0\.1:1055'
```

Check that the target is reachable through the proxy:

```sh
curl -I --max-time 5 --proxy http://127.0.0.1:1056 http://100.94.44.45:8080/responses
```

Any HTTP response from the target service, including `404 Not Found`, means the proxy path is
working. A proxy failure usually mentions `127.0.0.1:7890`, connection refused, timeout, or failed
socket connection.

Check which Codex app-server process is running:

```sh
pgrep -af 'codex.*app-server|bootstrap-fork --type=extensionHost|server-main.js'
```

If an old `codex app-server` process is still running after the patch, kill only that old process
and reload VS Code:

```sh
kill <old-codex-app-server-pid>
```

## Current Fix

The VS Code Codex extension is configured to use a stable wrapper:

```text
/workspace/cann-recipes-train/llm_rl/qwen3/codex-vscode-proxy-wrapper.sh
```

The setting is stored in:

```text
/root/.vscode-server/data/Machine/settings.json
```

as:

```json
"chatgpt.cliExecutable": "/workspace/cann-recipes-train/llm_rl/qwen3/codex-vscode-proxy-wrapper.sh",
"http.proxy": "http://127.0.0.1:1056",
"http.proxyStrictSSL": false
```

The same persistent settings are also written to:

```text
/root/.vscode-server/data/User/settings.json
```

## Why This Is More Stable

The extension normally runs its bundled CLI from a versioned path like:

```text
/root/.vscode-server/extensions/openai.chatgpt-*/bin/linux-aarch64/codex
```

That path changes whenever the extension upgrades, so patching that file directly is not stable.

The stable wrapper lives in the workspace instead. On each launch it finds the newest installed
`openai.chatgpt-*` extension bundle and execs its real `codex` binary.

Recent extension builds may still launch their bundled `codex` path directly instead of honoring
`chatgpt.cliExecutable`. To make upgrades more robust, the remote VS Code `http.proxy` setting is
also pinned to the Tailscale HTTP proxy. The extension reads `http.proxy` and injects it into the
`codex app-server` environment as `HTTP_PROXY` and `HTTPS_PROXY`.

## Proxy Behavior

When the extension starts:

```text
codex app-server ...
```

the wrapper sets:

```sh
HTTP_PROXY=http://127.0.0.1:1056
http_proxy=http://127.0.0.1:1056
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```

and unsets:

```sh
HTTPS_PROXY ALL_PROXY https_proxy all_proxy
```

This avoids the extension accidentally using VS Code's `http.proxy` value such as
`http://127.0.0.1:7890`.

## One-Time Reload Required

After changing `chatgpt.cliExecutable`, reload the VS Code window or reconnect the remote window so
the extension restarts and uses the wrapper.

## Current Installed Bundle Patch

For immediate recovery, the current extension bundle was also patched:

```text
/root/.vscode-server/extensions/openai.chatgpt-26.5616.81150-linux-arm64/bin/linux-aarch64/codex
```

is now a shell wrapper, and the original binary is:

```text
/root/.vscode-server/extensions/openai.chatgpt-26.5616.81150-linux-arm64/bin/linux-aarch64/codex-real
```

This per-version patch may be overwritten by future extension upgrades, but the
`chatgpt.cliExecutable` and `http.proxy` settings should continue to point at the stable workspace
wrapper and the correct proxy.

## Verification

Run:

```sh
/workspace/cann-recipes-train/llm_rl/qwen3/codex-vscode-proxy-wrapper.sh --version
```

Expected output should include the current bundled Codex CLI version.

Also verify the proxy path itself:

```sh
curl -I --max-time 5 --proxy http://127.0.0.1:1056 http://100.94.44.45:8080/responses
```

Any HTTP response from the target service, even `404 Not Found`, means the proxy path is working.
