# Codex VS Code + Tailscale Userspace Recovery Runbook

Date captured: 2026-06-08 UTC

This document records the current working fallback state for the VS Code Codex plugin when kernel TUN routing is unstable. Use it to return to the known-good userspace proxy setup if a later TUN recovery attempt fails halfway.

## Known-Good State

The VS Code Codex plugin reaches the OpenAI-compatible endpoint through Tailscale userspace HTTP proxy:

- Target API endpoint: `http://100.94.44.45:8080`
- Tailscale userspace HTTP proxy: `http://127.0.0.1:1056`
- Tailscale userspace SOCKS5 proxy: `127.0.0.1:1055`
- Tailscale socket: `/home/weijia/tailscale/run/tailscaled.sock`
- Tailscale state dir: `/home/weijia/tailscale/state`

Current userspace daemon command:

```bash
tailscaled \
  --statedir=/home/weijia/tailscale/state \
  --socket=/home/weijia/tailscale/run/tailscaled.sock \
  --tun=userspace-networking \
  --socks5-server=127.0.0.1:1055 \
  --outbound-http-proxy-listen=127.0.0.1:1056
```

Current Codex config:

```toml
[model_providers.OpenAI]
name = "OpenAI"
base_url = "http://100.94.44.45:8080"
wire_api = "responses"
requires_openai_auth = true
```

Current VS Code Codex wrapper paths:

```text
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64/bin/linux-aarch64/codex
/root/.vscode-server/extensions/openai.chatgpt-26.602.40724-linux-arm64/bin/linux-aarch64/codex
```

The active path is whichever extension directory appears in the running app-server or extension host logs. As of 2026-06-09, the active directory is `openai.chatgpt-26.602.71036-linux-arm64`; the older `26.602.40724` directory may still exist but is not sufficient by itself.

Wrapper content:

```sh
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
```

Important: do not put `100.94.44.45` or `100.64.0.0/10` in `NO_PROXY` in this fallback mode. Codex must use `127.0.0.1:1056` to reach the tailnet endpoint.

## VS Code Extension Auto-Update Failure Mode

On 2026-06-09 the VS Code Codex plugin started reconnecting again because VS Code had installed a newer extension directory:

```text
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64
```

The previous wrapper was only installed in:

```text
/root/.vscode-server/extensions/openai.chatgpt-26.602.40724-linux-arm64
```

The new `71036` directory initially contained a raw ELF `codex` binary, so `app-server` inherited the VS Code/SSH environment:

```text
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
```

That made requests to `http://100.94.44.45:8080/responses` go through the wrong proxy and return `502 Bad Gateway`.

Fix applied:

```text
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64/bin/linux-aarch64/codex      -> shell wrapper
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64/bin/linux-aarch64/codex-real -> original ELF
```

After restarting only the VS Code extension host, the new process was:

```text
/root/.vscode-server/extensions/openai.chatgpt-26.602.71036-linux-arm64/bin/linux-aarch64/codex-real app-server --analytics-default-enabled
```

and its environment correctly contained:

```text
HTTP_PROXY=http://127.0.0.1:1056
http_proxy=http://127.0.0.1:1056
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```

If this happens again, first check the active extension directory:

```bash
pgrep -af 'codex.*app-server|codex-real.*app-server|bootstrap-fork --type=extensionHost'
```

Then install the same wrapper in the active `openai.chatgpt-*` directory, not just the old one.

## Quick Health Check

Run these checks before changing anything:

```bash
pgrep -af 'tailscale|tailscaled'
tailscale --socket=/home/weijia/tailscale/run/tailscaled.sock status
curl -sS --proxy http://127.0.0.1:1056 --connect-timeout 5 --max-time 10 \
  -o /tmp/codex_tailscale_proxy_check.out \
  -w '%{http_code} %{remote_ip}:%{remote_port}\n' \
  http://100.94.44.45:8080/
```

Expected:

```text
200 127.0.0.1:1056
```

Check the VS Code Codex app-server environment:

```bash
PID=$(pgrep -f 'codex-real app-server' | head -1)
tr '\0' '\n' < /proc/"$PID"/environ | sort | grep -Ei 'HTTP_PROXY|NO_PROXY|http_proxy|no_proxy'
```

Expected:

```text
HTTP_PROXY=http://127.0.0.1:1056
NO_PROXY=127.0.0.1,localhost
http_proxy=http://127.0.0.1:1056
no_proxy=127.0.0.1,localhost
```

Check Codex logs for successful Responses API calls:

```bash
sqlite3 -separator ' | ' /root/.codex/logs_2.sqlite \
  "select datetime(ts,'unixepoch'), target, substr(feedback_log_body,1,1800)
   from logs
   where feedback_log_body like '%100.94.44.45%'
      or feedback_log_body like '%127.0.0.1:1056%'
      or feedback_log_body like '%proxy(%'
   order by ts desc limit 20;"
```

Good signs:

- `proxy(http://127.0.0.1:1056/) intercepts 'http://100.94.44.45:8080/'`
- `Request completed method=POST url=http://100.94.44.45:8080/responses status=200 OK`

Bad signs:

- `proxy(http://127.0.0.1:7890/) intercepts ...`
- direct `connecting to 100.94.44.45:8080` followed by TCP timeout
- `502 Bad Gateway` from `127.0.0.1:7890`

## Restore This Fallback State

Use this if TUN recovery fails and Codex stops working.

### 1. Start Userspace Tailscale

Do not delete TUN interfaces unless you are deliberately doing TUN cleanup. To restore the fallback, just ensure the userspace daemon is running.

```bash
mkdir -p /home/weijia/tailscale/run /home/weijia/tailscale/logs /home/weijia/tailscale/state

pgrep -af 'tailscaled.*userspace-networking' || \
nohup tailscaled \
  --statedir=/home/weijia/tailscale/state \
  --socket=/home/weijia/tailscale/run/tailscaled.sock \
  --tun=userspace-networking \
  --socks5-server=127.0.0.1:1055 \
  --outbound-http-proxy-listen=127.0.0.1:1056 \
  > /home/weijia/tailscale/logs/tailscaled-userspace.log 2>&1 &
```

Verify:

```bash
tailscale --socket=/home/weijia/tailscale/run/tailscaled.sock status
curl -sS --proxy http://127.0.0.1:1056 --connect-timeout 5 --max-time 10 \
  -o /tmp/codex_tailscale_proxy_check.out \
  -w '%{http_code} %{remote_ip}:%{remote_port}\n' \
  http://100.94.44.45:8080/
```

### 2. Restore Codex Provider Config

Ensure `/root/.codex/config.toml` contains:

```toml
[model_providers.OpenAI]
name = "OpenAI"
base_url = "http://100.94.44.45:8080"
wire_api = "responses"
requires_openai_auth = true
```

### 3. Restore VS Code Codex Wrapper

Find the active extension directory:

```bash
pgrep -af 'codex.*app-server|codex-real.*app-server|bootstrap-fork --type=extensionHost'
ls -d /root/.vscode-server/extensions/openai.chatgpt-*-linux-arm64
```

Ensure the active directory's `bin/linux-aarch64/codex` is the wrapper shown in "Known-Good State".

The original real binary should exist as:

```text
/root/.vscode-server/extensions/openai.chatgpt-<active-version>-linux-arm64/bin/linux-aarch64/codex-real
```

The wrapper must be executable:

```bash
chmod +x /root/.vscode-server/extensions/openai.chatgpt-<active-version>-linux-arm64/bin/linux-aarch64/codex
```

### 4. Restart Only the Codex App-Server

This is safe for Tailscale; it only restarts the VS Code Codex backend process.

```bash
pkill -f 'codex-real app-server' || true
```

If VS Code does not respawn it automatically, reload the VS Code window or restart the extension host.

Verify:

```bash
pgrep -af 'codex.*app-server|codex-real app-server'
PID=$(pgrep -f 'codex-real app-server' | head -1)
tr '\0' '\n' < /proc/"$PID"/environ | sort | grep -Ei 'HTTP_PROXY|NO_PROXY|http_proxy|no_proxy'
```

## Why This Fallback Works

The correct endpoint is `100.94.44.45:8080`, but direct kernel routing currently sends it to the physical LAN gateway:

```text
100.94.44.45 via 192.168.0.1 dev enp23s0f3
```

That direct path times out. The userspace proxy path works:

```text
Codex app-server -> HTTP_PROXY 127.0.0.1:1056 -> Tailscale userspace daemon -> 100.94.44.45:8080
```

The previous SSH `RemoteForward 7890 127.0.0.1:7890` caused a different failure mode. When Codex inherited `HTTP_PROXY=http://127.0.0.1:7890`, requests to `100.94.44.45:8080` returned `502 Bad Gateway`. The fallback wrapper intentionally avoids `7890` and pins Codex to `1056`.

## Current TUN Problem Summary

TUN is not proven impossible in this container. In fact, a TUN daemon using `--tun=tscodex0` previously started and reached `Running`:

```text
Program starting ... --tun=tscodex0
Engine created.
Switching ipn state Starting -> Running
```

It later stopped via SIGTERM:

```text
tailscaled got signal terminated; shutting down
```

The older `tailscale0` name had a separate failure:

```text
TUN device tailscale0 is busy
tstun.New("tailscale0"): device or resource busy
```

Current mixed state:

- userspace daemon is running
- `tailscale0` TUN interface still exists with old address `100.119.17.14`
- policy routing table 52 still has old routes such as `100.94.44.44 dev tailscale0`
- `100.94.44.45` is not routed through table 52

For future TUN recovery, prefer `--tun=tscodex0` over `--tun=tailscale0`, and avoid touching the Codex fallback until TUN is proven stable.

## Do Not Do This During Fallback Recovery

- Do not add `100.94.44.45` to `NO_PROXY`.
- Do not point Codex at `127.0.0.1:7890`.
- Do not delete `tailscale0` or `tscodex0` unless intentionally doing TUN cleanup.
- Do not kill all `tailscaled` processes unless ready to immediately restore userspace mode.
- Do not change `/root/.codex/config.toml` back to `100.94.44.44`.

## Minimal Rollback Checklist

If everything is on fire, run:

```bash
mkdir -p /home/weijia/tailscale/run /home/weijia/tailscale/logs /home/weijia/tailscale/state

pgrep -af 'tailscaled.*userspace-networking' || \
nohup tailscaled \
  --statedir=/home/weijia/tailscale/state \
  --socket=/home/weijia/tailscale/run/tailscaled.sock \
  --tun=userspace-networking \
  --socks5-server=127.0.0.1:1055 \
  --outbound-http-proxy-listen=127.0.0.1:1056 \
  > /home/weijia/tailscale/logs/tailscaled-userspace.log 2>&1 &

curl -sS --proxy http://127.0.0.1:1056 --connect-timeout 5 --max-time 10 \
  -o /tmp/codex_tailscale_proxy_check.out \
  -w '%{http_code} %{remote_ip}:%{remote_port}\n' \
  http://100.94.44.45:8080/

pkill -f 'codex-real app-server' || true
```

Then open or reload the VS Code Codex plugin and send a small test message.
