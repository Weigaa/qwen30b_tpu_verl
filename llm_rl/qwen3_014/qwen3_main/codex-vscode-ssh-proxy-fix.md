# VS Code Codex SSH 代理故障排查与修复

本文记录通过 VS Code Remote SSH 使用 Codex 插件时，出现 `Connection failed: error sending request` 的定位过程和修复方案。

## 典型现象

VS Code 中 Codex 插件提示以下错误，并持续重试：

```text
Connection failed: error sending request
```

## 本次故障根因

本环境使用自定义 OpenAI 兼容网关 `http://100.94.44.45:8080`，该网关在远端环境中可以直接访问；但 VS Code Server 的 Machine 配置中设置了全局代理 `http://100.97.129.55:7890`，该代理端口已经不可用。

关键点是：仅在 Codex 启动包装脚本中清除 `HTTP_PROXY` 和 `HTTPS_PROXY` 不一定生效。Codex VS Code 插件会读取 VS Code 的 `http.proxy` 配置，并在启动 `app-server` 时重新注入代理环境变量。因此最终的 Codex 进程仍带有失效代理，导致 Responses API 的流式请求失败。

## 快速定位

先分别检查 API 网关和代理端口：

```bash
timeout 3 bash -c '</dev/tcp/100.94.44.45/8080'
timeout 3 bash -c '</dev/tcp/100.97.129.55/7890'
```

再检查 Codex app-server 是否继承了代理：

```bash
pid=$(pgrep -f '/openai.chatgpt-.*/bin/.*/codex .*app-server' | head -1)
tr '\0' '\n' </proc/$pid/environ | grep -i proxy
```

如果网关端口可达、代理端口不可达，且进程中存在 `HTTP_PROXY` 或 `HTTPS_PROXY`，即可确定是代理注入问题。

## 修复方案

编辑远端 VS Code Server 的 Machine 配置：

```bash
$HOME/.vscode-server/data/Machine/settings.json
```

删除失效的 `http.proxy`，并关闭 VS Code 的代理支持：

```json
{
  "http.proxySupport": "off"
}
```

本次实际修改为：

```diff
- "http.proxy": "http://100.97.129.55:7890",
- "http.proxySupport": "override",
+ "http.proxySupport": "off",
```

不要只依赖 `http.noProxy`。插件启动 Codex 时会直接读取 `http.proxy` 并注入代理，`http.noProxy` 不会阻止该注入逻辑。

如果仍需保留其它代理用途，应确保代理服务实际监听，并将自定义网关配置为直连；对于当前自定义网关，关闭 VS Code 全局代理最简单可靠。

修改后执行：

1. `Developer: Reload Window`
2. 如果 app-server 没有重启，再执行 `Developer: Restart Extension Host`

此外，Codex 启动包装脚本可以作为额外保护，清除继承的代理变量：

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY='*'
export no_proxy='*'
```

但这不能替代删除 VS Code 的 `http.proxy`，因为插件可能在启动 app-server 时再次注入代理。

## 修复后验证

确认新的 app-server 不再包含代理变量：

```bash
pid=$(pgrep -f '/openai.chatgpt-.*/bin/.*/codex .*app-server' | head -1)
tr '\0' '\n' </proc/$pid/environ | grep -i proxy
```

该命令应无输出。然后使用插件内置 Codex 二进制发送最小请求，确认网关、鉴权、模型和流式 Responses API 均正常：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy NO_PROXY='*' no_proxy='*' \
  CODEX_HOME=/root/.codex \
  /root/.vscode-server/extensions/openai.chatgpt-*/bin/linux-*/codex \
  exec --skip-git-repo-check 'Reply with exactly OK.'
```

返回 `OK` 且退出码为 `0`，说明 API 网关、API Key、模型和流式请求均正常；若 VS Code 仍失败，应优先检查插件重启后的 app-server 环境，而不是重复修改模型或 API Key。

## 本次环境结果

- API 网关 `100.94.44.45:8080`：可连接。
- 失效代理 `100.97.129.55:7890`：不可连接。
- 修复后 app-server：不再包含 `HTTP_PROXY`、`HTTPS_PROXY` 等代理变量。
- 最小 Codex 请求：返回 `OK`，退出码为 `0`。
