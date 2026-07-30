#!/bin/sh

# Stable entrypoint for the VS Code Codex extension.
# Point chatgpt.cliExecutable at this file so extension upgrades keep using
# the local Tailscale HTTP proxy without patching each openai.chatgpt-* bundle.

if [ "${1:-}" = "app-server" ]; then
  export HTTP_PROXY="http://127.0.0.1:1056"
  export http_proxy="http://127.0.0.1:1056"
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="127.0.0.1,localhost"
  unset HTTPS_PROXY ALL_PROXY https_proxy all_proxy
fi

if [ -n "${CODEX_REAL:-}" ]; then
  exec "$CODEX_REAL" "$@"
fi

EXT_ROOT="${CODEX_VSCODE_EXTENSIONS_ROOT:-/root/.vscode-server/extensions}"

for ext_dir in $(ls -td "$EXT_ROOT"/openai.chatgpt-*-linux-* 2>/dev/null); do
  for candidate in \
    "$ext_dir/bin/linux-aarch64/codex-real" \
    "$ext_dir/bin/linux-aarch64/codex" \
    "$ext_dir/bin/linux-x64/codex-real" \
    "$ext_dir/bin/linux-x64/codex"
  do
    if [ -x "$candidate" ] && [ "$candidate" != "$0" ]; then
      exec "$candidate" "$@"
    fi
  done
done

echo "Unable to find bundled Codex CLI under $EXT_ROOT/openai.chatgpt-*-linux-*/bin/." >&2
exit 127
