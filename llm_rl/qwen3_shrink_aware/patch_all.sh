#!/bin/bash
# 移除 set -o pipefail，因为它可能导致 grep/sort 等管道命令非预期退出，
# 或者保留它但在循环结构上做调整。为了脚本鲁棒性，这里主要修改循环内部逻辑。

echo "Applying patches in numerical order..."

# 定义两个数组来记录成功和失败的补丁（可选，为了最后看报告）
failed_patches=()
success_count=0

# 使用进程替换 < <(...) 而不是管道 | while，
# 这样可以在循环结束后保留变量（failed_patches）的值
while IFS= read -r PATCH_FILE; do
    # skip empty lines
    [[ -z "$PATCH_FILE" ]] && continue
    PATCH_REL_PATH=$(realpath --relative-to=. "$PATCH_FILE")
    
    echo -n "Applying $PATCH_REL_PATH ... "

    # 尝试应用补丁
    # 2>&1 将错误输出捕获，以便如果失败了可以打印出来看原因
    OUTPUT=$(git apply --ignore-whitespace "$PATCH_REL_PATH" 2>&1)
    
    # 检查上一条命令的退出状态
    if [ $? -eq 0 ]; then
        echo "[SUCCESS]"
        ((success_count++))
    else
        # 核心修改：这里不再 exit 1，而是输出警告并继续
        echo "[SKIPPED]"
        echo "    /!\\ Warning: Failed to apply $PATCH_REL_PATH"
        echo "    /!\\ Reason: $OUTPUT" # 输出具体的 git 错误信息
        failed_patches+=("$PATCH_REL_PATH")
    fi
done < <(find ./patches -type f -name "*.patch" | sort -V)

echo "------------------------------------------------"
echo "Patching process finished."
echo "Total Success: $success_count"

# 如果有失败的补丁，列出来
if [ ${#failed_patches[@]} -gt 0 ]; then
    echo "Total Skipped: ${#failed_patches[@]}"
    echo "The following patches were NOT applied:"
    for p in "${failed_patches[@]}"; do
        echo " - $p"
    done
    # 可选：如果你希望即使有跳过，脚本最终也返回非0状态码给CI/CD，取消下面这行的注释
    # exit 1 
else
    echo "All patches applied successfully!"
fi