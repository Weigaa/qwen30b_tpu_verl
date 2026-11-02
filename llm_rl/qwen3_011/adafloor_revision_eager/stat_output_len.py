import json
import csv
import sys
from statistics import mean, median

def try_import_tiktoken():
    try:
        import tiktoken  # type: ignore
        return tiktoken
    except Exception:
        return None

def load_records(path: str):
    # 兼容两种：jsonl（每行一个对象） 或 单个json数组
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON 顶层不是 list，无法当作 records 处理")
            return data
        else:
            records = []
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"第 {ln} 行不是合法 JSON：{e}\n内容片段：{line[:200]}") from e
            return records

def main():
    if len(sys.argv) < 2:
        print("用法：python stat_output_len.py /path/to/your.jsonl")
        sys.exit(1)

    path = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) >= 3 else "output_lengths.csv"

    tiktoken = try_import_tiktoken()
    enc = None
    if tiktoken is not None:
        # 这里用 o200k_base 通用编码；如果你想严格对齐某个模型，用对应 encoding
        enc = tiktoken.get_encoding("o200k_base")

    records = load_records(path)

    rows = []
    missing = 0
    for i, obj in enumerate(records, start=1):
        if not isinstance(obj, dict) or "output" not in obj:
            missing += 1
            continue
        out = obj["output"]
        if out is None:
            missing += 1
            continue
        out = str(out)

        len_chars = len(out)
        len_bytes = len(out.encode("utf-8"))
        len_words = len(out.split())

        row = {
            "line_no": i,
            "len_chars": len_chars,
            "len_bytes_utf8": len_bytes,
            "len_words": len_words,
        }

        if enc is not None:
            row["len_tokens_tiktoken"] = len(enc.encode(out))

        rows.append(row)

    if not rows:
        print("没有统计到任何包含 output 的行。请确认文件格式/字段名。")
        sys.exit(2)

    # 写 CSV
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 终端摘要（以字符数为主）
    lens = [r["len_chars"] for r in rows]
    print(f"Records(含 output)：{len(rows)} 行；缺失/无 output：{missing} 行")
    print(f"len_chars: min={min(lens)}, max={max(lens)}, mean={mean(lens):.2f}, median={median(lens)}")

    # 如果有 token 也给一份摘要
    if "len_tokens_tiktoken" in rows[0]:
        tlens = [r["len_tokens_tiktoken"] for r in rows]
        print(f"len_tokens_tiktoken: min={min(tlens)}, max={max(tlens)}, mean={mean(tlens):.2f}, median={median(tlens)}")

    print(f"已输出明细到：{out_csv}")

if __name__ == "__main__":
    main()
