import json

def transform_data(input_file, output_file):
    # 读取原始数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []
    for item in data:
        # 过滤掉 duration >= 1500 的
        if item.get("duration", float("inf")) >= 1500:
            continue

        # 构建新结构
        new_item = {
            "key": item["id"],  # 用 id 作为 key
            "video_id": item["video_id"],
            "question": item["question"],
        }

        # 添加 candidates 作为 answer_choice
        for i, choice in enumerate(item.get("candidates", [])):
            new_item[f"answer_choice_{i}"] = choice

        # 添加正确答案
        new_item["answer_id"] = item["correct_choice"]
        new_item["answer"] = item["candidates"][item["correct_choice"]]

        output.append(new_item)

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    transform_data("/home/cyixiao/Project/videollm/LongVideoBench/lvb_val.json", "output.json")