import os

def process_file(input_path, output_path, label_base_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            print(f"跳过空行: {line}")  # 调试：打印被跳过的行
            continue
        # 提取模型名
        model_path = parts[0]
        print(f"模型路径: {model_path}")  # 调试：打印模型路径
        model_name = model_path.replace('models\\', '').replace('_collision_detection.ply', '')
        print(f"模型名: {model_name}")  # 调试：打印模型名
        # 构造_label.txt路径
        label_txt_path = os.path.join(label_base_dir, model_name, f"{model_name}_label.txt")
        print(f"标签文件路径: {label_txt_path}")  # 调试：打印标签文件路径
        # 读取_label.txt的六个浮点数
        if os.path.exists(label_txt_path):
            print(f"找到标签文件: {label_txt_path}")  # 调试：标签文件存在
            with open(label_txt_path, 'r', encoding='utf-8') as lf:
                label_values = lf.read().strip().split()
                print(f"标签值: {label_values}")  # 调试：打印标签值
                label_values = label_values[:6]  # 只取前六个
        else:
            print(f"标签文件不存在，补零: {label_txt_path}")  # 调试：标签文件不存在
            label_values = ['0'] * 6  # 如果没有则补零
        # 追加到原行
        new_line = line.strip() + ' ' + ' '.join(label_values) + '\n'
        print(f"新行内容: {new_line}")  # 调试：打印新行内容
        new_lines.append(new_line)

    print(f"最终写入文件: {output_path}")  # 调试：打印输出文件路径
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"写入完成，共{len(new_lines)}行")  # 调试：打印写入行数

# 用法示例
label_base_dir = r'F:\orign'
process_file('test', 'test_with_label.txt', label_base_dir)
