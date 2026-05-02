# test.py
import os
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from model import create_model
from config import Config

# ---------------------------- 配置 ----------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './checkpoints/best_model.pth'      # 训练得到的最佳模型
TEST_IMG_DIR = './test_images'                   # 存放测试图片的文件夹
LABELS_JSON = os.path.join(TEST_IMG_DIR, 'labels.json')   # 可选：真实标签映射文件
OUTPUT_VIS = './visualization.png'               # 输出的可视化网格图
OUTPUT_TXT = './prediction_results.txt'          # 详细文本结果

# ---------------------------- 预处理 ----------------------------
def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# ---------------------------- 加载类别名称 ----------------------------
def get_food101_classes():
    """获取 Food-101 的 101 个类别名称"""
    try:
        from torchvision import datasets
        # 临时获取类别（会触发一次 tiny 下载，仅元数据）
        tmp = datasets.Food101(root=Config.data_root, split='test', download=True)
        return tmp.classes
    except Exception as e:
        print(f"从 torchvision 加载类别失败: {e}")
        # 备用：从 GitHub 读取
        import requests
        url = "https://raw.githubusercontent.com/Documents/Food101/master/meta/classes.txt"
        resp = requests.get(url, timeout=10)
        return [line.strip() for line in resp.text.splitlines()]

# ---------------------------- 加载模型 ----------------------------
def load_model(num_classes):
    model = create_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

# ---------------------------- 预测单张图片 ----------------------------
def predict_image(model, img_path, transform):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = prob.argmax()
        pred_prob = prob[pred_idx]
    return pred_idx, pred_prob, img

# ---------------------------- 可视化网格 ----------------------------
def draw_predictions_grid(image_paths, pred_labels, pred_probs, true_labels, class_names):
    """
    绘制网格图，每张图片下方显示预测标签（概率）和真实标签（若有）。
    true_labels 可以为 None 或字典。
    """
    n = len(image_paths)
    cols = min(5, n)               # 每行最多5张
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    else:
        axes = axes.reshape(rows, cols)

    correct_count = 0
    for idx, (ax_row, img_path) in enumerate(zip(axes.flat, image_paths)):
        img = plt.imread(img_path)
        pred_name = class_names[pred_labels[idx]]
        pred_prob = pred_probs[idx]
        true_name = true_labels.get(os.path.basename(img_path)) if true_labels else None

        # 显示图片
        ax_row.imshow(img)
        ax_row.axis('off')

        # 构建标题文本
        if true_name is not None:
            is_correct = (true_name == pred_name)
            if is_correct:
                correct_count += 1
                color = 'green'
                title = f"True: {true_name}\nPred: {pred_name} ({pred_prob:.2f})"
            else:
                color = 'red'
                title = f"True: {true_name}\nPred: {pred_name} ({pred_prob:.2f})"
            # 添加边框
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax_row.transAxes,
                                     linewidth=3, edgecolor=color, facecolor='none')
            ax_row.add_patch(rect)
        else:
            title = f"Pred: {pred_name} ({pred_prob:.2f})"

        ax_row.set_title(title, fontsize=9)

    # 隐藏多余的子图（当图片数不能填满网格时）
    for ax in axes.flat[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_VIS, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化图片已保存至 {OUTPUT_VIS}")

    if true_labels is not None:
        accuracy = correct_count / n
        print(f"准确率 (Top-1): {accuracy:.2%} ({correct_count}/{n})")
        return accuracy
    else:
        return None

# ---------------------------- 主流程 ----------------------------
def main():
    print("1. 加载类别名称...")
    class_names = get_food101_classes()
    num_classes = len(class_names)
    print(f"   Food-101 类别数: {num_classes}")

    print("2. 加载模型...")
    model = load_model(num_classes)

    print("3. 加载图片...")
    valid_ext = ('.jpg', '.jpeg', '.png')
    img_paths = [os.path.join(TEST_IMG_DIR, f) for f in os.listdir(TEST_IMG_DIR)
                 if f.lower().endswith(valid_ext)]
    img_paths.sort()  # 保证顺序一致
    if not img_paths:
        print(f"错误: 目录 {TEST_IMG_DIR} 中没有找到图片文件")
        return
    print(f"   发现 {len(img_paths)} 张图片")

    # 读取真实标签（如果存在 labels.json）
    true_labels = None
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, 'r', encoding='utf-8') as f:
            true_labels = json.load(f)
        print(f"   已加载真实标签映射，共 {len(true_labels)} 项")
    else:
        print("   未找到 labels.json，将只显示预测结果（不计算准确率）")

    # 预测所有图片
    transform = get_transform()
    pred_labels = []
    pred_probs = []
    pil_images = []   # 用于可能的高级显示，但这里直接使用原图路径即可
    for img_path in img_paths:
        pred_idx, pred_prob, _ = predict_image(model, img_path, transform)
        pred_labels.append(pred_idx)
        pred_probs.append(pred_prob)
        print(f"   {os.path.basename(img_path)} -> {class_names[pred_idx]} ({pred_prob:.4f})")

    # 保存详细文本结果
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("Food-101 自定义图片预测结果\n")
        f.write("=" * 60 + "\n")
        for i, img_path in enumerate(img_paths):
            f.write(f"图片: {os.path.basename(img_path)}\n")
            f.write(f"  预测类别: {class_names[pred_labels[i]]} (置信度: {pred_probs[i]:.4f})\n")
            if true_labels:
                true = true_labels.get(os.path.basename(img_path))
                if true:
                    correct = (true == class_names[pred_labels[i]])
                    f.write(f"  真实类别: {true}  {'✓' if correct else '✗'}\n")
            f.write("\n")
        if true_labels:
            acc = sum(1 for i, p in enumerate(img_paths)
                      if true_labels.get(os.path.basename(p)) == class_names[pred_labels[i]]) / len(img_paths)
            f.write(f"\n总体准确率: {acc:.2%} ({int(acc*len(img_paths))}/{len(img_paths)})\n")
    print(f"详细结果已保存至 {OUTPUT_TXT}")

    # 生成可视化网格图
    print("4. 生成可视化网格图...")
    draw_predictions_grid(img_paths, pred_labels, pred_probs, true_labels, class_names)

    print("完成！")

if __name__ == '__main__':
    main()