from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _register_fonts() -> str:
    for font_path in [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\STSONG.TTF"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    ]:
        if not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("CJK-Regular", str(font_path)))
            return "CJK-Regular"
        except Exception:
            continue
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        return "Helvetica"


def _load_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def build_report(
    run_dir: str | Path,
    output_pdf: str | Path | None = None,
    github_url: str = "TODO: replace with Public GitHub Repo URL",
    weights_url: str = "TODO: replace with Google Drive model weights URL",
) -> Path:
    run_path = Path(run_dir)
    output = Path(output_pdf) if output_pdf else run_path / "HW1_Fashion_MNIST_Report.pdf"
    font_name = _register_fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle", parent=styles["Title"], fontName=font_name, fontSize=22, leading=30))
    styles.add(ParagraphStyle(name="CNHeading", parent=styles["Heading2"], fontName=font_name, fontSize=15, leading=22))
    styles.add(ParagraphStyle(name="CNBody", parent=styles["BodyText"], fontName=font_name, fontSize=10.5, leading=17))

    history: List[Dict[str, float]] = _load_json(run_path / "history.json", [])
    config: Dict[str, object] = _load_json(run_path / "config.json", {})
    metrics: Dict[str, object] = _load_json(run_path / "test_metrics.json", {})
    search_results: List[Dict[str, object]] = _load_json(run_path / "hyperparam_results.json", [])
    best = max(history, key=lambda row: row.get("val_accuracy", 0.0)) if history else {}

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    story = []

    def heading(text: str) -> None:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(text, styles["CNHeading"]))

    def para(text: str) -> None:
        story.append(Paragraph(text, styles["CNBody"]))
        story.append(Spacer(1, 0.12 * cm))

    story.append(Paragraph("HW1: Fashion-MNIST MLP 图像分类实验报告", styles["CNTitle"]))
    para(f"GitHub Repo: {github_url}<br/>模型权重下载地址: {weights_url}")

    heading("1. 任务与方法")
    para(
        "本实验使用 NumPy 手工实现 MLP 分类器，在 Fashion-MNIST 上完成 10 类服装图像分类。"
        "网络结构为输入层 784 维、两个可配置隐藏层和 10 维输出层。代码中实现了线性层、"
        "ReLU/Sigmoid/Tanh 激活、Softmax 交叉熵、手写反向传播、SGD、学习率衰减和 L2 正则化，"
        "未使用 PyTorch、TensorFlow、JAX 等自动微分框架。"
    )

    heading("2. 数据处理")
    para(
        "原始 60,000 张训练图片按固定随机种子划分训练集和验证集，测试集使用官方独立 10,000 张图片。"
        "所有 28x28 灰度图先缩放到 [0,1]，再使用 Fashion-MNIST 常用均值 0.2860 和标准差 0.3530 标准化，"
        "最终展平为 784 维向量输入 MLP。"
    )

    heading("3. 实验设置")
    table_data = [
        ["超参数", "取值"],
        ["Hidden Dimensions", str(config.get("hidden_dims", ""))],
        ["Activations", str(config.get("activations", ""))],
        ["Epochs", str(config.get("epochs", ""))],
        ["Batch Size", str(config.get("batch_size", ""))],
        ["Learning Rate", str(config.get("learning_rate", ""))],
        ["LR Decay", str(config.get("lr_decay", ""))],
        ["Weight Decay", str(config.get("weight_decay", ""))],
    ]
    table = Table(table_data, colWidths=[5.5 * cm, 8.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef6")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(table)

    if best:
        para(
            f"最佳验证准确率出现在第 {int(best.get('epoch', 0))} 轮，"
            f"Validation Accuracy = {best.get('val_accuracy', 0):.4f}。"
            f"最终测试集 Accuracy = {metrics.get('test_accuracy', 0):.4f}。"
        )

    heading("4. 训练曲线")
    for image_name, caption in [
        ("loss_curve.png", "训练集与验证集 Loss 曲线"),
        ("accuracy_curve.png", "验证集 Accuracy 曲线"),
    ]:
        path = run_path / "figures" / image_name
        if path.exists():
            story.append(Image(str(path), width=15.0 * cm, height=9.2 * cm))
            para(caption)

    heading("5. 权重可视化与空间模式观察")
    path = run_path / "figures" / "first_layer_weights.png"
    if path.exists():
        story.append(Image(str(path), width=12.4 * cm, height=12.4 * cm))
    para(
        "第一层权重恢复为 28x28 图像后，可以观察到不少隐藏单元呈现局部明暗对比、条纹状纹理和弱边界响应，"
        "但整体空间模式仍明显比卷积网络学到的边缘/轮廓模板更分散。这说明 MLP 展平输入后会学习位置相关的低级视觉特征，"
        "对衣物边缘、鞋底方向和包袋块状纹理有一定响应，但缺少平移共享带来的清晰局部结构归纳偏置。"
    )

    heading("6. 测试结果与混淆矩阵")
    cm_path = run_path / "figures" / "confusion_matrix.png"
    if cm_path.exists():
        story.append(Image(str(cm_path), width=15.5 * cm, height=14.5 * cm))
    para(
        "混淆矩阵展示了真实类别与预测类别的对应关系。Fashion-MNIST 中 T-shirt/top、Shirt、Coat、Pullover "
        "在灰度形状上较为相似，通常更容易彼此混淆；Sandal、Sneaker、Ankle boot 也可能因轮廓接近而产生误判。"
    )

    heading("7. 错例分析")
    err_path = run_path / "figures" / "error_examples.png"
    if err_path.exists():
        story.append(Image(str(err_path), width=15.0 * cm, height=11.3 * cm))
    para(
        "错分样例中，常见原因包括衣物类别之间边界模糊、图像分辨率仅 28x28 导致细节缺失、服装姿态或局部纹理异常，"
        "以及 MLP 展平输入后缺少卷积网络对局部空间结构的归纳偏置。"
    )

    heading("8. 超参数查找")
    if search_results:
        rows = [["lr", "hidden_dims", "weight_decay", "activations", "best_val_acc"]]
        for row in search_results[:8]:
            rows.append(
                [
                    str(row.get("learning_rate", "")),
                    str(row.get("hidden_dims", row.get("hidden_dim", ""))),
                    str(row.get("weight_decay", "")),
                    str(row.get("activations", row.get("activation", ""))),
                    f"{float(row.get('best_val_accuracy', 0)):.4f}",
                ]
            )
        t = Table(rows, colWidths=[2.4 * cm, 2.4 * cm, 3.0 * cm, 3.0 * cm, 3.2 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef6")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(t)
    else:
        para("已提供 hyperparam_search.py 支持学习率、两层隐藏层大小、正则化强度和激活函数组合的网格搜索；运行后结果会写入报告。")

    story.append(PageBreak())
    heading("附录：复现命令")
    para("训练: python scripts/train.py --epochs 30 --hidden-dims 256,128 --activations relu,tanh --learning-rate 0.05 --weight-decay 1e-4")
    para("测试: python scripts/evaluate.py --weights outputs/run/best_model.npz")
    para("超参搜索: python scripts/hyperparam_search.py --epochs 8 --max-train 12000 --max-val 2000")

    doc.build(story)
    return output
