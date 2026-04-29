# HW1: Fashion-MNIST MLP with NumPy

本仓库实现了一个只依赖 NumPy 的两隐藏层 MLP，用于完成 Fashion-MNIST 十分类作业。代码没有使用 PyTorch、TensorFlow、JAX 等自动微分框架，而是在 `fashion_mlp/autodiff.py` 中手写线性层、激活函数、交叉熵和反向传播。

## 代码结构

- `fashion_mlp/data.py`: Fashion-MNIST 下载、IDX 解析、标准化、训练/验证/测试划分。
- `fashion_mlp/autodiff.py`: 手写参数对象、线性层、ReLU/Sigmoid/Tanh、Softmax Cross-Entropy 与反向传播。
- `fashion_mlp/model.py`: 两隐藏层 MLP 分类器，支持两层隐藏层大小和每层激活函数配置，支持模型权重保存/加载。
- `fashion_mlp/trainer.py`: SGD、学习率衰减、L2 正则化、训练循环、验证集最优权重保存。
- `scripts/hyperparam_search.py`: 网格搜索学习率、隐藏层大小、正则化强度和激活函数。
- `scripts/evaluate.py`: 加载最优权重，在测试集上输出 Accuracy 和 Confusion Matrix。
- `fashion_mlp/visualization.py`: Loss/Accuracy 曲线、第一层权重、混淆矩阵、错例可视化。
- `fashion_mlp/reporting.py`: 自动生成 PDF 实验报告。

## 环境依赖

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

依赖项：

- `numpy`
- `Pillow`
- `pandas`
- `reportlab`

## 训练

首次运行会自动下载 Fashion-MNIST 到 `data/fashion-mnist`。

```bash
python scripts/train.py --epochs 30 --hidden-dims 256,128 --activations relu,tanh --learning-rate 0.05 --lr-decay 0.95 --weight-decay 1e-4
```

训练结束后会在 `outputs/run_时间戳/` 中生成：

- `best_model.npz`: 验证集准确率最高的模型权重。
- `history.json` / `history.csv`: 每轮训练和验证指标。
- `test_metrics.json`: 测试集 Accuracy、Confusion Matrix、各类别准确率。
- `figures/`: Loss 曲线、Accuracy 曲线、第一层权重图、混淆矩阵、错例图。
- `HW1_Fashion_MNIST_Report.pdf`: 实验报告。

快速调试可使用小数据子集：

```bash
python scripts/train.py --epochs 3 --max-train 5000 --max-val 1000 --max-test 1000
```

`--hidden-dims 256,128` 表示两层隐藏层分别为 256 和 128 个神经元；`--activations relu,tanh` 表示第一层隐藏层使用 ReLU，第二层隐藏层使用 Tanh。也可以写成 `--activations relu`，表示两层都使用 ReLU。

## 测试

```bash
python scripts/evaluate.py --weights outputs/run_时间戳/best_model.npz
```

脚本会打印测试集 Accuracy 和 Confusion Matrix，并刷新 `test_metrics.json`。

## 超参数搜索

```bash
python scripts/hyperparam_search.py --epochs 8 --max-train 12000 --max-val 2000
```

默认搜索：

- learning rate: `0.1, 0.05, 0.02`
- hidden dims: `128,64; 256,128`
- weight decay: `0, 0.0001, 0.001`
- activations: `relu,relu; relu,tanh; tanh,tanh; sigmoid,sigmoid`

结果保存在 `outputs/hyperparam_search/hyperparam_results.csv` 和 `hyperparam_results.json`。

也可以自定义搜索范围：

```bash
python scripts/hyperparam_search.py --learning-rates 0.05,0.02 --hidden-dims "128,64;256,128" --weight-decays 0.0001,0.001 --activations "relu,relu;relu,tanh;sigmoid,sigmoid"
```
