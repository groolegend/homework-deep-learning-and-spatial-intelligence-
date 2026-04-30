# HW1: 从零构建三层神经网络分类器（EuroSAT）

本项目使用 `numpy` 从零实现三层全连接神经网络（MLP），完成 EuroSAT 地表覆盖图像分类任务。

---

## 数据集

使用 EuroSAT RGB 数据集。

将数据放置在项目目录下：

```
git clone

EuroSAT_RGB/
  ├── AnnualCrop/
  ├── Forest/
  ├── Residential/
  └── ...
```


---

## 环境配置

```bash
uv venv hw1 --python 3.12 --managed-python --seed
source hw1/bin/activate
uv pip install numpy matplotlib
```

---

## 运行方式

### 训练模型

```bash
python main.py train
```

---

### 从 checkpoint 继续训练

```bash
python main.py train --ckpt checkpoints/best_single_strong.npz
```

---

### 测试模型

```bash
python main.py eval --ckpt checkpoints/best_single_strong.npz
```

---



---

## 模型

三层 MLP：

```
Input → Linear → ReLU → Linear → ReLU → Linear → Softmax
```

主要配置：

- Hidden dim: 1024 / 512  
- Dropout: 0.15  
- LayerNorm: True  
- Label smoothing: 0.05  

---

## 训练设置

- Optimizer: SGD + Momentum  
- Learning rate schedule: OneCycle  
- Batch size: 128  
- Epochs: 160  
- Early stopping: patience = 25  

---

## 资源


Checkpoint:  
https://drive.google.com/file/d/1ZXv4tK337yd3XqlosaHnmNReWe_H6cVd/view?usp=sharing
