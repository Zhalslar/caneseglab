# CaneSegLab 项目文档

## 1. 怎么启动

### 第一步：进入项目目录

先把项目放到任意目录，例如：

```bat
D:\yourdir
```

然后进入项目根目录。

### 第二步：安装 Python 依赖

如果还没有虚拟环境，建议在项目根目录执行：

```bat
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

如果不想建虚拟环境，也可以直接：

```bat
python -m pip install -r requirements.txt
```

### 第三步：启动项目

最推荐的方法：

```bat
start.bat
```

它会自动：

- 切换到项目根目录
- 优先使用 `.venv\Scripts\python.exe`
- 打开浏览器
- 启动 Web 操作台

启动后默认访问地址：

- `http://127.0.0.1:5378`

### 也可以用命令行启动

```bat
python main.py web
```

## 2. 常用命令

### 1. 启动 Web 操作台

```bat
python main.py web
```

### 2. 标注转 mask

如果 JSON 标注文件就在 `data/samples/sugarcane` 目录下：

```bat
python main.py mask --input-dir data/samples/sugarcane --output-dir data/samples/sugarcane_masks
```

### 3. 训练

当前项目里比较规范的一组训练目录示例：

```bat
python main.py train --image-dir data/samples/all/images --mask-dir data/samples/all/masks
```

新训练的产物会保存到：

- `data/train/run-时间戳/`

常见文件包括：

- `best_model.pt`
- `model.onnx`
- `history.json`

补充说明：

- 当前仓库的 `data/train/` 根目录下已经放了现成的 `best_model.pt`、`model.onnx`、`model.trt`，所以不重新训练也可以直接做演示。

### 4. 导出 ONNX

如果 `data/train` 下已经有 `best_model.pt`：

```bat
python main.py export-onnx --output data/train/model.onnx
```

### 5. ONNX 推理验证

```bat
python main.py verify-onnx --image data/samples/demo/IMG_20260110_102451.jpg
```

### 6. TensorRT 推理

```bat
python main.py infer-trt --image data/samples/demo/IMG_20260110_102451.jpg
```

说明：

- 这一项要求环境里已经装好 TensorRT 相关依赖。
- 同时 `data/train` 下要有可用的 `model.trt`。
- 如果同学电脑没有 NVIDIA CUDA / TensorRT 环境，这一步可以先不做。
