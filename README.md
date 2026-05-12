# CaneSegLab 项目交接说明

## 1. 这份代码有没有绝对路径

目前检查下来，代码里没有写死你电脑上的 `C:\...`、`D:\...` 这类绝对路径。

项目主要使用的是项目根目录下的相对路径，例如：

- `data/samples`
- `data/train`
- `caneseglab/web_static`

`start.bat` 里也用了 `%~dp0`，会先切换到它自己所在的目录再启动，所以把整个项目文件夹拷走后，一般不会因为路径变化而跑不起来。

需要注意的地方只有一个：

- 如果同学不是在项目根目录启动，而是在别的目录直接执行 `python main.py ...`，相对路径可能会找不到。

所以最稳妥的做法是：

- 先进入项目根目录再运行命令。
- 或者直接双击 `start.bat`。

## 2. 建议拷贝哪些文件

### 方案 A：只想让同学直接运行现成项目

至少拷贝这些：

- `main.py`
- `requirements.txt`
- `start.bat`
- `caneseglab/`
- `data/`
- `README.md`

这是最省事的方案。同学拿到后安装依赖就能直接启动，现有模型、示例图片、数据集也都还在。

### 方案 B：想尽量少拷贝，只保留运行必需内容

如果只是为了演示网页、查看现成结果、做 ONNX / TensorRT 推理，最少建议保留：

- `main.py`
- `requirements.txt`
- `start.bat`
- `caneseglab/`
- `data/train/`
- `data/samples/demo/`

如果还想在网页里选择别的数据集图片，再额外拷贝对应的数据集目录，例如：

- `data/samples/all/`
- `data/samples/zsl2/`
- `data/samples/gzy/`

### 方案 C：同学还要继续训练

如果同学还要重新训练模型，除了代码外，还要保留训练数据目录。当前项目里可直接用于训练的目录示例有：

- `data/samples/all/images`
- `data/samples/all/masks`
- `data/samples/zsl2/images`
- `data/samples/zsl2/masks`

如果同学还要从 Labelme 标注重新生成 mask，还要保留原始标注目录，例如：

- `data/samples/sugarcane`
- `data/samples/gzy/images`

## 3. 哪些文件不用拷贝

这些通常不用拷贝给同学：

- `.venv/`
- `__pycache__/`
- `*.pyc`

如果只是代码交接，`paper.md` 也不是运行必需文件，可以按需决定是否一起给。

## 4. 同学拿到后怎么启动

### 第一步：进入项目目录

先让同学把项目放到任意目录，例如：

```bat
D:\labelme
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

## 5. 常用命令

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

## 6. 交接时最稳妥的做法

如果你现在只是想让同学“拿到就能运行”，最简单的交接方式就是直接拷贝下面这些：

- `main.py`
- `requirements.txt`
- `start.bat`
- `caneseglab/`
- `data/`
- `README.md`

同时提醒同学：

- 不要在别的目录直接运行 `python main.py ...`
- 先进入项目根目录，或者直接双击 `start.bat`

这样基本就不会因为路径问题导致迁移后运行不了。
