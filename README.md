## 1. 运行

python main.py

## 2. 报错解决方案

### 2.1. AttributeError: module 'torch' has no attribute 'get_default_device'

降级 transformers 到 4.49.0
`pip install transformers==4.49.0`

### 2.2. 安装的 magic-pdf 始终是 0.6.1

确保 python 版本在 3.10 ～ 3.12 之间
