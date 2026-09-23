# 本地知识库问答（Local RAG QA）

一个完全跑在本机的 RAG 问答系统：上传 PDF → 自动转 Markdown → 切块向量化 → 网页提问。
不依赖任何云端服务，文档、向量库、模型推理全在你自己的机器上。

## 功能

- **多格式入库**：PDF（`pymupdf4llm` 转 Markdown）、Markdown、TXT；`.docx/.doc` 需额外安装 `docling`
- **按标题切块**：用 Markdown 标题层级切，超长块递归细分，保留章节信息用于引用展示
- **本地向量检索**：`znbang/bge:small-zh-v1.5-f16` 嵌入 + ChromaDB 持久化
- **本地生成**：Ollama 跑 `qwen3.5:4b`，流式逐字输出
- **忠实作答**：只依据召回资料回答，资料不足时明确拒答而不是编造
- **进度可见**：上传后实时显示「格式转换 / 切块 / 向量化 n/N」
- **可中断**：生成过程中可以随时停止

## 架构流程

```
PDF ─ converter.py ─→ Markdown ─ chunker.py ─→ 切块
                                                │
                                      ollama embed（批处理）
                                                │
                                          ChromaDB 持久化
                                                │
提问 ─ 查询改写嵌入 ─→ 向量检索 Top-5 ─→ 拼 prompt ─→ ollama chat 流式生成 ─→ 网页
```

| 文件 | 职责 |
|---|---|
| `app.py` | FastAPI 服务：上传、状态轮询、检索问答、流式输出 |
| `converter.py` | 格式转换（PDF/MD/TXT/DOCX → Markdown） |
| `chunker.py` | Markdown 按标题切块 + 超长块递归细分 |
| `static/index.html` | 前端页面（原生 JS，无构建步骤） |

## 环境要求

- Python 3.11+
- [Ollama](https://ollama.com/download)
- 显存参考：6GB（RTX 3060 Laptop）实测可跑 `num_ctx=8192`，占用约 5.4GB

拉取模型：

```bash
ollama pull znbang/bge:small-zh-v1.5-f16
ollama pull qwen3.5:4b
```

## 安装

```bash
git clone <你的仓库地址>
cd <仓库目录>
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

服务起在 `http://127.0.0.1:8000`，脚本会自动打开浏览器。
先上传文档，等进度显示「可以提问了」，再输入问题。

## 关键配置（`app.py` 顶部）

```python
EMB_MODEL = "znbang/bge:small-zh-v1.5-f16"   # 嵌入模型
GEN_MODEL = "qwen3.5:4b"                     # 生成模型

THINK = False          # 关闭思考链，见下方「踩坑记录」
GEN_OPTIONS = {
    "num_ctx": 8192,       # 上下文窗口
    "num_predict": 512,    # 生成上限，必须设——不设=无限生成
    "temperature": 0.2,    # RAG 要事实性，模型自带默认 1.0 太高
    "top_p": 0.9,
    "repeat_penalty": 1.05,
}
KEEP_ALIVE = "30m"     # 模型常驻显存
EMB_BATCH  = 16        # 嵌入批大小
TOP_K      = 5         # 召回块数
```

## 踩坑记录

### 1. 思考型模型会导致「静默卡死」

`qwen3.5:4b` 的 `ollama show` 里 Capabilities 含 `thinking`，是混合推理模型。
默认开启思考链时，思考内容走**独立的 `thinking` 字段**，正文 `response` 全程为空串。

如果代码只读 `response`：

- 浏览器一个字都收不到
- 模型却在满速生成思考内容，直到吃满 `num_ctx` 才停
- 全程没有任何异常抛出，控制台不报错

表现就是「卡很久 → 显卡归零 → 页面空白 → 无任何报错」。

**解决**：调用时传 `think=False`，并始终显式设置 `num_predict`。
实测该模型 `think=True` 时 5 题有 3 题烧完 2048 token 仍在思考、正文返回空字符串，
不仅慢 20~30 倍，而且经常不收敛——RAG 场景不要开。

### 2. 切块长度要匹配嵌入模型的窗口

`bge-small-zh-v1.5` 最大输入 512 token，中文大致 1 字≈1 token。
块长超过约 450 字会被静默截断，导致块尾内容永远检索不到。
`chunker.py` 默认值按此设定（`max_len=450`、`chunk_size=320`）。

### 3. `num_ctx` 与显存

6GB 显存同时装下 4B 生成模型 + 嵌入模型后余量不多（系统桌面本身占约 2GB）。
跑模型时避免再开大型游戏或多标签浏览器，否则会溢出到内存或导致 CUDA 上下文丢失。

## 目录说明

```
.
├── app.py               # 服务入口
├── chunker.py           # 切块
├── converter.py         # 格式转换
├── requirements.txt
├── static/
│   └── index.html       # 前端页面
└── data/                # 运行时生成，已被 .gitignore 排除
    ├── db/              # ChromaDB 向量库
    └── <id>_<文件名>    # 上传的原始文档
```

> `data/` 里的内容全部可重建：重新上传文档即可。

## License

MIT
