# app.py —— 本地 RAG 问答系统：总入口（启动服务 + 打开网页）
# 依赖模块：chunker.py（切块）/ converter.py（格式转换）
import os, sys, time, threading, traceback, json as jsonlib
import requests
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
import ollama, chromadb

from chunker import chunk_markdown        # 模块①：通用切块
from converter import to_markdown         # 模块②：通用转换

# ============================ 配置区 ============================
EMB_MODEL = "znbang/bge:small-zh-v1.5-f16"
GEN_MODEL = "qwen3.5:4b"

# bge 检索模型要求：查询侧加指令前缀，文档侧不加
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# ★★★ 这一块是解决"提问卡死 / 显卡空转"的核心 ★★★
# qwen3.5 是「思考型」模型（ollama show 里 Capabilities 带 thinking）。
# 默认它会把思考过程写进 thinking 字段，而正文 response 全程为空。
# 你的旧代码只取 response，于是浏览器一个字都收不到，
# 模型却在满速生成思考内容，直到吃满 num_ctx 才停下 —— 表现为"卡好久后显卡不工作、cmd 无报错"。
THINK = False          # 关掉思考链，RAG 场景不需要它
GEN_OPTIONS = {
    "num_ctx": 8192,       # 上下文窗口（6GB 显存实测可跑，16k 也能加载）
    "num_predict": 512,    # ★ 硬上限：不设 = 无限生成，会一路烧到吃满 num_ctx
    "temperature": 0.2,    # 模型自带默认 1.0，对事实问答太高，会胡编
    "top_p": 0.9,
    "repeat_penalty": 1.05,
}
KEEP_ALIVE = "30m"         # 模型常驻显存，避免每次提问重新加载
EMB_BATCH = 16             # 嵌入按批发送，比一块一请求快很多
TOP_K = 5                  # 召回块数
CTX_CHAR_BUDGET = 2400     # 拼进 prompt 的资料上限，防止撑爆上下文
HISTORY_TURNS = 2          # 携带的历史轮数
REQUEST_TIMEOUT = 300.0    # ★ HTTP 超时：没有它，后端一旦卡住前端会永久等待

SYSTEM_PROMPT = (
    "你是一个严谨的本地知识库问答助手。请遵守：\n"
    "1. 只依据【参考资料】作答，不得使用资料之外的知识或凭猜测补充。\n"
    "2. 资料不足以回答时，直接说明“根据已有资料无法回答该问题”，不要编造。\n"
    "3. 用简体中文回答，先给结论再给依据，控制在 200 字以内。\n"
    "4. 不要复述问题，不要输出思考过程。\n"
    "5. 用纯文本作答：需要分条时写 1. 2. 3.，不要使用 ** # ` 等 Markdown 标记。"
)
# ===============================================================

if getattr(sys, "frozen", False):
    BASE = os.path.dirname(sys.executable)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

app = FastAPI()
client = chromadb.PersistentClient(path=os.path.join(DATA, "db"))
jobs = {}

# ---------- ollama 客户端：带超时，避免无限阻塞 ----------
try:
    _ollama = ollama.Client(timeout=REQUEST_TIMEOUT)
    print(f"[init] ollama 客户端超时 = {REQUEST_TIMEOUT}s")
except TypeError:
    _ollama = ollama
    print("[init] 当前 ollama 库版本不支持 timeout 参数，使用默认客户端")


def _f(obj, *keys, default=None):
    """兼容 dict / pydantic 两种返回结构的安全取值，可多层取。"""
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def _ctl(payload: dict) -> str:
    """控制帧：以 @@ 开头的一行 JSON，前端单独解析，不会混进正文。

    注意开头必须带 "\n"：正文分片末尾不带换行，若控制帧直接贴上去，
    就会和正文粘成同一行，前端解析失败并把 JSON 原样显示给用户。
    """
    return "\n@@" + jsonlib.dumps(payload, ensure_ascii=False) + "\n"


def _chat(**kw):
    """按需传入 think 参数；老版本 ollama 库不支持时自动降级。"""
    try:
        return _ollama.chat(think=THINK, **kw)
    except TypeError:
        print("[warn] ollama 库不支持 think 参数：若该模型仍输出思考链，"
              "可用 ollama pull 升级，或改走 Modelfile 关闭思考")
        return _ollama.chat(**kw)


def check_env():
    try:
        requests.get("http://localhost:11434", timeout=2)
    except Exception:
        return "未检测到 Ollama，请先安装：https://ollama.com/download"
    names = [m["name"] for m in requests.get(
        "http://localhost:11434/api/tags", timeout=5).json().get("models", [])]
    missing = [m for m in [EMB_MODEL, GEN_MODEL] if m not in names]
    if missing:
        return "缺少模型：ollama pull " + " && ollama pull ".join(missing)
    return "ok"


def ingest_file(file_id, path):
    """后台线程：转换 → 切块 → 批量向量化。"""
    try:
        jobs[file_id] = {"status": "processing", "msg": "格式转换中…"}
        md = to_markdown(path)
        if not md or not md.strip():
            raise ValueError("转换结果为空（可能是扫描版 PDF，需要先做 OCR）")

        jobs[file_id] = {"status": "processing", "msg": "切块中…"}
        chunks = chunk_markdown(md)
        if not chunks:
            raise ValueError("没有切出任何有效文本块，请确认文档有可提取的文字")

        col = client.get_or_create_collection(f"doc_{file_id}")
        total = len(chunks)
        for start in range(0, total, EMB_BATCH):
            batch = chunks[start:start + EMB_BATCH]
            res = _ollama.embed(
                model=EMB_MODEL,
                input=[c["text"] for c in batch],
                keep_alive=KEEP_ALIVE,
            )
            col.add(
                ids=[f"{file_id}_{start + i}" for i in range(len(batch))],
                embeddings=_f(res, "embeddings", default=[]),
                documents=[c["text"] for c in batch],
                metadatas=[c["meta"] or {"src": "doc"} for c in batch],
            )
            jobs[file_id] = {"status": "processing",
                             "msg": f"向量化 {min(start + EMB_BATCH, total)}/{total}"}

        jobs[file_id] = {"status": "done", "msg": f"完成，共 {total} 块"}
        print(f"[ingest] {file_id} 入库完成：{total} 块")
    except Exception as e:
        traceback.print_exc()                      # ← 让异常出现在 cmd 里
        jobs[file_id] = {"status": "failed", "msg": f"{type(e).__name__}: {e}"}


def warmup():
    """启动时预加载两个模型，否则第一次提问要等模型载入显存。"""
    try:
        print("[warmup] 预加载模型…")
        t0 = time.time()
        _ollama.embed(model=EMB_MODEL, input="预热", keep_alive=KEEP_ALIVE)
        list(_chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={**GEN_OPTIONS, "num_predict": 1},
            keep_alive=KEEP_ALIVE,
            stream=True,
        ))
        print(f"[warmup] 完成，耗时 {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[warmup] 跳过（不影响使用）：{type(e).__name__}: {e}")


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


@app.get("/env")
def env():
    return {"status": check_env()}


@app.post("/upload")
def upload(file: UploadFile):
    file_id = os.urandom(4).hex()
    path = os.path.join(DATA, f"{file_id}_{file.filename}")
    with open(path, "wb") as f:
        f.write(file.file.read())
    jobs[file_id] = {"status": "processing", "msg": "排队中"}
    threading.Thread(target=ingest_file, args=(file_id, path), daemon=True).start()
    return {"file_id": file_id, "filename": file.filename}


@app.get("/status/{file_id}")
def status(file_id: str):
    return jobs.get(file_id, {"status": "unknown", "msg": ""})


@app.post("/ask")
def ask(q: dict):
    question = (q.get("text") or "").strip()
    file_id = q.get("file_id") or ""
    history = q.get("history") or []

    def stream():
        t0 = time.time()
        try:
            if not question or not file_id:
                yield _ctl({"error": "缺少 file_id 或问题为空"})
                return

            # ---- 1. 检索 ----
            yield _ctl({"status": "retrieving"})     # 立刻回一帧，界面上马上有反馈
            try:
                col = client.get_collection(f"doc_{file_id}")
            except Exception:
                yield _ctl({"error": "找不到该文档的索引，请重新上传文档"})
                return

            emb = _f(_ollama.embed(model=EMB_MODEL,
                                   input=QUERY_PREFIX + question,
                                   keep_alive=KEEP_ALIVE), "embeddings")
            hits = col.query(query_embeddings=emb, n_results=TOP_K,
                             include=["documents", "metadatas"])
            docs = (hits.get("documents") or [[]])[0]
            metas = (hits.get("metadatas") or [[]])[0]
            if not docs:
                yield _ctl({"error": "没有检索到相关内容"})
                return

            refs = []
            for m in metas:
                s = (m or {}).get("section") or (m or {}).get("chapter") or ""
                if s and s not in refs:
                    refs.append(s)
            yield _ctl({"refs": refs, "hits": len(docs),
                        "t_retrieve": round(time.time() - t0, 2)})

            # ---- 2. 组装 prompt（带字符预算，防止撑爆上下文）----
            picked, used = [], 0
            for d in docs:
                if used + len(d) > CTX_CHAR_BUDGET:
                    break
                picked.append(d)
                used += len(d)
            context = "\n\n---\n\n".join(picked)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-HISTORY_TURNS:]:
                if h.get("q") and h.get("a"):
                    messages.append({"role": "user", "content": h["q"]})
                    messages.append({"role": "assistant", "content": h["a"]})
            messages.append({
                "role": "user",
                "content": f"【参考资料】\n{context}\n\n【问题】\n{question}",
            })

            # ---- 3. 生成（流式）----
            t_gen = time.time()
            first_tok = None
            chars = 0
            noted_thinking = False

            for part in _chat(model=GEN_MODEL, messages=messages,
                              options=GEN_OPTIONS, keep_alive=KEEP_ALIVE,
                              stream=True):
                msg = _f(part, "message") or part

                # 安全网：万一 think=False 没生效，思考内容单独提示，
                # 绝不再出现"后端在跑、前端一片空白"的静默卡死。
                if _f(msg, "thinking"):
                    if not noted_thinking:
                        noted_thinking = True
                        yield _ctl({"status": "thinking"})
                    continue

                piece = _f(msg, "content") or ""
                if piece:
                    if first_tok is None:
                        first_tok = time.time()
                        print(f"[ask] 首 token 延迟 {first_tok - t_gen:.1f}s")
                    chars += len(piece)
                    yield piece

            dt = time.time() - t_gen
            print(f"[ask] 检索 {t_gen - t0:.1f}s | 生成 {chars} 字/{dt:.1f}s "
                  f"({chars / dt if dt else 0:.1f} 字/秒) | 总计 {time.time() - t0:.1f}s")
            yield _ctl({"done": True, "chars": chars,
                        "seconds": round(time.time() - t0, 2)})

        except Exception as e:
            traceback.print_exc()                    # ← 关键：错误必须打到 cmd
            yield _ctl({"error": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    threading.Thread(target=warmup, daemon=True).start()
    threading.Timer(1.5, lambda: os.system("start http://localhost:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
