# chunker.py —— 通用切块模块：Markdown 按标题切，超长块递归细分
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DEFAULT_HEADERS = [
    ("#", "chapter"),
    ("##", "section"),
    ("###", "subsection"),
]

def chunk_markdown(md: str,
                   min_len: int = 30,
                   max_len: int = 450,
                   chunk_size: int = 320,
                   chunk_overlap: int = 60,
                   headers=None) -> list[dict]:
    """
    通用 Markdown 切块。
    输入: markdown 文本
    输出: [{"text": ..., "meta": {...}}, ...]
    
    min_len   : 小于这个长度的碎块丢弃（孤立标题等）
    max_len   : 大于这个长度的块交给递归切分器二次细分
    chunk_size: 二次切分的目标大小

    注意 max_len / chunk_size 是按嵌入模型窗口定的：
    bge-small-zh-v1.5 的最大输入是 512 token，中文大致 1 字≈1 token，
    所以块长超过 ~450 字就会被模型悄悄截断，检索质量下降。
    旧默认值 800/500 偏大（剩下 200 字的内容永远搜不到）。
    """
    if headers is None:
        headers = DEFAULT_HEADERS

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "，", ""],
    )

    chunks = []
    for c in splitter.split_text(md):
        text = c.page_content
        if len(text.strip()) < min_len:
            continue
        if len(text) > max_len:
            for s in fallback.split_text(text):
                if len(s.strip()) >= min_len:
                    chunks.append({"text": s, "meta": c.metadata})
        else:
            chunks.append({"text": text, "meta": c.metadata})
    return chunks