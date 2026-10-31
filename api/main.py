# api/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ragdn.pipeline import RAG  # load chunks, index, embedder, reranker, manifest

from openai import APITimeoutError, APIConnectionError
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAG.load()          # load model MỘT lần lúc khởi động
    yield

app = FastAPI(title="RAG Luật Doanh nghiệp", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:7860"], allow_methods=["*"], allow_headers=["*"])

class Q(BaseModel):
    question: str = Field(min_length=3, max_length=500)

@app.get("/health")
def health(request: Request):
    return {"status": "ok", "corpus_version": request.app.state.rag.manifest["corpus_version"]}

@app.get("/documents")
def documents(request: Request):        # danh sách văn bản đã nạp cho UI
    return [{k: d.get(k) for k in ("doc_id", "ten_vb", "so_hieu", "ngay_hieu_luc", "hop_nhat_den")}
            for d in request.app.state.rag.manifest["documents"]]

@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, q: Q):
    try:
        r = request.app.state.rag.answer(q.question)
    except (TimeoutError, APITimeoutError):
        raise HTTPException(503, "Dịch vụ LLM quá thời gian, vui lòng thử lại.")
    except APIConnectionError:
        raise HTTPException(503, "Không thể kết nối tới dịch vụ LLM, vui lòng kiểm tra lại.")
    return {
        "answer": r["answer"], "refused": r["refused"], "disclaimer": r.get("disclaimer"),
        "sources": [{"chunk_id": c["chunk_id"], "ten_vb": c["ten_vb"], "dieu": c["dieu"],
                     "khoan": c["khoan"], "page": c["page"], "hieu_luc": c["hieu_luc"],
                     "sua_doi": c["sua_doi"], "text": c["text"]} for c in r["sources"]],
        "corpus_version": request.app.state.rag.manifest["corpus_version"],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>RAG Chatbot Luật Doanh nghiệp</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; display: flex; justify-content: center; }
            .chat-container { width: 100%; max-width: 860px; background: #1e293b; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); overflow: hidden; display: flex; flex-direction: column; height: 90vh; }
            .header { background: #334155; padding: 16px 20px; font-weight: bold; border-bottom: 1px solid #475569; display: flex; justify-content: space-between; align-items: center; }
            .messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; scroll-behavior: smooth; }
            .msg { padding: 14px 18px; border-radius: 8px; max-width: 88%; line-height: 1.6; font-size: 15px; word-break: break-word; }
            .user { background: #2563eb; align-self: flex-end; color: white; }
            .bot { background: #334155; align-self: flex-start; border: 1px solid #475569; width: 100%; max-width: 92%; }
            
            /* Thẻ trích dẫn bấm được trong câu trả lời */
            .cite-link {
                color: #38bdf8;
                background: rgba(56, 189, 248, 0.15);
                border: 1px solid rgba(56, 189, 248, 0.4);
                padding: 1px 6px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 600;
                font-size: 0.9em;
                margin: 0 2px;
                display: inline-block;
                transition: all 0.2s ease;
            }
            .cite-link:hover {
                background: #38bdf8;
                color: #0f172a;
                transform: translateY(-1px);
            }

            /* Khu vực danh sách nguồn trích dẫn */
            .sources-container {
                margin-top: 14px;
                padding-top: 10px;
                border-top: 1px dashed #475569;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .sources-title {
                font-size: 13px;
                font-weight: 700;
                color: #94a3b8;
                margin-bottom: 2px;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .source-item {
                background: #0f172a;
                border: 1px solid #475569;
                border-radius: 6px;
                overflow: hidden;
                transition: border-color 0.3s ease, box-shadow 0.3s ease;
            }
            .source-item.highlighted {
                border-color: #38bdf8;
                box-shadow: 0 0 14px rgba(56, 189, 248, 0.7);
            }
            .source-header {
                padding: 9px 12px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                cursor: pointer;
                user-select: none;
                font-size: 13px;
                background: rgba(255, 255, 255, 0.02);
            }
            .source-header:hover {
                background: rgba(255, 255, 255, 0.06);
            }
            .source-badge {
                font-weight: bold;
                color: #38bdf8;
                background: rgba(56, 189, 248, 0.15);
                padding: 2px 7px;
                border-radius: 4px;
                border: 1px solid rgba(56, 189, 248, 0.3);
                font-size: 12px;
                margin-right: 8px;
                flex-shrink: 0;
            }
            .source-title-text {
                flex: 1;
                color: #e2e8f0;
                font-weight: 500;
            }
            .source-khoan {
                color: #facc15;
                font-size: 12px;
                font-weight: normal;
                margin-left: 4px;
            }
            .source-page {
                color: #94a3b8;
                font-size: 11px;
                margin-left: 6px;
            }
            .toggle-icon {
                font-size: 11px;
                color: #94a3b8;
                padding-left: 10px;
                white-space: nowrap;
            }
            .source-body {
                display: none;
                padding: 12px;
                border-top: 1px solid #334155;
                background: #090d16;
                font-size: 12.5px;
                line-height: 1.6;
                color: #cbd5e1;
            }
            .source-body pre {
                white-space: pre-wrap;
                word-wrap: break-word;
                margin: 0;
                font-family: inherit;
            }
            .source-alert {
                color: #f87171;
                font-size: 12px;
                font-weight: bold;
                margin-bottom: 6px;
            }

            .input-box { display: flex; padding: 16px; background: #0f172a; gap: 10px; border-top: 1px solid #334155; }
            input { flex: 1; padding: 12px 16px; border-radius: 6px; border: 1px solid #475569; background: #1e293b; color: white; font-size: 15px; outline: none; }
            input:focus { border-color: #3b82f6; }
            button { background: #2563eb; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-weight: bold; }
            button:disabled { background: #475569; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="header">
                <span>⚖️ Tra cứu Luật Doanh nghiệp</span>
                <span style="font-size: 12px; color: #4ade80;">● Local Server Online</span>
            </div>
            <div class="messages" id="msgs">
                <div class="msg bot">Xin chào! Tôi có thể hỗ trợ bạn tra cứu các quy định về Luật Doanh nghiệp hiện hành. Hãy đặt câu hỏi cho tôi!</div>
            </div>
            <div class="input-box">
                <input type="text" id="q" placeholder="Ví dụ: Không đăng ký thay đổi nội dung đăng ký doanh nghiệp đúng hạn bị phạt bao nhiêu tiền?" onkeypress="if(event.key==='Enter') send()">
                <button id="btn" onclick="send()">Gửi</button>
            </div>
        </div>
        <script>
            let msgCounter = 0;

            function escapeHtml(str) {
                if (!str) return "";
                return String(str)
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }

            function toggleSource(msgId, idx) {
                const body = document.getElementById(`src-body-${msgId}-${idx}`);
                const icon = document.getElementById(`src-icon-${msgId}-${idx}`);
                if (!body) return;
                if (body.style.display === "block") {
                    body.style.display = "none";
                    if (icon) icon.innerText = "Chi tiết ▼";
                } else {
                    body.style.display = "block";
                    if (icon) icon.innerText = "Thu gọn ▲";
                }
            }

            function focusSource(msgId, idx) {
                const item = document.getElementById(`src-item-${msgId}-${idx}`);
                const body = document.getElementById(`src-body-${msgId}-${idx}`);
                const icon = document.getElementById(`src-icon-${msgId}-${idx}`);
                if (!item) return;

                if (body) {
                    body.style.display = "block";
                    if (icon) icon.innerText = "Thu gọn ▲";
                }

                item.scrollIntoView({ behavior: "smooth", block: "nearest" });
                item.classList.add("highlighted");
                setTimeout(() => {
                    item.classList.remove("highlighted");
                }, 2500);
            }

            async function send() {
                const inp = document.getElementById("q"), btn = document.getElementById("btn"), msgs = document.getElementById("msgs");
                const text = inp.value.trim();
                if (!text) return;
                inp.value = ""; inp.disabled = true; btn.disabled = true;
                msgs.innerHTML += `<div class="msg user">${escapeHtml(text)}</div>`;
                const botDiv = document.createElement("div");
                botDiv.className = "msg bot";
                botDiv.innerText = "Đang tra cứu cơ sở dữ liệu luật...";
                msgs.appendChild(botDiv);
                msgs.scrollTop = msgs.scrollHeight;
                
                try {
                    const res = await fetch("/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: text })
                    });
                    const data = await res.json();
                    
                    msgCounter++;
                    const currentMsgId = "msg-" + msgCounter;

                    let answerHtml = escapeHtml(data.answer).replace(/\\n/g, "<br>");
                    answerHtml = answerHtml.replace(/\\[(\\d+)\\]/g, (match, n) => {
                        return `<span class="cite-link" onclick="focusSource('${currentMsgId}', ${n})" title="Nhấp để xem nguồn [${n}]">[${n}]</span>`;
                    });

                    let html = `<div>${answerHtml}</div>`;

                    if (data.sources && data.sources.length) {
                        let sourcesHtml = `<div class="sources-container"><div class="sources-title">📚 Nguồn trích dẫn (${data.sources.length}):</div>`;
                        data.sources.forEach((s, i) => {
                            const n = i + 1;
                            const khoanText = (s.khoan && s.khoan.length) ? `<span class="source-khoan">(Khoản ${escapeHtml(s.khoan.join(', '))})</span>` : "";
                            const pageText = s.page ? `<span class="source-page">— Trang ${s.page}</span>` : "";
                            const repealedText = (s.hieu_luc === false) ? `<div class="source-alert">⚠️ ĐIỀU NÀY ĐÃ BỊ BÃI BỎ</div>` : "";
                            
                            sourcesHtml += `
                                <div class="source-item" id="src-item-${currentMsgId}-${n}">
                                    <div class="source-header" onclick="toggleSource('${currentMsgId}', ${n})">
                                        <div style="display:flex; align-items:center; flex:1; min-width:0;">
                                            <span class="source-badge">[${n}]</span>
                                            <span class="source-title-text">${escapeHtml(s.ten_vb)}, Điều ${escapeHtml(String(s.dieu))}${khoanText}${pageText}</span>
                                        </div>
                                        <span class="toggle-icon" id="src-icon-${currentMsgId}-${n}">Chi tiết ▼</span>
                                    </div>
                                    <div class="source-body" id="src-body-${currentMsgId}-${n}">
                                        ${repealedText}
                                        <pre>${escapeHtml(s.text)}</pre>
                                    </div>
                                </div>
                            `;
                        });
                        sourcesHtml += `</div>`;
                        html += sourcesHtml;
                    }
                    botDiv.innerHTML = html;
                } catch(e) {
                    botDiv.innerText = "Lỗi kết nối tới máy chủ local: " + e;
                }
                inp.disabled = false; btn.disabled = false; inp.focus();
                msgs.scrollTop = msgs.scrollHeight;
            }
        </script>
    </body>
    </html>
    """