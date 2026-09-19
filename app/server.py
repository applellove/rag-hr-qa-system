"""
对话服务API提供
"""
import json

import uvicorn
from fastapi import FastAPI,Request
from starlette.responses import RedirectResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from rag_chat import get_assistant

#创建FastAPI应用对象
app= FastAPI(title="智能问答系统",description="基于知识库及工具调用的问答系统")

#挂载静态文件目录
app.mount("/static",StaticFiles(directory="./static"),name="static")
@app.post("/chat")
async def chat_stream(request: Request):
    """接收页面请求和模型交互响应流式输出内容"""

    #get 请求获取参数
    # question = request.query_params.get("question","")
    #post请求获取参数
    body =await request.json()
    question = body.get("question")

    if not question:
        return {"error":"问题不能为空"}

    assistant = get_assistant()

    def generation():
        chunk_count=0
        for chunk in assistant.chat_stream(question):
            chunk_count+=1
            data = json.dumps({"content":chunk,"done":False},ensure_ascii=False)
            #流式输出固定格式data:跟上空格
            yield f"data: {data}\n\n"
        print(f"流式输出完成，共输出{chunk_count}个回答片段")
        #发送完成信号
        done_data =json.dumps({"content":"","done":True},ensure_ascii=False)
        yield f"data: {done_data}\n\n"
    return StreamingResponse(
        generation(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", #不进行缓存，直接将内容展示在页面
            "Connection": "keep-alive" #告知客户端和服务器一直保持连接，一个链接响应多次内容
        }
    )




@app.get("/")
async def root():
    return RedirectResponse("/static/chat.html")

if __name__=="__main__":
    print("="*50)
    print("启动人事智能问答系统")
    print("="*50)
    print("页面访问路径（直接使用）：http://localhost:8000/")
    print("接口测试路径：http://localhost:8000/chat?question=你的问题")

    uvicorn.run(app, host="0.0.0.0", port=8000)
