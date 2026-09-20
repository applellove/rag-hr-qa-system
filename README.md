# rag-hr-qa-system
基于LangChain + FastAPI 实现的智能人事知识库RAG问答系统
> 本项目为个人实训学习项目，仅用于学习交流，非商用。项目核心代码由本人独立编写调试。


## 项目介绍
本项目是面向企业人事文档的智能问答系统，使用RAG检索增强生成技术，上传人事相关文档后，用户可以通过接口提问，系统基于知识库内容给出精准回答，减少大模型幻觉。

## 技术栈
- Python、FastAPI：后端流式问答接口
- LangChain：RAG流程编排，文档加载、文本分割、向量检索
- 向量库：Chroma
- LLM：通义千问API
- 前端：简单静态页面，SSE流式输出回答

## 功能特性
1. 支持知识库文档解析与向量化存储
2. 流式输出问答结果，提升交互体验
3. 基于文档内容回答，减少大模型幻觉
4. 环境变量管理密钥，.gitignore保护隐私信息

## 项目目录
```bash

rag-hr-qa-system/
├── app/
│   ├── chroma_db      # Chroma向量持久化存储目录
│   ├── docs           # 存放知识库文档
│   ├── static         # 前端静态页面
│   ├── chat.py
│   ├── rag_chat.py    # RAG核心逻辑
│   ├── server.py      # FastAPI服务入口
│   └── tools.py       # 工具调用模块
├── .env               # 存放API密钥（不上传git）
├── .gitignore
└── requirements.txt   # 项目依赖
```
## 🚀快速启动
1.安装项目依赖
```bash
pip install -r requirements.txt
```
2.在项目根目录新建 `.env` 文件，填入通义千问 API 密钥
```bash
API_KEY="你的API密钥"
```
3.启动 FastAPI 后端服务
```bash
uvicorn app.server:app --reload
```
4.访问本地静态页面，进行知识库问答
