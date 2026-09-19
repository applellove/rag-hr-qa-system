"""
基于RAG的知识库问答系统 - 支持多文档相似度检索
"""
import os

from dashscope import Generation
from dashscope.finetune.customize_types import DashScopeBase
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from openai import embeddings

from app.chat import storage_file, add_chat_history
from chat import chat_history,save_to_file,load_to_file
from tools import execute_tool,decide_tool
storage_file = "rag_chat.json"
#向量数据存放位置
VECTOR_DB_PATH ="./chroma_db"

load_dotenv()

#文档存放路径
DOCS_DIR = "./docs"

#支持加载的文档及其对应的类加载器
LOAD_MAPPING ={
    ".pdf":PyPDFLoader,
    ".txt":TextLoader,
    ".docx":Docx2txtLoader
}


class RagAssistant:
    """存有构建知识库及相似度检索的各个相关方法"""
    def __init__(self, docs_dir=DOCS_DIR):
        self.docs_dir = docs_dir #文档存放路径
        self.vectorstore =None #用于后续存储向量数据库对象

    def get_all_documents(self):
        """加载指定目录下所有支持的文档"""
        all_documents = []#保存所有文件的文档块对象

        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir) #创建目录
            print(f"已创建目录{self.docs_dir}，请将文档放入后重新运行")
            return
        for file_name in os.listdir(self.docs_dir):
            file_path = os.path.join(self.docs_dir, file_name)#文件完整路径
            ext = os.path.splitext(file_name)[1]  #文件后缀名

            if ext in LOAD_MAPPING:
                print(f"加载文档{file_name}")
                loader_class = LOAD_MAPPING.get(ext) #通过后缀名获取加载器类型

                #创建加载器对象，txt类型的加载器对象需要指定编码格式

                if ext == ".txt":
                    loader = loader_class(file_path,encoding="utf-8")
                else:
                    loader = loader_class(file_path)
                #加载文档返回文档块列表
                documents = loader.load()

                # print(f"document对象结构:{documents[0]}")
                #更改文档片段来源为文件名称
                for doc in documents:
                    doc.metadata["source"] = file_name

                print(f"{file_name}文件共加载{len(documents)}页/段")

                #将加载的文档对象放到总列表中
                all_documents.extend(documents)
        return all_documents


    def build_knowledge_base(self):
        """构建向量知识库"""

        #所有文档片段
        all_docs = self.get_all_documents()
        print(f"共加载{len(all_docs)}个文档片段")

        print("正在进行文档分块")
        text_splitter = RecursiveCharacterTextSplitter( #文档切块对象
            separators=["\n\n","\n","。","；"],#优先按照这些符号进行分块
            chunk_size=500, #每块最大多少个字符，超过当前字符数的文档块需要再进行拆分
            chunk_overlap=100 #上下块重叠的字符数
        )

        chunks =text_splitter.split_documents(all_docs)
        print(f"共切分为{len(chunks)}个文档块")

        print("正在进行向量化......")
        embedding = DashScopeEmbeddings(
            dashscope_api_key=os.getenv("API_KEY"),
            model=os.getenv("EM_MODEL")
        )

        print("正在存入向量数据库......")
        self.vectorstore = Chroma.from_documents(
            documents=chunks, #文档块
            embedding=embedding, #向量模型
            persist_directory=VECTOR_DB_PATH #向量文件的存储位置
        )
        #向量数据持久化到文件中
        self.vectorstore.persist()
        print("√ 知识库构建完成")

    def search_documents(self,question,k=3):
        """
        从知识库中检索相似片段
        :param question: 用户问题
        :param k: 检索几个片段
        :return: 检索到的片段
        """
        docs = self.vectorstore.similarity_search(question,k=k)

        return docs

    def init(self):
        """构建还是加载向量知识库"""
        if not os.path.exists(VECTOR_DB_PATH) or not os.listdir(VECTOR_DB_PATH):
            self.build_knowledge_base() #构建知识库
        else:
            #读取向量知识库
            print("正在加载已有知识库......")
            embedding = DashScopeEmbeddings(
                dashscope_api_key=os.getenv("API_KEY"),
                model=os.getenv("EM_MODEL")
            )
            self.vectorstore = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embedding
            )

    def chat(self,prompt):
        """
        和模型对话
        :param prompt:用户提问的问题
        :return:
        """
        # mes = [
        #     {"role": "system", "content": SYSTEM_MESSAGE},  # 系统提示词
        #     {"role": "user", "content": prompt}  # 用户信息
        # ]
        #通过用户问题检索相关文档
        docs = self.search_documents(prompt)

        #将检索到的片段添加来源并转为字符串
        context_parts =[]
        for doc in docs:
            source = doc.metadata.get("source","未知来源")
            context_parts.append(f"【文档片段 - 来自《{source}》】\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        #将用户问题和文档中检索到的片段构成增强提示词
        user_prompt = f"""
            【知识库文档】
            {context}
            
            【员工问题】
            {prompt}
            
            请根据文档内容回答问题，并标注信息来源文件：
        """


        # 用户问题添加到对话历史
        chat_history = add_chat_history("user", user_prompt)

        # 和模型进行对话返回内容
        resps = Generation.call(
            api_key=os.getenv("API_KEY"),
            model=os.getenv("CHAT_MODEL"),
            messages=chat_history,
            result_format="message",  # 返回值类型
            stream=True,  # 开启流式输出
            incremental_output=True  # 开启增量输出
        )
        full_answer = ""  # 拼接完整ai回复的变量

        for resp in resps:
            if resp.status_code == 200:
                result = resp.output.choices[0].message.content  # 模型的输出内容
                print(result, end="", flush=True)  # end = "":每输出回答片段的拼接内容 flush:强制刷新 不要缓存
                full_answer += result
            else:
                print(f"错误：{resp}")
        add_chat_history("assistant", full_answer)  # 将模型的回复内容(assistant)放到对话历史
        save_to_file(storage_file,chat_history)  # 传入已经更改添加用户对话记录的chat_history

    def chat_stream(self, prompt):
        """
        和模型对话
        :param prompt:用户提问的问题
        :return:
        """
        # mes = [
        #     {"role": "system", "content": SYSTEM_MESSAGE},  # 系统提示词
        #     {"role": "user", "content": prompt}  # 用户信息
        # ]
        #根据用户问题判断是否使用工具
        tool_name,params = decide_tool(prompt)
        tool_result = None
        if tool_name:
            tool_result = execute_tool(tool_name, **params)
            print(f"工具执行返回{tool_result}")
        tool_info = f"\n【工具调用结果】\n{tool_result}\n" if tool_result else ""

        # 通过用户问题检索相关文档
        docs = self.search_documents(prompt)

        # 将检索到的片段添加来源并转为字符串
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(f"【文档片段 - 来自《{source}》】\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # 将用户问题和文档中检索到的片段构成增强提示词
        user_prompt = f"""
                {tool_info}
               【知识库文档】
               {context}

               【员工问题】
               {prompt}

               请根据文档内容回答问题，并标注信息来源文件：
           """

        # 用户问题添加到对话历史
        chat_history = add_chat_history("user", user_prompt)

        # 和模型进行对话返回内容
        resps = Generation.call(
            api_key=os.getenv("API_KEY"),
            model=os.getenv("CHAT_MODEL"),
            messages=chat_history,
            result_format="message",  # 返回值类型
            stream=True,  # 开启流式输出
            incremental_output=True  # 开启增量输出
        )
        full_answer = ""  # 拼接完整ai回复的变量

        for resp in resps:
            if resp.status_code == 200:
                result = resp.output.choices[0].message.content  # 模型的输出内容
                # print(result, end="", flush=True)  # end = "":每输出回答片段的拼接内容 flush:强制刷新 不要缓存
                yield  result
                full_answer += result
            else:
                # print(f"错误：{resp}")
                yield f"错误：{resp}"

        add_chat_history("assistant", full_answer)  # 将模型的回复内容(assistant)放到对话历史
        save_to_file(storage_file, chat_history)  # 传入已经更改添加用户对话记录的chat_history

assistant =None #接收rag对象的变量

def get_assistant():
    """返回RagAssistant的单例对象"""
    global assistant

    if assistant is None:
        assistant = RagAssistant()
        assistant.init()
    return assistant

def main():
    print("="*50)
    print("人事问答智能助手")
    print("="*50)

    global chat_history #chat_history表示是全局变量

    chat_history=load_to_file(storage_file)

    rag = RagAssistant()
    rag.init()

    while True:
        #获取用户输入的问题
        user_input = input("\n用户:")

        if"/exit" == user_input:
            break

        print("\nAI助手:")
        rag.chat(user_input)
        print()

if __name__ == '__main__':
    main()

# if __name__ == "__main__":
#     rag = RagAssistant()
#     rag.init()
#     print(rag.search_documents("离职流程有哪些",3))