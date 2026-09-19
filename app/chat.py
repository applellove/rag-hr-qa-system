"""
调用API和大模型进行对话，并进行对话记忆的存储
"""
import os
import json
from dashscope.aigc import Generation
from dotenv import load_dotenv

#设置模型的系统提示词规定模型的角色身份、功能列表、回答风格等
SYSTEM_MESSAGE="你是公司的人事小助手，专门回答请假、入离职、薪资、考勤等人事问题。回答要专业简洁，不知道的问题不要回答。"

#加载环境配置
load_dotenv()

#保存对话历史的列表
chat_history =[
    {"role":"system","content":SYSTEM_MESSAGE}
]

#定义持久化保存对话的文件
storage_file="chat.json"

def save_to_file(storage_file=storage_file,info=chat_history):
    """
    将对话持久化到文件中
    :param storage_file: 保存的文件名称默认是 chat.json中
    :param info: 需要持久化的数据
    :return:
    """
    try:
        with open(storage_file, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)#indent格式化缩进两个空格，更具有可读性
    except Exception as e:
        print(f"持久化保存失败{e}")

def load_to_file(storage_file=storage_file):
    """
    从指定文件中读取内容并放到列表中
    :param storage_file: 从哪个文件读取
    :return:读取到的内容
    """
    if not os.path.exists(storage_file):
        return chat_history
    try:
        with open(storage_file, "r", encoding="utf-8") as f:
            info = json.load(f)
            return info
    except Exception as e:
        print(f"内容加载失败：{e}")
        return info


def add_chat_history(role,content):
    """
    添加对话到对话历史列表
    :param role: 角色
    :param content: 内容
    :return:
    """
    chat_history.append(
        {"role":role,"content":content}
    )
    return chat_history



def chat(prompt):
    """
    和模型对话
    :param prompt:用户提问的问题
    :return:
    """
    mes=[
        {"role":"system","content":SYSTEM_MESSAGE},#系统提示词
        {"role":"user","content":prompt} #用户信息
    ]

    #用户问题添加到对话历史
    add_chat_history("user",prompt)

    #和模型进行对话返回内容
    resps= Generation.call(
        api_key=os.getenv("API_KEY"),
        model=os.getenv("CHAT_MODEL"),
        messages=chat_history,
        result_format="message", #返回值类型
        stream=True, #开启流式输出
        incremental_output=True #开启增量输出
    )
    full_answer ="" #拼接完整ai回复的变量

    for resp in resps:
        if resp.status_code == 200:
            result=resp.output.choices[0].message.content #模型的输出内容
            print(result,end="",flush=True) #end = "":每输出回答片段的拼接内容 flush:强制刷新 不要缓存
            full_answer += result
        else:
            print(f"错误：{resp}")
    add_chat_history("assistant",full_answer) #将模型的回复内容(assistant)放到对话历史
    save_to_file(info=chat_history) #传入已经更改添加用户对话记录的chat_history

def main():
    print("="*50)
    print("人事问答智能助手")
    print("="*50)

    global chat_history #chat_history表示是全局变量

    chat_history=load_to_file()

    while True:
        #获取用户输入的问题
        user_input = input("\n用户:")

        if"/exit" == user_input:
            break

        print("\nAI助手:")
        chat(user_input)
        print()

if __name__ == '__main__':
    main()
