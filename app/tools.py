"""
工具模块 - 用于封装挂载到模型的各种工具
"""
from  datetime import datetime
import json
import os

from dashscope import Generation
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

@tool
def get_current_datetime()->str:
    """
    获取当日期和时间
    当用户询问"今天几号"、"现在几点"、"当前时间"等问题的时候调用
    :return: 时间日期格式化之后的字符串
    """
    now = datetime.now()

    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool

def get_weather(city:str)->str:
    """
    查询某地的实时天气。比如青岛、上海、北京
    :param city: 中午的城市名称
    :return: 相应城市的天气和温度数据
    """
    try:
        import httpx

        resp = httpx.get(f"https://wttr.in/{city}",params={"format":"%C %t","lang":"zh"},timeout=5)
        return f"{city}当前天气: {resp.text}"
    except Exception :
        return  f"暂时查不到{city}的天气（网络可能受限）"

#工具列表
TOOLS = [
    get_current_datetime,
    get_weather
]


def decide_tool(question: str) -> tuple:
    """
    让模型根据问题判断是否需要调用工具。如果调用那个工具以及工具执行所需参数
    :param question: 用户问题
    :return: 工具名称,工具参数 (tool_name,param_dict) 或 None,None
    """
    tool_list = []
    for tool in TOOLS:
        tool_list.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args
        })

    prompt = f"""
        你是一个智能助手，根据用户问题来决定是否调用工具。
        可用工具:
        {json.dumps(tool_list, ensure_ascii=False, indent=2)}
        用户问题:{question}

        请严格以纯JSON的格式输出问题，不要包含任何其他描述性的文字:
        {{"tool":"工具名称","params":{{"参数名":"参数值"}} }}

        如果不需要使用任何工具，输出:
        {{"tool":"none","params":{{}} }}

        规则:
        1.只能选择一个最相关的工具
        2.参数从用户的问题中提取
        3.如果用户询问的是制度问题，不需要调用工具
        4.只输出JSON，不要有任何解释
    """
    resp = Generation.call(
        model=os.getenv("CHAT_MODEL"),
        api_key=os.getenv("API_KEY"),

        messages=[{"role": "user", "content": prompt}],
        result_format="message"
    )
    # 新加打印日志
    print("resp.output:",resp.output)


    result = resp.output.choices[0].message.content
    print(f"AI工具使用的原始返回:{result}")

    info = json.loads(result)
    tool_name = info.get("tool", "none")
    params = info.get("params", "none")

    if tool_name != "none":
        return tool_name, params

    return None, None

def execute_tool(tool_name:str, **kwargs)->str:
    """
    根据工具名称执行对应工具获取工具执行结果
    """
    for tool in TOOLS:
        if tool.name == tool_name:
            return tool.invoke(kwargs)
    return f"未找到可执行的工具：{tool_name}"

if __name__ == "__main__":
    # print(get_weather.invoke("东京"))
    # decide_tool("北京天气怎么样")
    tool_name,params =decide_tool("济南天气如何")
    if tool_name :
        print(execute_tool(tool_name, **params))