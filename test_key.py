import dashscope
from dashscope import Generation

dashscope.base_http_api_url = "https://ws-mfge2ikzdsgmopfv.cn-beijing.maas.aliyuncs.com"
# 替换成你完整的sk-ws密钥
MY_KEY = "sk-ws-H.EPEYYIE.qavx.MEYCIQD2TzYYgRCDeog4bDEmFFdHu51lzZw06BHBq80JikPfJQIhAIxmXahrJbA-95p_U8B_KEJIQRugZyhMkPRqZZpA_O0x"

resp = Generation.call(
    model="qwen3.7-flash-2026-07-15",
    api_key=MY_KEY,
    messages=[{"role":"user","content":"你好"}],
    result_format="message"
)
print("完整返回：",resp)
