# coding: utf-8
import com
import time

# 以下密钥信息从控制台获取   https://console.xfyun.cn/services/bm35
appid = "d6cc6d42"  # 填写控制台中获取的 APPID 信息
api_secret = "NTc5Nzc0MGQ3MGY5ZDhlMTI3MjIyNWY5"  # 填写控制台中获取的 APISecret 信息
api_key = "18cf6189bce88b14d2549ce84e3b2b72"  # 填写控制台中获取的 APIKey 信息
domain = "4.0Ultra"

Spark_url = "wss://spark-api.xf-yun.com/v4.0/chat"  # Max服务地址
text = [
    {"role": "system", "content": "你现在扮演一个问答评分助手，要求如下：1. 每次你输出一个问题后，在同一行或下一行紧跟一个特殊分隔符 <shut> 后直接输出一个数字评分。2. 数字评分基于上一轮对话内容的评价：- 如果上一轮内容既正确又优秀，输出数字 **2**；- 如果上一轮内容正确但表现一般，输出数字 **1**；- 如果上一轮内容错误，输出数字 **0**。3. 评分时只输出对应数字，不添加多余文字或解释开始的一次不用加入评分，评分由回答决定。4. 例子格式：- “中医是什么？” <shut>      中医很好。        “西瓜怎么吃最好吃？” `<shut>0    5. 每次你的评分判断应直接依据上一轮对话内容，并输出相应的数字评分。请严格按照以上要求生成内容。"}
]


def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text


def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length


def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text


if __name__ == '__main__':

    while (1):
        Input = input("\n" + "我:")
        question = checklen(getText("user", Input))
        com.answer = ""
        print("星火:", end="")
        com.main(appid, api_key, api_secret, Spark_url, domain, question)
        # print(SparkApi.answer)
        getText("assistant", com.answer)




