from openai import OpenAI
import sys
import time

# === 配置 ===
client = OpenAI(
    api_key="ollama",                 # 可随便写，不会被校验
    base_url="http://127.0.0.1:8998/v1"
)
model_name = "SmolMind"               # 与服务端保持一致
stream = True                         # 是否使用流式输出
history_messages_num = 1              # 限定携带多少历史消息（Q/A对 * 3）

# === 会话初始化 ===
conversation_history = [
    {"role": "system", "content": "你是一个友好且聪明的AI助手。"}
]

print("💬 SmolMind Chat 正在运行，输入内容开始对话（输入 exit 退出）\n")

while True:
    try:
        query = input("[Q]: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 再见！")
            break
        if not query:
            continue

        conversation_history.append({"role": "user", "content": query})

        # === 调用模型 ===
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation_history[-history_messages_num:],  # 截取部分历史
            stream=stream,
        )

        # === 输出 ===
        print("[A]: ", end="", flush=True)
        assistant_res = ""

        if stream:
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    text = delta.content
                    print(text, end="", flush=True)
                    assistant_res += text
            print("\n")
        else:
            assistant_res = response.choices[0].message.content
            print(assistant_res + "\n")

        # === 保存历史 ===
        conversation_history.append({"role": "assistant", "content": assistant_res})

    except KeyboardInterrupt:
        print("\n🛑 已中断对话。")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 出错：{e}")
        time.sleep(1)
