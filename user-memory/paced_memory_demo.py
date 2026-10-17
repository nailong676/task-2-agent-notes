#!/usr/bin/env python3
"""限速版跨会话记忆演示（账号 3 RPM 约束）

复用 quickstart_user 的持久化记忆（Session 1 已写入 3 条笔记），
在新会话中验证：记忆召回 → 事实更新（搬家/换框架）→ 更新后召回。
每次 LLM 调用间隔 >=35s，429 时退避重试。
"""

import time
from dotenv import load_dotenv

load_dotenv()

from conversational_agent import ConversationalAgent, ConversationConfig
from background_memory_processor import BackgroundMemoryProcessor, MemoryProcessorConfig
from config import Config, MemoryMode
from memory_manager import create_memory_manager

PACE_SECONDS = 35   # 3 RPM -> 每轮 chat+process 约 2 次调用，间隔 35s 安全边际
RETRY_SECONDS = 40

user_id = "quickstart_user"
memory_mode = MemoryMode.NOTES


def call_with_retry(fn, label):
    for attempt in range(3):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print(f"  [rate-limit] {label} 第{attempt+1}次 429，{RETRY_SECONDS}s 后重试")
                time.sleep(RETRY_SECONDS)
            else:
                raise


def show_memory(tag):
    mm = create_memory_manager(user_id, memory_mode)
    print(f"\n===== {tag} =====")
    print(mm.get_context_string())


def main():
    Config.create_directories()
    agent = ConversationalAgent(
        user_id=user_id, memory_mode=memory_mode,
        config=ConversationConfig(enable_memory_context=True,
                                  enable_conversation_history=True),
        verbose=False)
    processor = BackgroundMemoryProcessor(
        user_id=user_id, memory_mode=memory_mode,
        config=MemoryProcessorConfig(conversation_interval=1,
                                     min_conversation_turns=1,
                                     output_operations=False),
        verbose=False)

    show_memory("记忆初始状态（Session 1 持久化的 3 条笔记）")

    agent.reset_session()
    print("\n🔄 新会话开始（跨会话测试）\n")

    rounds = [
        ("召回", "What do you remember about my work and preferences?"),
        ("更新1", "Actually, I moved from Beijing to Shanghai last month."),
        ("更新2", "I also switched from PyTorch to JAX for better performance."),
        ("验证", "Which city do I live in now, and which deep-learning framework do I currently use?"),
    ]

    for i, (tag, msg) in enumerate(rounds, 1):
        print(f"\n[Round {i} · {tag}] 👤 {msg}")
        response = call_with_retry(lambda: agent.chat(msg), f"round{i}-chat")
        print(f"🤖 {response}")

        processor.increment_conversation_count()
        results = call_with_retry(lambda: processor.process_recent_conversations(),
                                  f"round{i}-memory")
        ops = results.get("operations", [])
        if ops:
            print("📝 Memory Operations:")
            for op in ops:
                print(f"   - {op['action'].upper()}: {op.get('content', '')[:100]}")
        else:
            print("📝 无记忆更新")

        if i < len(rounds):
            print(f"  ...间隔 {PACE_SECONDS}s 遵守 3 RPM...")
            time.sleep(PACE_SECONDS)

    show_memory("最终记忆状态（应包含城市/框架的更新）")


if __name__ == "__main__":
    main()
