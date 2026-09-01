#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI聊天助手 - 纯requests实现，配置直接写在代码中
兼容所有OpenAI API格式的大模型服务
"""

import sys
import json
import requests
from typing import List, Dict

class AIChat:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o-mini",
        system_prompt: str = "你是一个有帮助、有创意、友好且专业的AI助手。请用简洁明了的语言回答问题。",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60
    ):
        """
        初始化AI聊天助手
        
        Args:
            api_key: API密钥（必填）
            base_url: API基础地址（必填）
            model: 使用的模型名称
            system_prompt: 系统提示词，定义AI的角色和行为
            temperature: 温度参数，控制回答的创造性(0-2)
            max_tokens: 最大生成token数
            timeout: 请求超时时间(秒)
        """
        # 配置API参数
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # 验证必要参数
        if not self.api_key:
            raise ValueError("API密钥不能为空，请在main函数中配置api_key")
        if not self.base_url:
            raise ValueError("API基础地址不能为空，请在main函数中配置base_url")
        
        # 确保base_url以/v1结尾
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"
        
        # 对话历史
        self.messages: List[Dict[str, str]] = []
        
        # 添加系统提示词
        if system_prompt:
            self.add_system_message(system_prompt)
            
        # 打印欢迎信息
        self._print_welcome()
    
    def add_system_message(self, content: str) -> None:
        """添加系统消息"""
        self.messages.append({"role": "system", "content": content})
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.messages.append({"role": "assistant", "content": content})
    
    def clear_history(self) -> None:
        """清空对话历史（保留系统提示词）"""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages
        print("\n✅ 对话历史已清空\n")
    
    def _print_welcome(self) -> None:
        """打印欢迎信息和使用说明"""
        print("=" * 60)
        print(f"🤖 AI聊天助手已启动")
        print(f"📦 当前模型: {self.model}")
        print(f"🌡️  温度: {self.temperature}")
        print(f"🔗 API地址: {self.base_url}")
        print("\n📝 使用说明:")
        print("  - 输入问题直接聊天")
        print("  - 输入 'clear' 或 '清空' 重置对话")
        print("  - 输入 'exit' 或 'quit' 或 '退出' 结束程序")
        print("  - 按 Ctrl+C 中断生成或退出")
        print("=" * 60 + "\n")
    
    def _parse_sse_chunk(self, line: bytes) -> Dict:
        """
        解析SSE流式响应的单个数据块
        
        Args:
            line: 原始字节行
            
        Returns:
            解析后的JSON数据，None表示空行或结束标记
        """
        line = line.decode("utf-8").strip()
        
        # 跳过空行
        if not line:
            return None
            
        # 跳过注释行
        if line.startswith(":"):
            return None
            
        # 处理数据行
        if line.startswith("data: "):
            data = line[6:]  # 去掉 "data: " 前缀
            
            # 检查是否是结束标记
            if data == "[DONE]":
                return None
                
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        
        return None
    
    def _stream_chat(self) -> str:
        """
        流式聊天，逐字打印回复
        
        Returns:
            完整的助手回复内容
        """
        full_response = ""
        
        # 构造请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 构造请求体
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        try:
            # 发送流式请求
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            print("\n🤖 助手: ", end="", flush=True)
            
            # 逐行处理流式响应
            for line in response.iter_lines():
                chunk = self._parse_sse_chunk(line)
                
                if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
            
            print("\n")  # 最后换行
            
        except KeyboardInterrupt:
            print("\n\n⚠️  生成已中断\n")
        except requests.exceptions.RequestException as e:
            print(f"\n\n❌ 请求错误: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                except:
                    print(f"响应内容: {e.response.text}")
            print()
        except Exception as e:
            print(f"\n\n❌ 未知错误: {str(e)}\n")
        
        return full_response
    
    def chat_loop(self) -> None:
        """启动交互式聊天循环"""
        while True:
            try:
                # 获取用户输入
                user_input = input("👤 你: ").strip()
                
                # 处理特殊命令
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "退出", "q"]:
                    print("\n👋 再见！")
                    break
                    
                if user_input.lower() in ["clear", "清空", "reset", "重置"]:
                    self.clear_history()
                    continue
                
                # 添加用户消息到历史
                self.add_user_message(user_input)
                
                # 获取流式回复
                assistant_response = self._stream_chat()
                
                # 添加助手回复到历史（如果有内容）
                if assistant_response:
                    self.add_assistant_message(assistant_response)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}\n")
                continue

def main():
    """主函数 - 所有配置都在这里修改"""
    # ====================== 配置区域开始 ======================
    # 请修改以下三个参数为你的实际配置
    API_KEY = "sk-1234"
    BASE_URL = "http://127.0.0.1:5172/v1"
    MODEL = "glm-5"
    
    # 可选配置
    SYSTEM_PROMPT = "你是一个有帮助、有创意、友好且专业的AI助手。请用简洁明了的语言回答问题。"
    TEMPERATURE = 0.7
    MAX_TOKENS = 4096
    TIMEOUT = 60
    # ====================== 配置区域结束 ======================
    
    try:
        # 初始化聊天助手
        chat = AIChat(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=TIMEOUT
        )
        
        # 启动聊天循环
        chat.chat_loop()
        
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

