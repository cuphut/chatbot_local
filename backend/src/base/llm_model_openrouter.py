import requests
import json
import os
from typing import Dict, List, Optional, Any, Union
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from dotenv import load_dotenv
import time
load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage

class OpenRouterClient:
    """
    Client để tương tác với OpenRouter API
    """
    def __init__(self, api_key:str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Khởi tạo OpenRouter 
        
        Args:
            api_key: API key của OpenRouter
            base_url: URL cơ sở cho API của OpenRouter
        """

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://watatek.com/", 
            "X-Title": "ChatBotAI" 
        }

    def generate(self, 
            model: str,
            prompt: Union[str, List[Dict[str, str]], Document, List[Document]],
            max_tokens: int = 1024,
            timeout: int = 60,
            **kwargs) -> Dict[str, Any]:
        
        if isinstance(prompt, Document):
            messages = [{"role": "user","content":prompt.page_content}]
        elif isinstance(prompt, list) and all(isinstance(p, Document) for p in prompt):
            messages = [{"role": "user","content":"\n\n".join(p.page_content for p in prompt)}]
        elif isinstance(prompt, str):
            messages = [{"role": "user","content":prompt}]
        elif isinstance(prompt, list) and all(isinstance(p, dict) for p in prompt):
            messages = prompt
        else:
            raise ValueError("Prompt phải là chuỗi, danh sách messages, Document hoặc danh sách Document")

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs
        }

        full_url = f"{self.base_url}/chat/completions"
        

        try:
            response = requests.post(
                full_url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error: {e}")
        
    
class OpenRouterRunnable(Runnable):
    def __init__(self, client: OpenRouterClient, model: str, max_tokens: int = 1024, **kwargs):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    def invoke(self, input: Any, config: Optional[Dict] = None) -> str:
        try:
            # Nếu input là ChatPromptValue, convert sang messages
            if hasattr(input, "to_messages"):
                messages_obj = input.to_messages()
                messages = [
                    {
                        "role": "system" if isinstance(msg, SystemMessage) else "user",
                        "content": msg.content
                    }
                    for msg in messages_obj
                ]
            else:
                raise ValueError("Invalid input format - expected ChatPromptValue with .to_messages() method")

            response = self.client.generate(
                model=self.model,
                prompt=messages,
                max_tokens=self.max_tokens,
                **self.kwargs
            )
            
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(f"API call failed: {str(e)}")


def get_openrouter_llm(model_name: str = "google/gemma-3-27b-it:free", 
                    api_key: Optional[str] = None,
                    max_tokens: int = 1024,
                    **kwargs) -> OpenRouterRunnable:
    """
    Tạo một client OpenRouter để sử dụng LLM
    
    Args:
        model_name: Tên mô hình trên OpenRouter
        api_key: API key cho OpenRouter (mặc định lấy từ biến môi trường)
        max_tokens: Số lượng token tối đa cho phản hồi
        **kwargs: Các tham số bổ sung cho API
        
    Returns:
        OpenRouterClient: Client đã cấu hình cho mô hình được chỉ định
    """
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("API key không được cung cấp và không tìm thấy trong biến môi trường OPENROUTER_API_KEY")
    
    client = OpenRouterClient(api_key=api_key)
    
    # Gắn thông tin mô hình đã chọn vào client để dễ sử dụng sau này
    client.default_model = model_name
    client.default_max_tokens = max_tokens
    client.default_kwargs = kwargs
    
    return OpenRouterRunnable(client, model=model_name, max_tokens=max_tokens, **kwargs)


# llm = get_openrouter_llm("google/gemma-3-27b-it:free")
# start_time = time.perf_counter()
# response = llm.invoke("Bài hát mới nhất của Sơn Tùng MTP?")
# end_time = time.perf_counter()
# print(response)
# print("⏱️ Thời gian phản hồi:", round(end_time - start_time, 2), "giây")
# # Để lấy danh sách các mô hình có sẵn:
# models = llm.get_available_models()
# print(models)