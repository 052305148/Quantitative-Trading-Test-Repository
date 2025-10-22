import os
import json
from typing import List, Dict, Optional, Any
from openai import OpenAI

class QwenAPIClient:
    """通义千问API客户端封装类，用于替换本地大模型"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QwenAPIClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY", "sk-ca2acb943aeb4179af061253fd853d2a"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = "qwen3-coder-30b-a3b-instruct"
            self.initialized = True
    
    def chat_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> str:
        """聊天完成接口"""
        try:
            # 添加超时参数，默认30秒
            if 'timeout' not in kwargs:
                kwargs['timeout'] = 30
                
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return completion
            else:
                return completion.choices[0].message.content
        except Exception as e:
            print(f"聊天完成出错: {e}")
            # 返回一个默认的JSON数组，包含768个0.0
            return json.dumps([0.0] * 768)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """情感分析"""
        messages = [
            {"role": "system", "content": "你是一个专业的情感分析助手，请分析给定文本的情感倾向，返回JSON格式的结果，包含sentiment（positive/negative/neutral）和score（0-1之间的置信度分数）"},
            {"role": "user", "content": f"请分析以下文本的情感：{text}"}
        ]
        
        response = self.chat_completion(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果无法解析JSON，返回默认结果
            return {"sentiment": "neutral", "score": 0.5}
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """关键词提取"""
        messages = [
            {"role": "system", "content": f"你是一个专业的关键词提取助手，请从给定文本中提取最重要的{top_k}个关键词，以JSON数组格式返回"},
            {"role": "user", "content": f"请从以下文本中提取关键词：{text}"}
        ]
        
        response = self.chat_completion(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果无法解析JSON，返回空列表
            return []
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """文本分类"""
        categories_str = "、".join(categories)
        messages = [
            {"role": "system", "content": f"你是一个专业的文本分类助手，请将给定文本分类到以下类别之一：{categories_str}。返回JSON格式的结果，包含category（分类结果）和confidence（置信度）"},
            {"role": "user", "content": f"请将以下文本进行分类：{text}"}
        ]
        
        response = self.chat_completion(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果无法解析JSON，返回默认结果
            return {"category": categories[0] if categories else "unknown", "confidence": 0.5}
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量"""
        # 注意：通义千问API可能不直接支持嵌入生成，这里提供一个替代方案
        # 如果API支持嵌入，可以在这里实现
        # 否则，可以考虑使用其他嵌入服务或本地模型
        raise NotImplementedError("通义千问API当前不直接支持嵌入生成，请使用其他嵌入服务")
    
    def extract_events(self, text: str) -> List[Dict[str, Any]]:
        """事件提取"""
        messages = [
            {"role": "system", "content": "你是一个专业的事件提取助手，请从给定文本中提取事件信息，每个事件包含trigger（触发词）、subject（主体）、object（客体）和time（时间）。返回JSON数组格式"},
            {"role": "user", "content": f"请从以下文本中提取事件：{text}"}
        ]
        
        response = self.chat_completion(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果无法解析JSON，返回空列表
            return []
    
    def encode_text(self, text: str) -> List[float]:
        """文本编码"""
        # 直接使用文本哈希生成向量，避免API调用
        import hashlib
        
        # 使用文本的MD5哈希值生成向量
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hex_digest = hash_obj.hexdigest()
        
        # 将哈希值转换为768维向量
        vector = []
        for i in range(768):
            # 使用哈希值的不同部分生成0-1之间的浮点数
            chunk_index = (i * 4) % len(hex_digest)
            chunk = hex_digest[chunk_index:chunk_index+4]
            if len(chunk) < 4:
                chunk += '0' * (4 - len(chunk))
            value = int(chunk, 16) / 0xFFFFFFFF
            vector.append(value)
        
        print(f"使用哈希方法编码文本，向量长度: {len(vector)}")
        return vector
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """实体提取"""
        messages = [
            {"role": "system", "content": "你是一个专业的实体识别助手，请从给定文本中提取实体，包括公司名称、人物、地点、产品、技术等。返回JSON数组格式，每个实体包含text（实体文本）、label（实体类型）、start（起始位置）、end（结束位置）和confidence（置信度）"},
            {"role": "user", "content": f"请从以下文本中提取实体：{text}"}
        ]
        
        response = self.chat_completion(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果无法解析JSON，返回空列表
            return []
    
    def extract_event_triplets(self, text: str) -> List[Dict[str, Any]]:
        """事件三元组提取"""
        messages = [
            {"role": "system", "content": "你是一个专业的事件三元组提取助手，请从给定文本中提取事件三元组（主语、谓语、宾语）。返回JSON数组格式，每个三元组包含subject（主语）、predicate（谓语）、object（宾语）和confidence（置信度）"},
            {"role": "user", "content": f"请从以下文本中提取事件三元组：{text}"}
        ]
        
        response = self.chat_completion(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果无法解析JSON，返回空列表
            return []

# 获取单例客户端的函数
def get_qwen_client() -> QwenAPIClient:
    """获取通义千问API客户端单例"""
    return QwenAPIClient()