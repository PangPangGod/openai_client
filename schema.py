from pydantic import BaseModel, Field
from typing import List

class Message(BaseModel):
    role: str = Field(..., description="메시지의 역할 (user, assistant 등)")
    content: str = Field(..., description="메시지의 내용")

class Messages(BaseModel):
    messages: List[Message] = Field(..., description="메시지 객체들의 리스트")

