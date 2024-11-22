from client.errors import MissingAPIKeyError
from client.schema import Messages, Message

from dotenv import load_dotenv
import os

from pydantic import BaseModel
from typing import Literal, Optional, Type, List


class ChatOpenAI:
    def __init__(self, api_key:str = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError()
        
        self.client = self._initialize_sync_client()
        self.async_client = self._initialize_async_client()
        

    def _initialize_sync_client(self):
        from openai import OpenAI
        OpenAI.api_key = self.api_key
        return OpenAI()

    def _initialize_async_client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self.api_key)
    
    def invoke(
        self, 
        messages: Messages|List[Message], 
        model: Literal["gpt-3.5-turbo", "gpt-4o-mini", "chatgpt-4o-latest"] = "gpt-4o-mini",
        max_tokens: int = 1024,
        structured_output: Optional[Type[BaseModel]] = None,
        verbose: bool = False, ## verbose 구현해서 debug용으로 사용할 수 있도록 ...
    ) -> str|BaseModel:
        """
        OpenAI Chat API를 호출합니다.

        Args:
            messages (Messages): 사용자 입력 프롬프트 List화. -> get_messages check.
            model (Literal): 사용할 OpenAI 모델. 미리 정의된 값 중 하나를 사용해야 합니다.
            max_tokens (int): 생성할 최대 토큰 수.
            structured_output (Optional[Type[BaseModel]]): 구조화된 응답을 처리할 Pydantic 모델.

        Returns:
            dict 또는 Pydantic 모델 인스턴스: 응답 데이터.
        """

        try:
            if structured_output:
                ### openai version check 해서 update(24-11-22 기준 beta 사용)
                response = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format=structured_output
                )
                return response.choices[0].message.parsed

            response = self.client.chat.completions.create(
                model=model,
                messages=ChatOpenAI.get_messages(messages),
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"OpenAI API 요청 중 오류 발생: {e}")
        
    async def ainvoke(
        self, 
        messages: Messages|List[Message], 
        model: Literal["gpt-3.5-turbo", "gpt-4o-mini", "chatgpt-4o-latest"] = "gpt-4o-mini",
        max_tokens: int = 1024,
        structured_output: Optional[Type[BaseModel]] = None,
        verbose: bool = False, ## verbose 구현해서 debug용으로 사용할 수 있도록 ...
    ) -> str|BaseModel:
        """
        비동기 클라이언트를 사용하여 OpenAI Chat API를 호출합니다.
        """

        try:
            if structured_output:
                response = await self.async_client.beta.chat.completions.parse(
                    model=model,
                    messages=ChatOpenAI.get_messages(messages),
                    max_tokens=max_tokens,
                    response_format=structured_output
                )
                return response.choices[0].message.parsed
            
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=ChatOpenAI.get_messages(messages),
                max_tokens=max_tokens,
            )

            print(response.choices[0].message.content)

            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"OpenAI API 요청 중 오류 발생: {e}")

    @staticmethod
    def get_messages(messages: List[Message]) -> List[dict[str, str]]:
        return [msg.model_dump() for msg in messages]


if __name__ == "__main__":
    # client = ChatOpenAI()
    # messages = [Message(role="user", content="hello world!")]
    # resp = client.invoke(messages)
    # print(resp)

    import asyncio

    messages = [Message(role="user", content="너는 누구니? 네 이름을 자세히 말해봐.")]

    client = ChatOpenAI()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tasks = [
        client.ainvoke(messages=messages, model="gpt-3.5-turbo"),
        client.ainvoke(messages=messages, model="gpt-4o-mini"),
    ]

    # 이벤트 루프에서 gather 실행
    responses = loop.run_until_complete(asyncio.gather(*tasks))
    print(responses)