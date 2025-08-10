from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
import dotenv
import os
dotenv.load_dotenv('.env')
class MySchema(BaseModel):
    foo: str


from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Optional
from uuid import UUID
from langchain_core.messages import BaseMessage

class cb_one(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"cb_one, token: {token}")
class cb_two(BaseCallbackHandler):
    def on_chat_model_start(self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(messages)
        print(f"cb_two, on chat_start messages: {messages}")    
        pass
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"cb_two, token: {token}")        
        
llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            cache=None,
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
            api_version="2024-08-01-preview",
            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
            temperature=0.1,
            max_tokens=12000,
            openai_api_type="azure",
            callbacks = [cb_one()]
        )

# First bind with callback
r1 = llm.model_copy()
r1.callbacks.append(cb_two())
# Finally, add structured output
final_runnable = r1.with_structured_output(MySchema)

result = final_runnable.invoke(["Return an object with foo='bar'"])
for i in final_runnable.stream(["Return an object with foo='bar'"]):
    print(i)
print("Structured output:", result)