from __future__ import annotations

from environs import Env
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI

env = Env()

def create_chatgpt(version: str) -> ChatOpenAI:
    gpt_vendor = env("GPT_VENDOR")
    if gpt_vendor == "AZURE":
        with env.prefixed(f"{gpt_vendor}_{version}_"):
            llm = AzureChatOpenAI(
                azure_endpoint=env("OPENAI_API_BASE"),
                openai_api_version=env("OPENAI_API_VERSION"),
                deployment_name=env("DEPLOYMENT_NAME"),
                openai_api_key=env("OPENAI_API_KEY"),
                openai_api_type=env("OPENAI_API_TYPE"),
                model_version=env("OPENAI_MODEL_VERSION"),
            )
            return llm
    elif gpt_vendor == "OPENAI":
        with env.prefixed(f"{gpt_vendor}_{version}_"):
            if (org:= env("OPENAI_ORGANIZATION", default=None)):
                llm = ChatOpenAI(
                    openai_api_key=env("OPENAI_API_KEY"), 
                    openai_organization=org,
                    model=env("OPENAI_MODEL"))
                return llm
            else:
                llm = ChatOpenAI(
                    openai_api_key=env("OPENAI_API_KEY"),
                    model=env("OPENAI_MODEL"))
                return llm
    raise NotImplementedError(f"We have not implemented ChatGPT model creation for vendor {gpt_vendor}")
        
        