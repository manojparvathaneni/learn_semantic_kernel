from dotenv import load_dotenv
import os
import asyncio
import certifi

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion, AzureAIInferenceChatPromptExecutionSettings

from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from colorama import Fore, Style

load_dotenv()

API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL')



async def main():


# Build the Azure client with certifi's CA bundle
    azure_client = ChatCompletionsClient(
        endpoint=BASE_URL,
        credential=AzureKeyCredential(API_KEY), # type: ignore
        connection_verify=certifi.where(),          # <- key line (fixes SSL on mac)
    )


    chat_completion = AzureAIInferenceChatCompletion(
        ai_model_id="gpt-4o-mini",
        client=azure_client                                      
    )

    system_message= "You are a Python expert."
    history = ChatHistory(system_message=system_message)

    settings = AzureAIInferenceChatPromptExecutionSettings(temperature=0)

    user_input = None
    while True:
        
        user_input = input(f"{Fore.CYAN}User -> {Style.RESET_ALL}")

        if user_input == "quit":
            break
        
        history.add_user_message(user_input)

        result = await chat_completion.get_chat_message_content(
            chat_history = history,
            settings = settings,
        )

        print(f"{Fore.MAGENTA}Assistant -> {Style.RESET_ALL}{str(result)}")
        history.add_message(result) # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
