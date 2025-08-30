from dotenv import load_dotenv
import os
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory

from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion, AzureAIInferenceChatPromptExecutionSettings

load_dotenv()

API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL')



async def main():

    kernel = Kernel()

    chat_completion = AzureAIInferenceChatCompletion(
        ai_model_id="gpt-4o-mini",
        api_key= API_KEY,
        endpoint= BASE_URL                                          
    )



    kernel.add_service(chat_completion)

    history = ChatHistory()

    settings = AzureAIInferenceChatPromptExecutionSettings(temperature=0)

    user_input = None
    while True:
        
        user_input = input("User -> ")

        if user_input == "quit":
            break
        
        history.add_user_message(user_input)

        result = await chat_completion.get_chat_message_content(
            chat_history = history,
            kernel = kernel,
            settings = settings,
        )

        print("Assistant > " + str(result))

        # Add the message from the agent to the chat history
        history.add_message(result)


if __name__ == "__main__":
    asyncio.run(main())
