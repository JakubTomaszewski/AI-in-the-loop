from typing import List

# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.messages import BaseMessage
# from pydantic import BaseModel, Field
from langchain_core.chat_history import InMemoryChatMessageHistory


# store = {}


# class InMemoryHistory(BaseChatMessageHistory, BaseModel):
#     """In memory implementation of chat message history."""

#     messages: List[BaseMessage] = Field(default_factory=list)

#     def add_message(self, message: BaseMessage) -> None:
#         """Add a self-created message to the store"""
#         print(f"************ Adding message to history: {message}")
#         print(f"************ Current history: {self.messages}")
#         self.messages.append(message)

#     def clear(self) -> None:
#         print("************ Clearing chat history")
#         self.messages = []


# def get_history() -> BaseChatMessageHistory:
#     if "chat_history" not in store:
#         print("************ Creating new InMemoryHistory instance")
#         store["chat_history"] = InMemoryHistory()
    
#     print(f"************ Returning history: {store['chat_history']}")
#     return store["chat_history"]


store = {}

def get_history(session_id: str):
    if session_id not in store:
        print("************ Creating new InMemoryChatMessageHistory instance")
        store[session_id] = InMemoryChatMessageHistory()

    print(f"************ Returning history: {store[session_id]}")
    return store[session_id]


def class_info_to_xml(input_data):
    ci_obj = input_data["class_information"]
    xml_lines = ["<class_information>"]
    for class_name, details in ci_obj.class_details.items():
        xml_lines.append(f"  <class name='{class_name}'>")
        xml_lines.append(f"    <performance>{details.performance}</performance>")
        xml_lines.append(
            f"    <prompt_summary>{details.prompt_summary}</prompt_summary>"
        )
        xml_lines.append("  </class>")
    xml_lines.append("</class_information>")

    new_data = input_data.copy()
    new_data["class_information"] = "\n".join(xml_lines)
    return new_data
