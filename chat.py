"""
a new day, a new file.
"""

import os
import redis

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import re


from haystack.agents.base import ConversationalAgentWithTools, Tool, ToolsManager
from haystack.agents.memory import Memory
from haystack.nodes import PromptNode, PromptTemplate, EmbeddingRetriever
from haystack.document_stores import WeaviateDocumentStore
from haystack.pipelines import Pipeline


class RedisConversationMemory(Memory):
    def __init__(self,
                 input_key: Optional[str] = "input", 
                 output_key: Optional[str] = "output",
                 memory_id: str = "agent_memory",
                 final_answer_pattern: str = r"Final Answer\s*:\s*(.*)",
                 window_size: int = 20,
                 expiration: int = 3600,
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 **kwargs):
        """
        An implementation of ConversationMemory for storing only the query and the final answer of the agent into a redis db.

        :param input_key: Optional input key, default is "input".
        :param output_key: Optional output key, default is "output"
        :param memory_id: ID of the unique memory to be used
        :param final_answer_pattern: A pattern for parsing the agent's final answer, which only gets saved
        :param window_size: Sliding window size to return the last N items. This is done to avoid too large messages
        :param expiration: Expiration time of the memory in seconds
        :param host: Redis Host
        :param port: Redis Port
        :param db:  Redis DB
        :param kwargs: Additional kwargs to be passed to redis.StrictRedis
        """
        self.input_key = input_key
        self.output_key = output_key
        self.window_size = window_size
        self.final_answer_pattern = final_answer_pattern
        self.__expiry_is_set = False
        self.expire = expiration
        self.redis = redis.StrictRedis(host=host,
                                       port=port,
                                       db=db,
                                       decode_responses=True,
                                       **kwargs)
        self.memory_id = memory_id


    def load(self, keys: Optional[List[str]] = None, **kwargs) -> Any:
        """
        Load conversation history as a formatted string.

        :param keys: Optional list of keys (ignored in this implementation).
        :param k: Optional integer specifying the number of most recent conversation snippets to load.
        :return: A formatted string containing the conversation history.
        """
        chat_list = self.redis.lrange(self.memory_id, self.window_size * -1, -1)

        if chat_list is None:
            return ""

        return "".join(chat_snippet for chat_snippet in chat_list)

    def __is_already_saved(self, string: str) -> bool:
        """
        Checks whether the supplied string is already contained in the last 2 messages (so 1 Human, 1 AI message) of the redis db.

        :param string: The string to check
        """
        previous_messages = self.redis.lrange(self.memory_id, -2, -1)
        return string in previous_messages

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a conversation snippet to memory.

        :param data: A dictionary containing the conversation snippet to save.
        """
        query_message = f"Human: {data[self.input_key]}" # type: ignore
        if not self.__is_already_saved(query_message):
            self.redis.rpush(self.memory_id, query_message)

        output_message = data[self.output_key] # type: ignore

        final_answer_match = re.search(self.final_answer_pattern, output_message)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip('" ')
            ai_message = f"AI: {final_answer}"
            if not self.__is_already_saved(ai_message):
                self.redis.rpush(self.memory_id, ai_message)

        if not self.__expiry_is_set:
            self.redis.expire(self.memory_id, self.expire)
            self.__expiry_is_set = True
    

    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self.redis.delete(self.memory_id)


# Activate logging
import logging

#logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
#logging.getLogger("haystack").setLevel(logging.DEBUG)

api_key=""
azure_deployment_name=""
azure_base_url=""

document_store = WeaviateDocumentStore(
    host="localhost",
    index="Document",
    embedding_dim=1536,
)

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="text-embedding-ada-002",
    api_key="",
    azure_deployment_name="",
    azure_base_url="",
    top_k=8,
)

lfqa_prompt = PromptTemplate(
    name="lfqa",
    prompt_text="Given the context please answer the question using your own words. Generate a comprehensive, summarized answer. If the information is not included in the provided context, reply with 'Provided documents didn't contain the necessary information to provide the answer'Context: {join(documents)};Question: {query}; Answer:",
)

prompt_node = PromptNode(
    model_name_or_path="text-davinci-003",
    default_prompt_template=lfqa_prompt,
    max_length=500,
    api_key="",
    model_kwargs={
        "azure_deployment_name": "",
        "azure_base_url": "",
        "temperature": 1,
    },
)


document_search = Pipeline()
document_search.add_node(component=retriever, name="Retriever", inputs=["Query"])
document_search.add_node(component=prompt_node, name="prompt_node", inputs=["Retriever"])

search_tool = Tool(name="DocumentStore",
                pipeline_or_node=document_search,
                description="Access this tool to find out missing information needed to answer questions",
                output_variable="results")

my_memory_template = """You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions
                     correctly, you have access to the following tools:
                     {tool_names_with_descriptions}
                     To answer questions, you'll need to go through multiple steps involving step-by-step thinking and
                     selecting the appropriate tools and give them the question as input; tools will respond with observations.
                     Decide if the observations provided by the tool contains information needed to answer questions.
                     When you are ready for a final answer, respond with the Final Answer:
                     You should avoid knowledge that is present in your internal knowledge.
                     You do not use prior knowledge, only the observations provided by the tools available to you
                     Use the following format:

                     Question: the question to be answered.
                     Thought: Reason if you have the final answer.
                     If yes, answer the question. If not, find out the missing information needed to answer it.
                     Tool: pick one of {tool_names}.
                     Tool Input: the full updated question to be answered
                     Observation: the tool will respond with the observation
                     ...
                     Final Answer: the final answer to the question

                     Thought, Tool, Tool Input, and Observation steps can be repeated multiple times,
                     but sometimes we can find an answer in the first pass
                     ---
                     Current Conversation:
                     {history}
                     Question: {query}
                     Thought: Let's think step-by-step, I first need to
                     {transcript}
                     """

conversational_agent_prompt_template = PromptTemplate("memory-shot-react", prompt_text=my_memory_template)

prompt_node = PromptNode("gpt-35-turbo", api_key=api_key, max_length=300, model_kwargs={
        "azure_deployment_name": azure_deployment_name,
        "azure_base_url": azure_base_url,
        "temperature": 1,
    },
    stop_words=["Observation:"]
    )
prompt_node.add_prompt_template(prompt_template=conversational_agent_prompt_template)
prompt_node.set_default_prompt_template(prompt_template=conversational_agent_prompt_template)

agent_memory_id = "123456"
redis_host="localhost"
redis_port=6379

agent = ConversationalAgentWithTools(
    prompt_node, memory=RedisConversationMemory(memory_id=agent_memory_id, host=redis_host, port=redis_port), tools_manager=ToolsManager(tools=[search_tool])
)

try:
    while True:
        user_input = input("Human (type 'exit' or press Ctrl+C to quit): ")
        if user_input.lower() == "exit":
            break
        else:
            assistant_response = agent.run(user_input)
            #print("Assistant:", assistant_response)
except KeyboardInterrupt:
    print("Ctrl+C detected. Exiting the loop.")
