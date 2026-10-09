from src.utils import GraphState
from src.prompt_template import planing_system_message, planing_human_message, planing_input_variables
from src.utils import PlanFormat

from dotenv import load_dotenv
from src.invocation import invoke_component
import os

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

def plan_agent(state: GraphState):
    original_question = state["original_question"]
    all_mem = []
    for past_exp in state["past_exp"]:
        memory = ""
        plan = ', '.join(past_exp["plan"])
        memory += f"Plan: [{plan}]\n"
        memory += f"Status: {past_exp['plan_summary']['output']} Score: {past_exp['plan_summary']['score']}\n"
        all_mem.append(memory)
    memory = ""
    if len(all_mem) == 0:
        memory = "empty"
    else:
        for id in range(len(all_mem)):
            memory += f"Trial {id}:\n{all_mem[id]}\n"
    
    messages = [
        SystemMessagePromptTemplate.from_template(planing_system_message),
        HumanMessagePromptTemplate.from_template(planing_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=planing_input_variables, messages=messages)
    fprompt = prompt.format(
        question = original_question,
        memory = memory
    )
    output = invoke_component("planner", prompt, {
        "question": original_question,
        "memory": memory
    }, PlanFormat)
    return {"plan": output.step}
