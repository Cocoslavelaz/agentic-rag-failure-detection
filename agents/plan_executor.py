import os

from agents.step_definer import task_define
from agents.rag import build_rag_agent
from src.utils import GraphState, PlanExecState, RagState, StepTaskFormat, StepTaskState, PlanFormat, PlanSummaryFormat, PlanSummaryState, QAAnswerFormat, QAAnswerState
from src.prompt_template import extract_system_messgage, extract_human_message, extract_input_variables
from src.prompt_template import qa_human_message, qa_input_variables, qa_system_message
from src.prompt_template import aggregate_human_message, aggregate_input_variables, aggregate_system_message
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.invocation import invoke_component
from src.telemetry import step_context
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState, StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

def build_plan_executor(retriever_tool = None):


    rag_agent = build_rag_agent(retriever_tool=retriever_tool)

    def single_task_execute(state: PlanExecState):
        cur_task = state["step_question"][-1]
        query = cur_task["task"]
        if cur_task["type"] == "aggregate":
            messages = [
                SystemMessagePromptTemplate.from_template(aggregate_system_message),
                HumanMessagePromptTemplate.from_template(aggregate_human_message),
            ]
            prompt = ChatPromptTemplate(input_variables=aggregate_input_variables, messages=messages)
            full_prompt = prompt.format(
                question=query,
            )
            response = invoke_component("aggregate", prompt, {"question": query}, QAAnswerFormat)
            response = QAAnswerState(**response)
            step_doc_ids = []
            step_notes = []
        else:
            response = rag_agent.invoke({
                "question": query
            })
            step_doc_ids = [response["doc_ids"]]
            step_notes = [response["notes"]]
            response = response["final_raw_answer"]
    
        return {"step_output": [response], "step_docs_ids": step_doc_ids, "step_notes": step_notes}
    
    def task_definer_out(state: PlanExecState):
        if state["stop"] == True:
            return END
        else:
            return "single_task_execute"
        
    graph_builder = StateGraph(PlanExecState)
    
    graph_builder.add_node("task_definer", task_define)
    def traced_task_execute(state: PlanExecState):
        route = "aggregate" if state["step_question"][-1]["type"] == "aggregate" else "rag"
        with step_context(f"step:{len(state['step_output'])}", route):
            return single_task_execute(state)

    graph_builder.add_node("single_task_execute", traced_task_execute)
    graph_builder.add_edge(START, "task_definer")
    graph_builder.add_edge("single_task_execute", "task_definer")
    graph_builder.add_conditional_edges("task_definer", task_definer_out)
    graph = graph_builder.compile()
    return graph

