import os 

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from src.utils import PlanExecState, PlanSummaryState, StepTaskState
from src.prompt_template import step_system_message, step_human_message, step_input_variables, summary_system_message, summary_human_message, summary_input_variables
from src.utils import StepTaskFormat, PlanSummaryFormat

from dotenv import load_dotenv
from src.invocation import invoke_component
from src.success import should_summarize

load_dotenv()

def task_define(state: PlanExecState):
    messages = [
        SystemMessagePromptTemplate.from_template(step_system_message),
        HumanMessagePromptTemplate.from_template(step_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=step_input_variables, messages=messages)

    # check stop or continue
    if should_summarize(state["plan"], state["step_output"]):
        # summary about this plan and then stop
        messages = [
            SystemMessagePromptTemplate.from_template(summary_system_message),
            HumanMessagePromptTemplate.from_template(summary_human_message),
        ]
        prompt = ChatPromptTemplate(input_variables=summary_input_variables, messages=messages)
        question = state["original_question"]
        plan = f"[{', '.join(state['plan'])}]"
        memory = ""
        for id, item in enumerate(state["step_output"]):
            memory += f"Task: {state['plan'][id]}\nQuestion: {state['step_question'][id]['task']}\nAnswer: {item['answer']}\nConfident score: {item['rating']}\n\n"
        full_prompt = prompt.format(
            question = question,
            plan = plan,
            memory = memory
        )
        output = invoke_component("summary", prompt, {
            "question": question, 
            "plan": plan,
            "memory": memory
        }, PlanSummaryFormat)
        output = PlanSummaryState(**output.model_dump())
        return {"plan_summary": output, "stop": True}
    else:
        plan = f"[{', '.join(state['plan'])}]"
        cur_step = state["plan"][len(state["step_output"])]
        memory = ""
        for id in range(len(state["step_output"])):
            memory += f"Task: {state['plan'][id]}\nAnswer: {state['step_output'][id]['answer']}\n\n"
        response = invoke_component("step_definer", prompt, {"plan": plan, "cur_step": cur_step, "memory": memory}, StepTaskFormat,
                                    step_id=f"step:{len(state['step_output'])}")
        response = StepTaskState(**response.model_dump())
        return {"step_question": [response]}
