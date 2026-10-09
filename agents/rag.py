import os
from time import perf_counter

from agents.step_definer import task_define
from src.utils import RagState, QAAnswerFormat, QAAnswerState
from src.prompt_template import extract_system_messgage, extract_human_message, extract_input_variables
from src.prompt_template import qa_human_message, qa_input_variables, qa_system_message
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.invocation import invoke_component
from src.telemetry import ACTIVE_TRACE
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState, StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

def build_rag_agent(retriever_tool = None):
    def retrieve(state: RagState):
        user_question = state["question"]
        trace = ACTIVE_TRACE.get()
        previous_events = len(trace.data["retrievals"]) if trace else 0
        started = perf_counter()
        try:
            list_docs, list_doc_ids = retriever_tool(query=user_question)
        except Exception as exc:
            if trace:
                trace.retrieval({"query": user_question, "status": "failed",
                                 "top_k_requested": getattr(retriever_tool, "top_k", None), "top_k_returned": None,
                                 "ordered_document_ids": None, "raw_faiss_scores": None,
                                 "cosine_similarities": None, "top1_uncertainty": None,
                                 "topk_mean_uncertainty": None, "score_margin": None,
                                 "similarity_validity_status": "retrieval_failed",
                                 "latency_ms": (perf_counter()-started)*1000,
                                 "error": {"type": type(exc).__name__, "message": str(exc)}})
            raise
        if trace and len(trace.data["retrievals"]) == previous_events:
            trace.retrieval({"query": user_question, "top_k_requested": getattr(retriever_tool, "top_k", None),
                             "top_k_returned": len(list_doc_ids), "ordered_document_ids": list_doc_ids,
                             "raw_faiss_scores": None, "cosine_similarities": None,
                             "similarity_validity_status": "empty" if not list_doc_ids else "scores_unavailable", "top1_uncertainty": None,
                             "topk_mean_uncertainty": None, "score_margin": None})
        if trace:
            trace.data["retrievals"][-1].update(latency_ms=(perf_counter()-started)*1000, status="completed", error=None)
            trace.save()
        state["documents"] = list_docs
        state["doc_ids"] = list_doc_ids
        return state

    def extract(state: RagState):
        # print("--------- EXTRACT -----------")
        list_docs = state["documents"]
        messages = [
            SystemMessagePromptTemplate.from_template(extract_system_messgage),
            HumanMessagePromptTemplate.from_template(extract_human_message),
        ]
        prompt = ChatPromptTemplate(input_variables=extract_input_variables, messages=messages)
        # LLM
        user_question = state['question']
        list_notes = []
        for rank, doc in enumerate(list_docs, start=1):
            note = invoke_component("extractor", prompt, {"passage": doc, "question": user_question},
                                    document_id=state["doc_ids"][rank-1], retrieval_rank=rank)
            list_notes.append(f"[{note}]")
        state["notes"] = list_notes
        return state

    def generate(state: RagState):
        # print("--------- GENERATE -----------")
        # print("log before gen", state)
        notes = state["notes"]
        doc_ids = state["doc_ids"]
        question = state["question"]
        tmps = []
        for doc_id, note in zip(doc_ids, notes):
            tmps.append(f"doc_{doc_id}: {note}")
        docs = "\n\n".join(tmps)
        messages = [
            SystemMessagePromptTemplate.from_template(qa_system_message),
            HumanMessagePromptTemplate.from_template(qa_human_message),
        ]
        prompt = ChatPromptTemplate(input_variables=qa_input_variables, messages=messages)
        # LLM
        full_prompt = prompt.format(
            context=docs,
            question=question,
        )
        # print(full_prompt)
        # Chain

        # Run
        response = invoke_component("qa", prompt, {"context": docs, "question": question}, QAAnswerFormat)
        response = QAAnswerState(**response)
        # print(response)
        return {"final_raw_answer": response}
        # return {"messages": [response]}
    # {"answer": response}


    rag_graph_builder = StateGraph(RagState)
    rag_graph_builder.add_node("retrieve", retrieve)
    rag_graph_builder.add_node("extract", extract)
    rag_graph_builder.add_node("generate", generate)

    rag_graph_builder.add_edge(START, "retrieve")
    rag_graph_builder.add_edge("retrieve", "extract")
    rag_graph_builder.add_edge("extract", "generate")
    rag_graph_builder.add_edge("generate", END)
    rag_graph = rag_graph_builder.compile()
    return rag_graph
