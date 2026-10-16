import os
import random
import json 
import numpy as np 
import argparse

from langgraph.graph import MessagesState, StateGraph, START, END
from transformers import AutoTokenizer, AutoModel
from corpus.retrieve import Retriever
from src.utils import load_dataset, load_corpus, parse_args
from src.utils import GraphState
from src.utils import RetrieveTopChunk
from agents.plan_executor import build_plan_executor
from agents.plan import plan_agent
from src.backend import LLMConfig
from src.telemetry import trajectory_context
from src.dataset_io import question_fields
from src.assets import local_embedding_path, corpus_store_path

from dotenv import load_dotenv
load_dotenv()

def plan_executor_node(state: GraphState):
    input = {
        "original_question": state["original_question"],
        "plan": state["plan"],
        "stop": False
    }
    output = plan_executor_agent.invoke(input)
    return {"past_exp": [output]}

if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(name=args.dataset, path=args.dataset_path)
    model_name_or_path = str(local_embedding_path(args.embedding_path))
    corpus_store_path(args.corpus_store)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)
    model = model.to(args.embedding_device)

    retrieve = Retriever(gpu_ids=args.gpus, device=args.faiss_device)
    retrieve.init_index_and_add(root_dir=args.index_root, dataset_name=args.index_prefix)

    dpr100_corpus = load_corpus(store_path=args.corpus_store)
    retriever_tool = RetrieveTopChunk(tokenizer=tokenizer, embedding_model=model, retrieval_model=retrieve, corpus=dpr100_corpus)
    
    global plan_executor_agent

    plan_executor_agent = build_plan_executor(retriever_tool=retriever_tool)


    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("planer_node", plan_agent)
    graph_builder.add_node("plan_executor_node", plan_executor_node)
    graph_builder.add_edge(START, "planer_node")
    graph_builder.add_edge("planer_node", "plan_executor_node")

    graph = graph_builder.compile()

    exp_name = args.exp
    model = args.model
    dataset_name = args.dataset
    
    c = 0
    llm_config = LLMConfig.from_env()
    save_dir = f"{exp_name}_{model}_{dataset_name}_{llm_config.configuration_id}"
    os.makedirs(save_dir, exist_ok=True)
    for id, item in enumerate(dataset):
        if id < args.start_index or id > args.end_index:
            continue
        question_id, question_text = question_fields(item, args.dataset_format)
        print(question_id)
        save_file = os.path.join(save_dir, f"{question_id}.json")
        if os.path.exists(save_file):
            continue
        if dataset_name == "fever":
            question = f"Verify this claim, answer SUPPORTS or REFUTES\n{question_text}"
        else:
            question = question_text
        inputs = {
            "original_question": f"{question}?"
        }
        try:
            with trajectory_context(question_id, llm_config, dataset_id=dataset_name) as trace:
                output = graph.invoke(inputs)
                if trace:
                    trace.data["final_agent_output"] = output
            print(output)
            print()

            with open(save_file, "w") as f:
                json.dump(output, f)
        except Exception as e: 
            print(e)
            pass
