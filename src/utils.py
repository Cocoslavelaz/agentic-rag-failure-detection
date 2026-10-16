import pandas as pd
import torch
import json
import torch.nn.functional as F
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import NamedTuple
from typing import Annotated, Sequence, Literal, Sequence, Optional
from typing_extensions import TypedDict, List
from pydantic import BaseModel, Field
from typing_extensions import NotRequired
from src.telemetry import ACTIVE_TRACE
from src.uncertainty import retrieval_features
import operator

def load_dataset(name, split):
    """
    Load queries and corpus.
    This function only supports ir_datasets
    """
    import ir_datasets
    nq_dataset = ir_datasets.load(f"{name}/{split}")
    df = pd.DataFrame(nq_dataset.queries_iter())
    nq_q_dict = {k: {"q": q, "a": a} for k, q, a in zip(df['query_id'].to_list(), df['text'].to_list(), df['answers'].to_list())}
    corpus = nq_dataset.docs_store()
    df_meta = pd.DataFrame(nq_dataset.qrels_iter())
    return corpus, nq_q_dict, df_meta 

def load_hf_model_causal_lm(dtype=torch.bfloat16, device_map="cuda:1"):
    model_id = "hf/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,)

    return model, tokenizer


class RetrieveTopChunk():
    def __init__(self, tokenizer: any = None, embedding_model: any = None, retrieval_model: any = None, corpus: any = None, top_k=10):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.retrieval_model = retrieval_model
        self.corpus = corpus
        self.top_k = top_k

    def __call__(self, query: str):
        batch_dict = self.tokenizer([query], max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: batch_dict[k].to(self.embedding_model.device) for k in batch_dict}
        with torch.no_grad():
            outputs = self.embedding_model(**batch_dict)
            dimension=768 # The output dimension of the output embedding, should be in [128, 768]
            embeddings = outputs.last_hidden_state[:, 0][:dimension]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy()
        list_docs = []
        list_doc_ids = []
        # print("log", embeddings.shape)
        top_doc = self.retrieval_model.search([0], embeddings, top_k=self.top_k)
        for doc_id in top_doc[0]:
            # print(doc_id, top_doc[0][doc_id])
            tmp = self.corpus.get(doc_id)
            list_docs.append(tmp.text)
            list_doc_ids.append(tmp.doc_id)

        trace = ACTIVE_TRACE.get()
        if trace:
            search = getattr(self.retrieval_model, "last_search", None) or {}
            scores = [float(score) for score in top_doc[0].values()]
            verified = search.get("document_norms_verified") is True and search.get("query_norms_verified") is True
            if search.get("faiss_device") == "cpu":
                verified = verified and search.get("cosine_validation", {}).get("status") == "verified"
            hits = search.get("faiss_hits", [[]])[0]
            if any(not h["valid_index_position"] for h in hits):
                verified = False
            trace.retrieval({"query": query, "top_k_requested": self.top_k,
                             "top_k_returned": len(list_doc_ids), "ordered_document_ids": [str(i) for i in list_doc_ids],
                             "raw_faiss_scores": scores, "cosine_similarities": scores if verified else None,
                             "embedding_model": getattr(getattr(self.embedding_model, "config", None), "_name_or_path", "unknown"),
                             **search, **retrieval_features(scores, cosine_verified=verified)})

        return list_docs, list_doc_ids

class RetrieveTopChunkMedcpt():
    def __init__(self, tokenizer: any = None, embedding_model: any = None, retrieval_model: any = None, corpus: any = None, top_k=10):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.retrieval_model = retrieval_model
        self.corpus = corpus
        self.top_k = top_k

    def __call__(self, query: str):
        batch_dict = self.tokenizer(
            [query], 
            truncation=True, 
            padding=True, 
            return_tensors='pt', 
            max_length=64,
        )
        batch_dict = {k: batch_dict[k].to(self.embedding_model.device) for k in batch_dict}
        with torch.no_grad():
            outputs = self.embedding_model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0, :]
    
        embeddings = embeddings.cpu().numpy()
        list_docs = []
        list_doc_ids = []
        # print("log", embeddings.shape)
        top_doc = self.retrieval_model.search([0], embeddings, top_k=self.top_k + 50)
        for doc_id in top_doc[0]:
            # print(doc_id, top_doc[0][doc_id])
            tmp = self.corpus.get(doc_id)
            if len(tmp.a) < 10:
                continue
            if len(list_docs) < self.top_k:
                list_docs.append(tmp.a)
                list_doc_ids.append(tmp.doc_id)
            else:
                break

        return list_docs, list_doc_ids

def load_jsonl(file_path):
    from src.dataset_io import load_dataset as read_local
    return read_local("local", file_path)

def load_dataset(name, path=None, *, split=None, provenance=None, evidence_policy='strict'):
    from src.datasets import load_dataset as read_local
    return read_local(name, path, split=split, provenance=provenance, evidence_policy=evidence_policy)

class GenericDoc(NamedTuple):
    doc_id: str
    d: str
    t: str
    a: str
    m: str
    def default_text(self):
        return self.a

def load_corpus(name="dpr", store_path=None):
    if name == "dpr":
        from src.assets import open_dpr_corpus
        corpus = open_dpr_corpus(store_path or os.getenv("DPR_CORPUS_STORE"))
    else:
        import ir_datasets
        from ir_datasets.util import GzipExtract, LocalDownload
        from ir_datasets.formats import TsvDocs
        path = os.getenv("PUBMED_CORPUS_PATH")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("Set PUBMED_CORPUS_PATH to an existing local corpus")
        DL_DOCS = GzipExtract(LocalDownload(path))
        tmp = TsvDocs(DL_DOCS, doc_cls=GenericDoc, skip_first_line=True)
        corpus = tmp.docs_store()
    return corpus

class QAAnswerFormat(BaseModel):
    analysis: str = Field(description="Your thoughts, analysis about the question and the context. Think step-by-step")
    answer: str = Field(description="The answer for the question")
    success: str = Field(description="binary output (Yes or No), indicate you can answer or not")
    rating: int = Field( default=None, description="How confident, from 0 to 10. More evidence, more agreement, more confident")

class QAAnswerState(TypedDict):
    analysis: str
    answer: str
    success: str | bool | int
    rating: int
    success_raw: NotRequired[str | bool | int]
    success_normalized: NotRequired[str]
    success_normalization_status: NotRequired[str]

class PlanFormat(BaseModel):
    analysis: str = Field(description= "Your analysis. Think step-by-step")
    step: List[str] = Field(description= "different steps to follow, should be in sorted order")

class PlanState(TypedDict):
    analysis: str
    step: List[str]

class StepTaskFormat(BaseModel):
    type: str = Field(description="Type of task, one of [aggregate, question-answering]")
    task: str = Field(description="The detail task to do in this step")

class StepTaskState(TypedDict):
    type: str
    task: str

class PlanSummaryFormat(BaseModel): 
    output: str = Field(description="your output, follow the format")
    answer: str = Field(description="Final answer for the question")
    score: int = Field(description="Confident score")

class PlanSummaryState(TypedDict):
    output: str
    answer: str
    score: int

class PlanExecState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    original_question: str
    plan: List[str] # The plan to follow
    step_question: Annotated[List[StepTaskState], operator.add] # List of sub tasks
    step_output: Annotated[List[QAAnswerState], operator.add] # Output of each sub tasks
    step_docs_ids: Annotated[List[List[str]], operator.add]
    step_notes: Annotated[List[List[str]], operator.add]
    plan_summary: PlanSummaryState
    stop: bool = False

class RagState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question: str  # User question
    documents: List[str]  # List of retrieved documents
    doc_ids: List[str] # List of doc_id
    notes: List[str] # List of notes from retrieved documents
    final_raw_answer: QAAnswerFormat
    
class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    original_question: str  # User question
    plan: List[str] 
    past_exp: Annotated[List[PlanExecState], operator.add]
    final_answer: str

def parse_args():
    parser=argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument("--model", choices=['gpt4omini', 'llama3-70B', 'llama3-8B', 'llama3-70B-0', 'mix01', 'mix02', 'mix03'])
    parser.add_argument("--dataset", choices=['nq', 'hotpotqa', 'triviaqa', "2wiki", "fever", "medmcqa", "simpleqa"])
    parser.add_argument("--exp", choices=['plan_rag_extract', 'plan_rag', 'llmcot', 'rag_extract', 'llmonly'])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--dataset-path', help='Local KILT/HotpotQA JSONL, JSON or Arrow IPC file')
    parser.add_argument('--dataset-format', choices=['kilt', 'hotpotqa'], default='kilt')
    parser.add_argument('--faiss-device', choices=['cpu', 'gpu'], default=os.getenv('FAISS_DEVICE', 'cpu'))
    parser.add_argument('--index-root', default=os.getenv('INDEX_ROOT', 'save_embs/gte-ml-base'))
    parser.add_argument('--index-prefix', default=os.getenv('INDEX_PREFIX', 'dpr100'))
    parser.add_argument('--embedding-path', default=os.getenv('EMBEDDING_MODEL_PATH', 'hf/gte-multilingual-base'))
    parser.add_argument('--embedding-device', default=os.getenv('EMBEDDING_DEVICE', 'cpu'))
    parser.add_argument('--corpus-store', default=os.getenv('DPR_CORPUS_STORE'))
    args=parser.parse_args()

    return args

if __name__ == "__main__":
    print(extract_short_answer(question="star wars episode ii attack of the clones characters", raw_answer="The main characters in 'Star Wars Episode II: Attack of the Clones' include Anakin Skywalker, Padm\u00e9 Amidala, Obi-Wan Kenobi, Yoda, Count Dooku, and Jango Fett. These characters drive the plot and are central to the film's themes of love, conflict, and the rise of the Sith."))
