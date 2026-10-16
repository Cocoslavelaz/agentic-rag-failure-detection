"""Explicit dataset backends; no schema inference from filenames."""


def load_dataset(name, path=None, *, split=None, provenance=None, evidence_policy='strict'):
    if name == '2wikimultihopqa':
        from .two_wiki import load_two_wiki
        return load_two_wiki(path, split=split, provenance=provenance, evidence_policy=evidence_policy)
    # Legacy callers use dataset name independently from the KILT row schema.
    from src.dataset_io import load_dataset as load_original
    if name in {'hotpotqa', 'kilt', 'local', 'nq', 'triviaqa', '2wiki', 'fever', 'medmcqa', 'simpleqa'}:
        return load_original(name, path)
    raise ValueError(f'Unknown dataset backend: {name!r}')


def agent_input(example, *, dataset_format):
    """Allowlist the single graph input; gold annotations never enter it."""
    if dataset_format == '2wikimultihopqa':
        question = example['question']
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Missing question')
    else:
        from src.dataset_io import question_fields
        _, question = question_fields(example, dataset_format)
    return {'original_question': question}
