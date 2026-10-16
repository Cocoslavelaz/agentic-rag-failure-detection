"""Explicit real-model integration; bounded smoke embeddings only, no LLM/graph."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import ijson
import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corpus_selection import select_smoke_documents
from src.faiss_process import ProcessFaissIndex, build_index
from src.embedding_encoder import EncoderConfig, GTEEncoder, sha256_file, stable_json
from src.local_retriever import LocalRetriever


def token_length_audit(corpus_dir, tokenizer, config):
    """No forward calls: histogram and exact padded-batch work over full corpus."""
    started = time.perf_counter()
    histogram = Counter()
    count = truncated = linear = quadratic = 0
    buffer = []
    def consume(texts):
        tokenized = tokenizer(texts, truncation=False, add_special_tokens=True,
                              return_attention_mask=False, verbose=False)
        lengths = [len(row) for row in tokenized['input_ids']]
        histogram.update(lengths)
        work_linear = work_quadratic = 0
        for offset in range(0, len(lengths), config.batch_size):
            batch = lengths[offset:offset + config.batch_size]
            padded = min(max(batch), config.document_max_length)
            work_linear += len(batch) * padded
            work_quadratic += len(batch) * padded * padded
        return len(lengths), sum(n > config.document_max_length for n in lengths), work_linear, work_quadratic
    with (Path(corpus_dir) / 'documents.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            buffer.append(json.loads(line)['embedding_text'])
            if len(buffer) == 256:
                n, t, l, q = consume(buffer)
                count, truncated, linear, quadratic = count + n, truncated + t, linear + l, quadratic + q
                buffer.clear()
        if buffer:
            n, t, l, q = consume(buffer)
            count, truncated, linear, quadratic = count + n, truncated + t, linear + l, quadratic + q
    all_lengths = np.repeat(list(histogram), list(histogram.values()))
    return {'mode': 'tokenization_only_no_model_forward', 'document_count': count,
            'token_length_histogram': dict(sorted(histogram.items())), 'truncated_document_count': truncated,
            'truncation_rate': truncated / count, 'untruncated_token_total': int(all_lengths.sum()),
            'untruncated_token_mean': float(all_lengths.mean()),
            'quantiles': {str(q): float(np.percentile(all_lengths, q)) for q in (0, 50, 90, 95, 99, 100)},
            'padded_batch_linear_work': linear, 'padded_batch_quadratic_work': quadratic,
            'seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-real-integration', action='store_true', required=True)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-lock', type=Path, required=True)
    parser.add_argument('--dataset-archive', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--audit-full-token-lengths', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not 100 <= args.limit <= 1000:
        parser.error('Phase 3A real smoke test must use 100–1,000 documents; full embedding is prohibited')
    if args.report.exists():
        parser.error('Report exists; choose a new path')
    config = EncoderConfig(seed=args.seed, device=args.device)
    selected, selection, corpus_manifest, corpus_manifest_hash = select_smoke_documents(args.corpus_dir, limit=args.limit, seed=args.seed)
    if sha256_file(args.dataset_archive) != corpus_manifest['dataset_sha256']:
        raise ValueError('Question source archive differs from corpus source')
    queries = []
    with zipfile.ZipFile(args.dataset_archive) as archive, archive.open('dev.json') as handle:
        for row in ijson.items(handle, 'item'):
            queries.append({'question_id': row['_id'], 'query': row['question']})
            if len(queries) == 5:
                break
    output_dir = args.corpus_dir / config.config_hash / 'smoke' / selection['selection_hash']
    if output_dir.exists():
        raise FileExistsError(f'Existing smoke asset preserved: {output_dir}')
    faiss_start = time.perf_counter()
    # FAISS remains CPU regardless of encoder device.
    load_start = time.perf_counter()
    encoder = GTEEncoder.from_local(args.model_dir, args.model_lock, config)
    print(stable_json({'stage': 'model_loaded', 'seconds': time.perf_counter() - load_start,
                       'device': encoder.device, 'loading_info': encoder.loading_info}), flush=True)
    if encoder.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    vectors = encoder.encode_documents([doc['embedding_text'] for doc in selected])
    doc_stats = encoder.last_stats
    print(stable_json({'stage': 'documents_encoded', 'count': len(vectors), 'seconds': doc_stats['seconds_including_tokenization']}), flush=True)
    manifest, build_seconds = build_index(output_dir, selected, vectors,
        corpus_checksum=corpus_manifest['corpus_checksum'], corpus_manifest_hash=corpus_manifest_hash,
        encoder_config=config, selection=selection, runtime={'resolved_device': encoder.device})
    loaded = ProcessFaissIndex.load(output_dir, expected_corpus_checksum=corpus_manifest['corpus_checksum'],
        expected_corpus_manifest_hash=corpus_manifest_hash, expected_encoder_config=config)
    query_vectors = encoder.encode_queries([q['query'] for q in queries])
    query_stats = encoder.last_stats
    cosine = loaded.compare_cosine(query_vectors, k=10)
    retriever = LocalRetriever(encoder, loaded)
    events = [{**question, 'retrieval': retriever.search(question['query'])} for question in queries]
    lengths = token_length_audit(args.corpus_dir, encoder.tokenizer, config) if args.audit_full_token_lengths else None
    full_count = corpus_manifest['unique_passages']
    if lengths and lengths['document_count'] != full_count:
        raise ValueError('Full token-audit count differs from corpus manifest')
    effective = doc_stats['effective_token_lengths']
    sample_linear = sum(len(effective[i:i+config.batch_size]) * max(effective[i:i+config.batch_size]) for i in range(0, len(effective), config.batch_size))
    sample_quadratic = sum(len(effective[i:i+config.batch_size]) * max(effective[i:i+config.batch_size])**2 for i in range(0, len(effective), config.batch_size))
    inference_seconds = sum(doc_stats['batch_inference_seconds'])
    estimates = {
        'full_document_count': full_count, 'full_embedding_executed': False,
        'fp32_vector_bytes': full_count * config.dimension * 4,
        'embedding_npy_bytes_estimate': full_count * config.dimension * 4 + 128,
        'flatip_index_bytes_estimate': full_count * config.dimension * 4 + 45,
        'mapping_bytes_exact_format_estimate': sum(len((stable_json({'row_id': i, 'document_id': '2wiki-v1:' + '0'*64}) + '\n').encode()) for i in range(full_count)),
        'one_extra_vector_checkpoint_bytes': full_count * config.dimension * 4,
        'count_scaled_seconds_reference': doc_stats['seconds_including_tokenization'] * full_count / len(vectors),
        'gpu_encoding_seconds': None if encoder.device == 'cpu' else doc_stats['seconds_including_tokenization'] * full_count / len(vectors),
        'gpu_estimate_note': 'No GPU time can be measured with installed CPU-only PyTorch; requires future CUDA probe, not a CPU-to-GPU speed extrapolation.' if encoder.device == 'cpu' else 'Count extrapolation only; token padding distribution must be considered.',
    }
    if lengths:
        estimates['token_work_scaled_inference_seconds_linear'] = inference_seconds * lengths['padded_batch_linear_work'] / sample_linear
        estimates['token_work_scaled_inference_seconds_quadratic'] = inference_seconds * lengths['padded_batch_quadratic_work'] / sample_quadratic
        estimates['token_work_estimate_note'] = 'Two illustrative scaling models (token-linear and attention-quadratic); not confidence bounds. Add measured full tokenization/I/O time; future thermal/load changes may dominate.'
    report = {'test_type': 'real_model_integration', 'scope': 'Phase 3A smoke engineering only, not retrieval-quality evaluation',
              'encoder_config': asdict(config), 'encoder_config_hash': config.config_hash,
              'model_lock_sha256': sha256_file(args.model_lock), 'corpus_checksum': corpus_manifest['corpus_checksum'],
              'corpus_manifest_hash': corpus_manifest_hash, 'output_directory': str(output_dir.resolve()),
              'document_encoding': doc_stats, 'query_encoding': query_stats, 'cosine_verification': cosine,
              'query_selection': 'first five original dev questions, only ID and question text read into retrieval inputs',
              'top10_retrieval_samples': events, 'index_build_seconds': build_seconds,
              'index_manifest': manifest, 'full_corpus_token_audit': lengths, 'full_corpus_estimates': estimates,
              'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                           'torch_cuda_available': torch.cuda.is_available(), 'torch_threads': torch.get_num_threads(),
                           'process_peak_ram_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
                           'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if encoder.device == 'cuda' else None,
                           'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if encoder.device == 'cuda' else None},
              'model_loading_info': encoder.loading_info, 'elapsed_after_selection_seconds': time.perf_counter() - faiss_start}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(stable_json({'stage': 'completed', 'max_cosine_error': cosine['max_absolute_error'],
                       'report': str(args.report), 'peak_ram': report['hardware']['process_peak_ram_bytes']}), flush=True)


if __name__ == '__main__':
    main()
