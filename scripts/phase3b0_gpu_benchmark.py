"""Explicit Phase 3B-0 benchmark only: <=1000 unique passages, no full task."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import psutil
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corpus_selection import select_smoke_documents, fingerprint
from src.embedding_encoder import EncoderConfig, GTEEncoder, sha256_file
from src.embedding_checkpoint import run_checkpoint, CheckpointInterrupted, atomic_json
from src.embedding_benchmark import encode_with_backoff, compare_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu-benchmark', action='store_true', required=True)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-lock', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    args = parser.parse_args()
    if args.count != 1000:
        parser.error('Phase 3B-0 fixed benchmark uses exactly 1000 unique passages; full task disabled')
    if args.output_dir.exists():
        parser.error('Output exists; choose a new directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not fall back and label CPU results as GPU')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected, selection, corpus, manifest_hash = select_smoke_documents(args.corpus_dir, limit=args.count, seed=20261010)
    atomic_json(args.output_dir/'selection.json', selection)
    config = EncoderConfig(device='cpu', batch_size=8, seed=20261010)
    cpu_encoder = GTEEncoder.from_local(args.model_dir, args.model_lock, config)
    texts = [doc['embedding_text'] for doc in selected]
    raw_lengths = [len(row) for row in cpu_encoder.tokenizer(texts, truncation=False, add_special_tokens=True,
        return_attention_mask=False, verbose=False)['input_ids']]
    buckets = {'1-64': sum(n<=64 for n in raw_lengths), '65-128': sum(64<n<=128 for n in raw_lengths),
               '129-256': sum(128<n<=256 for n in raw_lengths), '257-512': sum(256<n<=512 for n in raw_lengths),
               '>512': sum(n>512 for n in raw_lengths)}
    if any(count == 0 for count in buckets.values()):
        raise ValueError('Benchmark selection must include all five token length bands')
    by_length = sorted(range(len(selected)), key=lambda i: (raw_lengths[i], selected[i]['document_id']))
    validation_rows = sorted(by_length[i] for i in np.linspace(0, len(selected)-1, 128, dtype=int))
    validation_docs = [selected[i] for i in validation_rows]
    validation_texts = [doc['embedding_text'] for doc in validation_docs]
    cpu = cpu_encoder.encode_documents(validation_texts)
    cpu_stats = cpu_encoder.last_stats
    with (args.output_dir/'cpu_128.npy').open('xb') as f:
        np.save(f, cpu, allow_pickle=False)
    cuda_config = replace(config, device='cuda')
    encoder = GTEEncoder(cuda_config, cpu_encoder.tokenizer, cpu_encoder.model)
    del cpu_encoder
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.synchronize()
    gpu = encoder.encode_documents(validation_texts)
    consistency = compare_embeddings(cpu, gpu)
    with (args.output_dir/'gpu_128.npy').open('xb') as f:
        np.save(f, gpu, allow_pickle=False)
    print(json.dumps({'stage': 'cpu_gpu_128_verified', **consistency}), flush=True)
    def before():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    def after():
        torch.cuda.synchronize()
        info = psutil.Process().memory_info()
        free, total = torch.cuda.mem_get_info()
        return {'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(),
                'process_peak_ram_bytes': getattr(info, 'peak_wset', info.rss),
                'process_current_ram_bytes': info.rss, 'gpu_free_bytes_after': free, 'gpu_total_bytes': total}
    benchmarks = []
    for batch in (8, 16, 32):
        vectors, result = encode_with_backoff(encoder, texts, batch, before_attempt=before,
            after_attempt=after, clear_oom=torch.cuda.empty_cache)
        result['full_corpus_count_scaled_seconds'] = corpus['unique_passages']/result['documents_per_second']
        result['numerical_consistency_with_cpu_128'] = compare_embeddings(cpu, vectors[validation_rows])
        with (args.output_dir/f'gpu_batch_{batch}.npy').open('xb') as f:
            np.save(f, vectors, allow_pickle=False)
        benchmarks.append(result)
        atomic_json(args.output_dir/f'batch_{batch}.json', result)
        print(json.dumps({'stage': 'batch_complete', 'batch': batch, 'docs_per_second': result['documents_per_second'],
                          'peak_vram': result['peak_vram_allocated_bytes']}), flush=True)
        del vectors
        torch.cuda.empty_cache()
    checkpoint_dir = args.output_dir/'checkpoint_128'
    kwargs = dict(corpus_checksum=corpus['corpus_checksum'], corpus_manifest_hash=manifest_hash,
                  expected_count=128, shard_size=64, selection_hash=fingerprint([d['document_id'] for d in validation_docs]))
    try:
        run_checkpoint(validation_docs, encoder, checkpoint_dir, stop_after_shards=1, **kwargs)
    except CheckpointInterrupted:
        pass
    resumed = run_checkpoint(validation_docs, encoder, checkpoint_dir, **kwargs)
    no_op = run_checkpoint(validation_docs, encoder, checkpoint_dir, **kwargs)
    if resumed['resumed_shards'] != 1 or resumed['newly_encoded_shards'] != 1 or no_op['newly_encoded_shards'] != 0:
        raise ValueError('Real CUDA checkpoint resume failed')
    report = {'phase': '3B-0', 'scope': 'benchmark_only', 'full_embedding_executed': False,
        'unique_document_count': len(selected), 'maximum_authorized_unique_documents': 5000,
        'model_lock_sha256': sha256_file(args.model_lock), 'corpus_checksum': corpus['corpus_checksum'],
        'corpus_manifest_hash': manifest_hash, 'selection': selection, 'encoder_configuration': asdict(cuda_config),
        'encoder_configuration_hash': cuda_config.config_hash, 'truncation_policy': '512/512, preserve complete original text',
        'token_length_statistics': {'buckets': buckets, 'quantiles': {str(q): float(np.percentile(raw_lengths,q)) for q in (0,50,90,95,99,100)},
                                    'mean': float(np.mean(raw_lengths)), 'truncated_count': sum(n>512 for n in raw_lengths)},
        'cpu_128_encoding': cpu_stats, 'cpu_gpu_128_consistency': consistency, 'validation_document_ids': [d['document_id'] for d in validation_docs],
        'batch_benchmarks': benchmarks, 'checkpoint_resume': resumed, 'checkpoint_second_resume': no_op,
        'hardware': {'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'gpu_capability': torch.cuda.get_device_capability(0), 'total_ram_bytes': psutil.virtual_memory().total,
            'nvidia_smi': subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32},
        'estimate_caveats': 'Single run per batch; first CUDA 128 pass warms kernels. Random hash selection spans all length bands. Full-corpus padding distribution, GPU background usage, clocks, temperature and checkpoint I/O may change throughput. No Qwen3 concurrency was tested.'}
    atomic_json(args.output_dir/'report.json', report)
    print(json.dumps({'stage': 'completed', 'report': str(args.output_dir/'report.json')}), flush=True)


if __name__ == '__main__':
    main()
