"""Shared, local-only GTE query/document encoder for the 2Wiki adaptation."""
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from src.embedding_contract import EncoderConfig, stable_json, sha256_file, validate_vectors


class EncoderOOMError(RuntimeError):
    pass


class GTEEncoder:
    def __init__(self, config, tokenizer, model):
        self.config, self.tokenizer = config, tokenizer
        available = torch.cuda.is_available()
        if config.device == 'cuda' and not available:
            raise RuntimeError('CUDA explicitly requested but unavailable; choose cpu or auto')
        self.device = 'cuda' if config.device == 'cuda' or config.device == 'auto' and available else 'cpu'
        self.device_resolution = {'requested': config.device, 'resolved': self.device,
                                  'fallback_reason': 'CUDA unavailable' if config.device == 'auto' and not available else None}
        self.model = model.to(self.device).eval()
        self.last_stats = None
        self.loading_info = None

    @classmethod
    def from_local(cls, model_dir, lock_path, config=None):
        """Import only hash-reviewed local code, not HF dynamic remote code."""
        from transformers import AutoTokenizer
        config = config or EncoderConfig()
        model_dir = Path(model_dir).resolve()
        lock = json.loads(Path(lock_path).read_text(encoding='utf-8'))
        for key in ('model_id', 'model_revision', 'tokenizer_revision', 'code_revision'):
            if getattr(config, key) != lock[key]:
                raise ValueError(f'Encoder lock mismatch: {key}')
        for name, record in lock['files'].items():
            if sha256_file(model_dir / name) != record['sha256']:
                raise ValueError(f'Local model/code/tokenizer hash mismatch: {name}')
        torch.manual_seed(config.seed)
        torch.set_num_threads(config.cpu_threads)
        package_name = '_ma_rag_reviewed_gte_' + config.code_revision
        package = types.ModuleType(package_name)
        package.__path__ = [str(model_dir / 'reviewed_code')]
        sys.modules[package_name] = package
        configuration = importlib.import_module(package_name + '.configuration')
        modeling = importlib.import_module(package_name + '.modeling')
        model_config = configuration.NewConfig.from_pretrained(str(model_dir), local_files_only=True)
        if model_config.hidden_size != config.dimension:
            raise ValueError('Model hidden_size differs from configured embedding dimension')
        if model_config.unpad_inputs or model_config.use_memory_efficient_attention:
            raise ValueError('Only reviewed padded SDPA path is configured for Phase 3A')
        model, info = modeling.NewModel.from_pretrained(
            str(model_dir), config=model_config, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float32, attn_implementation=config.attention_implementation,
            low_cpu_mem_usage=True, output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError(f'Incomplete model weight load: {info}')
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
        result = cls(config, tokenizer, model)
        result.loading_info = info
        return result

    def encode_documents(self, texts):
        return self._encode(texts, self.config.document_max_length, 'documents')

    def encode_queries(self, texts):
        return self._encode(texts, self.config.query_max_length, 'queries')

    def _encode(self, texts, max_length, role):
        texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Each encoding input must be a nonempty string')
        started = time.perf_counter()
        lengths, effective_lengths, outputs_cpu, batch_times = [], [], [], []
        for offset in range(0, len(texts), self.config.batch_size):
            batch_text = texts[offset:offset + self.config.batch_size]
            raw = self.tokenizer(batch_text, truncation=False, add_special_tokens=True,
                                 return_attention_mask=False, verbose=False)
            lengths.extend(len(ids) for ids in raw['input_ids'])
            batch = self.tokenizer(batch_text, max_length=max_length, padding=True,
                                   truncation=True, return_tensors='pt')
            effective_lengths.extend(batch['attention_mask'].sum(dim=1).tolist())
            if not torch.all(batch['input_ids'][:, 0] == self.tokenizer.cls_token_id):
                raise ValueError('Tokenizer first position is not CLS')
            batch_start = time.perf_counter()
            try:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with torch.inference_mode():
                    output = self.model(**batch)
                    hidden = output.last_hidden_state
                    if hidden.ndim != 3 or hidden.shape[0] != len(batch_text) or hidden.shape[2] != self.config.dimension:
                        raise ValueError('Unexpected model hidden-state shape')
                    cls_vector = hidden[:, 0, :].float()
                    if not torch.all(torch.isfinite(cls_vector)) or torch.any(torch.linalg.vector_norm(cls_vector, dim=1) == 0):
                        raise ValueError('Invalid or zero CLS embedding')
                    normalized = F.normalize(cls_vector, p=2, dim=1)
                    array = normalized.detach().to('cpu', dtype=torch.float32).numpy().copy()
                validate_vectors(array, self.config.dimension)
                outputs_cpu.append(array)
                batch_times.append(time.perf_counter() - batch_start)
                del output, hidden, cls_vector, normalized, batch
            except torch.cuda.OutOfMemoryError as exc:
                # Drop GPU references before the caller retries the same inputs.
                for key in list(batch):
                    batch[key] = None
                output = hidden = cls_vector = normalized = None
                raise EncoderOOMError(f'CUDA OOM at {role} batch starting {offset}; no documents skipped or retried') from exc
        embeddings = np.concatenate(outputs_cpu, axis=0) if outputs_cpu else np.empty((0, self.config.dimension), dtype=np.float32)
        elapsed = time.perf_counter() - started
        self.last_stats = {
            'role': role, 'count': len(texts), 'max_length': max_length,
            'untruncated_token_lengths': lengths, 'effective_token_lengths': effective_lengths,
            'truncated_count': sum(length > max_length for length in lengths),
            'token_length_quantiles': {str(q): float(np.percentile(lengths, q)) for q in (0, 50, 95, 100)} if lengths else {},
            'batch_inference_seconds': batch_times, 'seconds_including_tokenization': elapsed,
            'texts_per_second': len(texts) / elapsed if elapsed and texts else None,
            'device_resolution': self.device_resolution, **validate_vectors(embeddings, self.config.dimension)}
        return embeddings
