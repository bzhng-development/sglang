# Debugging Memory Leaks in SGLang SRT

## Quick Diagnosis

```bash
# 1. Find all CUDA memory allocations
grep -r "torch.cuda.*alloc\|empty_cache\|synchronize" python/sglang/srt/ --include="*.py" | cut -d: -f1 | sort | uniq

# 2. Check for missing cleanup in CUDA operations
jq '.[] | select(.signature | contains("cuda")) | select(.doc_summary == null or (.doc_summary | contains("cleanup") | not)) | "\(.file):\(.line) \(.qualname)"' docs/dev/refs/code-index/function-index.json

# 3. Find tensor creation without proper lifecycle management
grep -r "torch\.zeros\|torch\.ones\|torch\.empty" python/sglang/srt/ --include="*.py" | grep -v "del\|None\|clear"
```

## Memory Pool Analysis

### Step 1: Trace Memory Pool Operations
```bash
# Find all memory pool allocations
jq '.[] | select(.module | contains("mem_cache")) | select(.qualname | contains("alloc")) | "\(.file):\(.line) \(.qualname)"' docs/dev/refs/code-index/function-index.json

# Find potential leaks in memory pool
grep -A5 -B5 "alloc" python/sglang/srt/mem_cache/memory_pool.py | grep -v "free\|release"
```

### Step 2: Track KV Cache Lifecycle
```bash
# Find KV cache operations
jq '.[] | select(.module | contains("mem_cache")) | select(.qualname | contains("kv\|cache")) | .qualname' docs/dev/refs/code-index/function-index.json

# Check for proper cleanup in request completion
grep -A10 "finish_request\|abort_request" python/sglang/srt/managers/scheduler.py
```

## Common Leak Patterns

### Pattern 1: Uncleaned Request State
```bash
# Find request state that might not be cleaned
grep -r "self\..*requests\|self\..*batches" python/sglang/srt/managers/ --include="*.py" | grep -v "clear\|del\|pop\|remove"
```

### Pattern 2: Circular References in Cached Objects
```bash
# Find lru_cache usage that might hold references
jq '.[] | select(.decorators | any(contains("lru_cache"))) | "\(.file):\(.line) \(.qualname)"' docs/dev/refs/code-index/function-index.json
```

### Pattern 3: CUDA Graph Memory
```bash
# Find CUDA graph allocations
grep -r "CUDAGraph\|make_graphed_callables" python/sglang/srt/ --include="*.py"
```

## Emergency Fixes

Add to `python/sglang/srt/managers/scheduler.py`:
```python
def emergency_cleanup(self):
    """Force release all cached memory"""
    self.running_batch.reqs.clear()
    self.waiting_queue.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()
```

## Related Files

Key files for memory management:
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
