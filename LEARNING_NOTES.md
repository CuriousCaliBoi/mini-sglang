# Mini-SGLang Deep Learning Journey

## Session Started: 2026-01-05

---

## Phase 1: Core Foundations ✅

### Key Data Structures Learned

#### 1. **Req** (Request) - `python/minisgl/core.py:22`
Represents a single user request with its state.

**Critical fields:**
- `host_ids`: Input tokens (CPU tensor)
- `table_idx`: Index into page table for KV cache
- `cached_len`: Tokens already in cache (from prefix matching!)
- `device_len`: Tokens currently processed
- `max_device_len`: Total tokens when complete

**Key insight:** `extend_len = device_len - cached_len` tells us how many tokens need actual computation!

**Example lifecycle:**
```
Input: "What is the capital of France?" → generate 20 tokens

Initial state:
  cached_len=5      # Prefix match: "What is the"
  device_len=7      # Full input length
  max_device_len=27 # 7 input + 20 output
  extend_len=2      # Only compute 2 new tokens!
  remain_len=20     # 20 tokens to generate

After generating "Paris":
  cached_len=7
  device_len=8
  remain_len=19
```

#### 2. **Batch** - `python/minisgl/core.py:73`
Collection of requests processed together.
- `phase`: "prefill" (processing input) or "decode" (generating output)
- `reqs`: List of Req objects
- `input_ids`: Token tensor for GPU
- `attn_metadata`: Metadata for attention kernels

#### 3. **Context** - `python/minisgl/core.py:101`
Global state shared across the system.
- `kv_cache`: The KV cache pool
- `attn_backend`: Attention implementation (FlashAttention/FlashInfer)
- `page_table`: Maps requests to cache pages
- `batch`: Currently executing batch

#### 4. **SamplingParams** - `python/minisgl/core.py:14`
Configuration for text generation.

### Message Flow Learned

Components communicate via **ZeroMQ messages**:

1. **TokenizeMsg**: API Server → Tokenizer
   ```python
   {uid, text, sampling_params}
   ```

2. **UserMsg**: Tokenizer → Scheduler
   ```python
   {uid, input_ids, sampling_params}
   ```

3. **DetokenizeMsg**: Scheduler → Detokenizer
   ```python
   {uid, next_token, finished}
   ```

4. **UserReply**: Detokenizer → API Server
   ```python
   {uid, incremental_output, finished}
   ```

### Complete Request Lifecycle

```
USER: "What is the capital of France?"
  ↓
[API Server] receives HTTP request
  ↓ TokenizeMsg
[Tokenizer Worker] text → tokens
  ↓ UserMsg{input_ids}
[Scheduler Rank 0]
  - match_prefix(input_ids) → cached_len ← KV CACHE MAGIC HERE!
  - Creates Req{cached_len, ...}
  - Builds Batch{phase="prefill"}
  ↓
[Engine] forward pass (only compute extend_len tokens!)
  ↓
[Scheduler] collects output
  ↓ DetokenizeMsg
[Detokenizer] token → text
  ↓ UserReply{" The"}
[API Server] streams to user

... (decode phase repeats) ...

Final: "The capital of France is Paris."
```

---

## Phase 2: KV Cache Deep Dive (Next)

### Files to Read:
- [ ] `python/minisgl/kvcache/base.py` - Abstract interface
- [ ] `python/minisgl/kvcache/naive_manager.py` - Baseline
- [ ] `python/minisgl/kvcache/radix_manager.py` - The magic!
- [ ] `python/minisgl/scheduler/cache.py` - Integration

### Experiments to Run:
1. Compare radix vs naive cache performance
2. Add debug prints to see cache hits
3. Visualize the radix tree structure

---

## Phase 3: Attention Backends (Upcoming)

### Files to Study:
- [ ] `python/minisgl/attention/base.py`
- [ ] `python/minisgl/attention/flashattention.py`
- [ ] `python/minisgl/attention/flashinfer.py`

---

## Phase 4: Advanced Optimizations (Upcoming)

- [ ] Chunked Prefill
- [ ] Overlap Scheduling
- [ ] CUDA Graphs

---

## Key Insights So Far

1. **The cached_len is the key optimization metric**
   - Set by `match_prefix()` in the cache manager
   - Determines how many tokens we skip computing
   - Radix cache maximizes this value

2. **Prefill vs Decode are fundamentally different**
   - Prefill: Process multiple input tokens at once
   - Decode: Generate one token at a time
   - Each can use different attention backends

3. **Page table maps requests to KV cache**
   - Each request gets a table_idx
   - Cache is organized in pages
   - Currently page_size=1 (see core.py:120)

---

## Next Steps

1. Finish installation
2. Run interactive shell
3. Add debug logging to radix_manager.py
4. Send requests with shared prefixes and watch cache hits
5. Measure performance difference: radix vs naive
