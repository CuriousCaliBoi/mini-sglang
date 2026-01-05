#!/usr/bin/env python3
"""
Experiment 1: Compare Radix vs Naive Cache Performance

This experiment demonstrates the impact of prefix caching by:
1. Running the same workload with radix cache
2. Running with naive cache (no prefix reuse)
3. Comparing throughput and latency

Expected outcome: Radix cache should show significant speedup
when requests share common prefixes.
"""

import time
from minisgl.llm import LLM
from minisgl.core import SamplingParams


def run_cache_experiment():
    print("=" * 80)
    print("EXPERIMENT 1: Radix vs Naive Cache Comparison")
    print("=" * 80)

    # Shared prefix: "Translate the following English text to French:"
    shared_prefix = "Translate the following English text to French: "

    # Different requests with the same prefix
    requests = [
        shared_prefix + "Hello, how are you?",
        shared_prefix + "The weather is nice today.",
        shared_prefix + "I love programming.",
        shared_prefix + "Good morning!",
        shared_prefix + "See you later.",
    ]

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=50     # Short outputs
    )

    print(f"\nShared prefix ({len(shared_prefix)} chars): '{shared_prefix}'")
    print(f"Number of requests: {len(requests)}")
    print(f"Each request differs only in the suffix\n")

    # TODO: This experiment requires understanding how to instantiate
    # LLM with different cache backends. The actual implementation may
    # vary based on the LLM class interface.

    print("NOTE: To run this experiment, you need:")
    print("1. Start server with radix cache: python -m minisgl --model MODEL --port 8000")
    print("2. Start server with naive cache: python -m minisgl --model MODEL --port 8001 --cache naive")
    print("3. Send requests to both servers and compare throughput")
    print("\nSee benchmark/online/bench_simple.py for a working example")


def analyze_cache_hits():
    """
    This function will be filled in after we understand the cache manager API
    """
    print("\n" + "=" * 80)
    print("ANALYZING CACHE BEHAVIOR")
    print("=" * 80)

    print("\nTo see cache hits in action, add debug prints to:")
    print("  python/minisgl/kvcache/radix_manager.py")
    print("\nIn the match_prefix() method, add:")
    print("  print(f'Cache hit: {cached_len} tokens reused out of {len(input_ids)}')")


if __name__ == "__main__":
    run_cache_experiment()
    analyze_cache_hits()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Read python/minisgl/kvcache/radix_manager.py to understand the implementation")
    print("2. Add debug prints to visualize cache hits")
    print("3. Run mini-sglang server with different cache modes")
    print("4. Measure the performance difference")
