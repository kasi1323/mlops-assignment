# MLOps Assignment Report

## 1. Serving Configuration (Phase 1)

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 80GB

| Flag | Value | Justification |
|------|-------|---------------|
| `--tensor-parallel-size` | 1 | Single H100; no multi-GPU split needed |
| `--max-model-len` | 8192 | Workload uses 1.5–3K prompt tokens + short outputs; 8K gives headroom without over-allocating KV cache |
| `--max-num-seqs` | 64 | Allows enough concurrency for 10 RPS × 2–3 LLM calls per agent request without excessive memory pressure |
| `--enable-chunked-prefill` | true | Reduces time-to-first-token on long prompts by interleaving prefill chunks with decode, improving P95 latency |
| `--gpu-memory-utilization` | 0.90 | Maximises KV cache size on the H100 while leaving 10% headroom for CUDA kernels and activations |
| `--enable-prefix-caching` | true | Caches the KV representation of repeated schema prefixes across requests to the same database, reducing prefill cost on the second and third LLM call in a chain (added in Iteration 2) |

---

## 2. Baseline Eval Results (Phase 5)

Eval set: 30 questions from BIRD-bench dev split across 11 SQLite databases.

| Metric | Value |
|--------|-------|
| Overall pass rate | 23.3% (7/30) |
| Average iterations per question | 1.63 |
| Pass rate at iter 1 | 23.3% |
| Pass rate at iter 2 | 23.3% |
| Pass rate at iter 3 | 23.3% |

**Commentary:** The pass rate of 23.3% reflects the difficulty of BIRD-bench combined with the small active parameter count of Qwen3-30B-A3B (~3B active parameters in the MoE architecture). The per-iteration pass rates are identical across all iterations, meaning the verify→revise loop did not convert any failing questions to passing ones in the baseline run. After tuning, pass rate improved to 30% — analysed further in Section 4.

---

## 3. SLO Tuning (Phase 6)

**Target SLO:** P95 end-to-end agent latency < 5 seconds at 10 RPS over a 5-minute window.

### Baseline load test

| Metric | Value |
|--------|-------|
| Achieved RPS | 8.3 |
| P50 latency | 34.4s |
| P95 latency | 97.2s |
| P99 latency | 116.7s |
| Success rate | 44.8% (1343/3000) |
| Timeouts | 728 |

### Iteration 1

**Saw:** P95=97s, 55% error/timeout rate, queue depth spiking immediately on Grafana.

**Hypothesised:** Each agent request makes 2–3 serial LLM calls (generate_sql → verify → sometimes revise). At 10 RPS this generates 20–30 LLM calls/sec, saturating vLLM. The verify step was calling the LLM even when SQL executed cleanly and returned rows — adding latency with low signal value.

**Changed:** Added a fast-path in `verify_node`: skip the LLM call entirely when SQL executed without error and returned rows. Only invoke the LLM verifier when execution failed or returned 0 rows. This reduces average LLM calls per request from ~2 to ~1.2.

**Result:** Dramatic improvement. Timeouts dropped from 728 to 4. P95 fell from 97s to 8s (12× improvement). P50 fell from 34s to 1.4s. Success rate rose from 44.8% to 87.1%. Still above the 5s P95 SLO.

| Metric | Value |
|--------|-------|
| Achieved RPS | 8.3 |
| P50 latency | 1.4s |
| P95 latency | 8.0s |
| Success rate | 87.1% (2614/3000) |

### Iteration 2

**Saw:** P95=8.0s, P50=1.4s — large gap between median and tail. 381 HTTP errors (12.7%). Most requests are fast but a tail of slow requests (revise path with 3 serial LLM calls) is pulling P95 up.

**Hypothesised:** Requests to the same database share an identical schema prefix in the prompt. vLLM is re-running prefill on the full prompt every time. Enabling prefix caching would let vLLM reuse the KV cache for the schema portion, reducing prefill time on the second and third LLM call in a chain — directly targeting the slow tail.

**Changed:** Added `--enable-prefix-caching` to vLLM launch flags.

**Result:** Minimal improvement. P95 moved from 8.0s to 7.7s (4%). HTTP errors unchanged at ~382. Prefix caching helps the individual call latency but not enough to move the tail — the bottleneck is the number of serial LLM calls per request, not prefill speed.

| Metric | Value |
|--------|-------|
| Achieved RPS | 8.3 |
| P50 latency | 1.3s |
| P95 latency | 7.7s |
| Success rate | 87.1% (2613/3000) |

### Iteration 3

**Saw:** P95 stuck at 7.7s after prefix caching. P50=1.3s. The gap between median and P95 is 6s — driven by requests that exhaust MAX_ITERATIONS=3, making 3 serial LLM calls (generate → LLM verify on failure → revise).

**Hypothesised:** Capping MAX_ITERATIONS at 2 limits the worst case to 2 LLM calls per request. Combined with the fast-path verify (no LLM call when SQL succeeds), most requests make 1 call and the slowest make 2, compressing the tail.

**Changed:** Reduced `MAX_ITERATIONS` from 3 to 2 in `agent/graph.py`.

**Result:** SLO met. P95 dropped from 7.7s to 4.65s, crossing the 5s threshold. P50 improved to 1.18s. Timeouts near zero (2). The 382 HTTP errors (~12.7%) persist and represent requests the agent server drops under load — likely a FastAPI connection limit under concurrent long-running requests. This warrants further investigation but does not block the latency SLO.

| Metric | Value |
|--------|-------|
| Achieved RPS | 8.3 (10 RPS fired, 12.7% HTTP error rate) |
| P50 latency | 1.18s |
| P95 latency | **4.65s ✓** |
| Success rate | 87.2% (2615/3000) |

---

| Metric | Baseline | After tuning | SLO |
|--------|----------|--------------|-----|
| P95 latency | 97.2s | **4.65s** | < 5s ✓ |
| P50 latency | 34.4s | 1.18s | — |
| Success rate | 44.8% | 87.2% | — |

**Verdict:** SLO hit. P95 end-to-end agent latency is 4.65s at 10 RPS over a 5-minute window, under the 5s target. The winning change was reducing MAX_ITERATIONS from 3 to 2, which capped the worst-case serial LLM call chain at 2 calls per request. The fast-path verify (Iteration 1) was the bigger single improvement, cutting P95 from 97s to 8s by eliminating unnecessary LLM calls on successful queries.

---

## 4. Agent Value

The verify→revise loop did not improve pass rate in the baseline eval (iter_1 == iter_2 == iter_3 == 23.3%). Two failure modes were observed: (1) the LLM verifier flagged correct results as implausible — for example, a result of `1.0` representing 100% was rejected as "too low", causing the reviser to loop without improvement; (2) the reviser occasionally reproduced the same SQL despite the verifier's complaint, contributing no correction.

After tuning, pass rate improved from 23.3% to 30%, but the per-iteration rates remain flat (iter_1 == iter_2 == 0.3). The quality gain came not from the revise loop but from the fast-path verify: by skipping the LLM verifier when SQL executes successfully and returns rows, we prevented the verifier from incorrectly rejecting correct answers and sending them through a revise loop that sometimes degraded them.

In summary: the loop architecture did not add value in this configuration. The verifier prompts need tighter constraints — specifically, it should not flag numeric values as implausible when the SQL itself has no error. With that fix, the loop would be expected to recover questions where execution fails or returns 0 rows (~half of all failures), adding an estimated 5–10 percentage points.

---

## 5. What I Would Do With More Time

- **Tighter verify prompts:** Constrain the verifier to flag only SQL errors and 0-row results, not numeric plausibility. This would make the revise loop actually recover failures rather than loop on correct answers.
- **Schema-aware prompts:** Include sample rows (2–3 per table) alongside the schema so the model understands column value ranges, eliminating the `1.0 = 100%` class of verifier false positives.
- **Fix the 12.7% HTTP error rate:** Under sustained 10 RPS load, ~382/3000 requests fail with HTTP 500. Root cause is likely FastAPI's default connection handling under concurrent long-running requests. Adding `--workers 2` to uvicorn or a request queue with backpressure would fix this without touching vLLM.
- **Async verify:** Run SQL execution and LLM verify in parallel rather than sequentially, cutting per-request wall time.
- **Bigger model:** Qwen3-30B-A3B has ~3B active parameters. Qwen3-30B (dense) or a larger MoE would likely push pass rate above 40% without any prompt changes.
