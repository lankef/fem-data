# GPU Profiling Summary — JAX on NVIDIA L40S (nsys)

**Job ID:** 7906801  
**GPU:** NVIDIA L40S (46 GB)  
**Framework:** JAX 0.6.2  
**Profiler:** NVIDIA Nsight Systems  

---

## Key Findings

### 1. JIT Compilation Dominates Total Runtime
The GPU was almost entirely idle for the majority of the trace. The `ptxas` process (CUDA compiler) was active during this period, confirming that JAX was spending most of the job compiling kernels rather than running them. This is expected on a forced recompile but should be eliminated in production runs via the JAX compilation cache.

**Fix:**
```python
import jax
jax.config.update("jax_compilation_cache_dir", "/scratch/user/jax_cache")
```

---

### 2. CPU-GPU Ping-Pong Pattern
Once the GPU became active, the trace revealed a clear alternating pattern: GPU active → GPU idle → GPU active → GPU idle. The CPU was busy exactly when the GPU was idle, indicating repeated synchronization between CPU and GPU — the GPU sits idle waiting for the CPU to finish between kernel launches.

**Likely causes:**
- `block_until_ready()` called inside a loop
- `float()`, `int()`, or `print()` on JAX arrays inside a loop (forces device→host transfer)
- Python-level loops over JAX ops instead of `vmap` or `lax.scan`
- Conditional logic (`if result > threshold`) on un-materialized JAX arrays

---

### 3. Significant CUDA API Overhead
The CUDA API row showed large gray blocks (no API calls being made) aligned with the CPU-busy periods, confirming the CPU stalls are preventing kernel launches. One `cudaStreamSynchronize` call (orange bar) was observed, indicating an explicit CPU-GPU sync barrier somewhere in the code.

---

## Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| High | Enable JAX compilation cache | Eliminates compile time on subsequent runs |
| High | Remove sync-forcing calls inside loops (`float()`, `block_until_ready()`, `print()`) | Eliminates CPU-GPU ping-pong |
| High | Replace Python loops with `jax.vmap` or `jax.lax.scan` | Keeps entire loop on GPU |
| Medium | Add `TraceAnnotation` to isolate stall location | Pinpoints exact bottleneck line |
| Low | Increase batch size (GPU memory ~46 GB, likely underutilized) | Better GPU occupancy |

---

## Diagnostic Code to Add

```python
# 1. Identify sync points in your loop
from jax.profiler import TraceAnnotation

for i in range(n):
    with TraceAnnotation("cpu_prep"):
        x = prepare_data(i)
    with TraceAnnotation("gpu_compute"):
        result = jitted_fn(x)
    with TraceAnnotation("sync_check"):
        val = float(result)  # <-- likely culprit

# 2. Time compile vs. execute separately
import time
result = my_fn(x).block_until_ready()          # 1st call: compile + run
t0 = time.time()
result = my_fn(x).block_until_ready()          # 2nd call: run only
print("Execution only:", time.time() - t0)

# 3. Replace loops with vmap
# Bad
for i in range(n):
    out = jitted_fn(x[i])

# Good
out = jax.vmap(jitted_fn)(x)
```

---

## Environment Notes
- CUDA driver version: 13.0 (backward compatible with CUDA 12 runtime)
- `ncu` profiling blocked by DCGM on cluster — use `nsys` instead
- Conda env provides CUDA 12.2 runtime libraries independently of system CUDA 13
