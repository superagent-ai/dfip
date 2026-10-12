# Reading notes for the DFIP review draft

These are methodological notes, not copied passages. PDFs were read from primary author/arXiv sources. This is a systems characterization; CUDA graphs, constrained selection, and candidate projection are not claimed as new concepts.

| Paper | Sections read | Consequence for this study |
| --- | --- | --- |
| [SGLang](https://arxiv.org/abs/2312.07104) | Design, evaluation setup, end-to-end performance, ablations, production case study | Preserve structured-selection semantics. Compare full request paths and separate the runtime substrate from a head experiment. |
| [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180) | Design, implementation, experimental setup, serving results, ablations | State exact baseline and workload assumptions. Serial resident latency cannot support arrival-rate or concurrent serving claims. |
| [PyGraph, v3](https://arxiv.org/abs/2503.19779v3) | Introduction, graph transformations, evaluation setup, component ablations, hardware comparison | Give graph replay an explicit ablation; report graph compatibility and failures. A small capture cache is not a novel compiler. The downloaded v3 PDF and older HTML have different evaluation configurations; the paper does not mix their numerical claims. |
| [Triton](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) | Programming model, related work, matrix-multiply evaluation | Custom code is not inherently faster than cuBLAS. Measure the entire API as well as projection alternatives. |
| [MLPerf Inference](https://arxiv.org/abs/1911.02549) | Benchmark design, scenarios, quality constraints, metrics | Separate quality/fidelity, resident latency, throughput, and billed cost. The DFIP protocol is not MLPerf-compliant. |
| [BANKING77 source paper](https://arxiv.org/abs/2003.04807) | Abstract and dataset provenance | Use a separately sourced intent corpus for new DFIP workloads, while acknowledging derived labels and unknown checkpoint training overlap. |

The draft follows a problem/contract/system/method/results/limitations structure, states research questions, retains negative results, and generates numerical tables from archived per-call samples. Intervals resample independent container rounds rather than treating calls as independent hardware replications. The original five-round target, actual completed counts, and attrition are disclosed.
