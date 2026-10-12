# Third-party notices

DFIP's original client, adapters and research tooling are MIT licensed. That
license does not replace the terms of model weights, upstream implementations,
or evaluation sources. No checkpoint weights or complete downloaded datasets
are distributed in this repository or its Python package.

## Archived upstream source

`benchmarks/research-audits/pinned-sources-2026-10-12.zip` contains unmodified
source, configuration and model-card snapshots from LiquidAI d1-omni-600M,
Fastino GLiNER2.5-Decide, vLLM Semantic Router d3-flash, and AutoTrust JEV-9B and
JEV-27B. These files retain the upstream notices inside the ZIP and their
original licenses. They are research evidence, not DFIP-authored code or a
compatibility guarantee. The Python wheel and source distribution exclude this
ZIP. DFIP source snapshots in `benchmarks/source-v*.zip` retain the MIT terms.

The upstream model-card license declarations at the exact recorded revisions
are linked below. Copies of available upstream license and notice files are in
`licenses/`, with exact source URLs and SHA-256 hashes in
[the license manifest](licenses/manifest.json). The pinned D3 `NOTICE`, omitted
by the original snapshot selector, is provided separately in that directory;
the immutable source archive is unchanged. JEV and GLiNER's pinned repositories
declare Apache-2.0 in their cards without separate license files. The Apache
text is included as [the pinned D3 license](licenses/vllm-sr--d3-flash--LICENSE.txt).

| Checkpoint | Pinned model-card terms |
| --- | --- |
| PPLX-Decider v1.1 27B | [Apache-2.0](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/6195aa55a72ae21e503c8a60198f7b390d42adb5/README.md) |
| Liquid d1-3B | [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE) |
| Clef-Flash | [Apache-2.0](https://huggingface.co/Cloudflare/clef-flash/blob/8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3/LICENSE) |
| GLiNER2.5-Decide | [Apache-2.0](https://huggingface.co/fastino/GLiNER2.5-Decide/blob/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90/README.md) |
| Julia-1 | [Apache-2.0](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/README.md) |
| Security-One 27B | [Apache-2.0](https://huggingface.co/superagent-ai/security-one-27b/blob/1ae09b6bc17a87332341bcd95ea42fa6b5e6b399/README.md); gated checkpoint terms also apply |
| Liquid d1-omni-600M | [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/LICENSE) |
| d3-flash | [Apache-2.0](https://huggingface.co/vllm-sr/d3-flash/blob/f6075bff3c0f7d5704e353d0fb2ebf94888d1691/LICENSE) |
| JEV-9B | [Apache-2.0](https://huggingface.co/autotrust/JEV-9B/blob/7ac669b8fb476ba926491e3d1c39f9330d68dbed/README.md) |
| JEV-27B | [Apache-2.0](https://huggingface.co/autotrust/JEV-27B/blob/14b425269ed24452e0a5ad5a8f3e51d139647a88/README.md) |

Liquid's license includes commercial-use conditions based on annual revenue;
read Section 5 of the linked text before commercial use. DFIP's MIT license
does not waive those conditions. Model downloads may require additional access
terms, and the runtime downloads upstream code rather than redistributing it
as DFIP. Source snapshots beyond the current four-model paper are historical
evidence, not part of its reported cohort.

## Evaluation sources

The original and strengthened studies use
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c76749ec58bd8c3d2ea706b31c333a9059c38f90)
(Apache-2.0) and
[PolyAI/BANKING77](https://huggingface.co/datasets/PolyAI/banking77/tree/8eed2b348ec07d40ffbe91dcc4718e2f0f977714)
(CC BY 4.0). Credit for BANKING77 belongs to PolyAI and its authors; its paper
is [Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/).
The benchmark tooling transforms records into decision questions and workload
fixtures. Outputs, IDs, derived metrics and limited audit examples may be
retained in result files; the downloaded corpora stay in the remote Volume.
The native conformance and shared-scheduler fixtures are synthetic.

Remote containers install PyTorch, Transformers, Triton, SGLang, GLiNER and
other dependencies under their own licenses. These GPU libraries are not
included in the client wheel. Model, project and platform names identify
upstream components and do not imply endorsement.
