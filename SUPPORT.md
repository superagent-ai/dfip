# Support

DFIP is a research preview maintained through
[GitHub issues](https://github.com/superagent-ai/dfip/issues). There is no managed
inference service or support SLA. Questions about research claims should include
the model, workload, and result file being discussed.

For a bug report, include:

- DFIP version or commit, Python version, model alias, and checkpoint revision.
- Whether it concerns the client default or an experimental entry point.
- A minimal synthetic request, the expected behavior, and the observed error.
- Relevant sanitized diagnostics and the Modal GPU type, if execution reached
  a remote worker.

`dfip --version`, `dfip models`, and `dfip diagnostics --model <alias>` provide
useful context. Diagnostics calls contact the deployed app and may incur usage.
Do not post tokens, account credentials, private prompts, downloaded weights,
or logs that contain them. Replace private requests with synthetic examples.

If a model fails to load, check that its pinned checkpoint was prepared in the
same Volume used by the deployment, that `DFIP_APP` matches the deployed app,
and that gated-model access was accepted before downloading. Cloud account,
quota, billing, and authentication problems should be taken to the relevant
Modal or Hugging Face support channel.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development,
[SECURITY.md](SECURITY.md) for reporting vulnerabilities, and
[benchmarks/README.md](benchmarks/README.md) for study reproduction.
Experimental paths require application-specific validation before use in
consequential workflows.
