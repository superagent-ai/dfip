# Security

DFIP 0.1.x is a research preview. Fixes are developed on `main`; there is no
long-term support or security certification for this version.

Report a suspected vulnerability through
[GitHub private vulnerability reporting](https://github.com/superagent-ai/dfip/security/advisories/new)
when available. If that form is unavailable, use the maintainer contact listed on
the [Superagent organization profile](https://github.com/superagent-ai) to arrange
a private report. Do not put credentials, private requests, or exploit details
in a public issue. Ordinary reproducibility problems belong in Issues.

Inference uses your authenticated Modal account. Keep Hugging Face credentials
in a Modal Secret; they are only attached to checkpoint preparation. The default
app exposes authenticated Modal methods, with no public HTTP endpoint.
Upstream checkpoint Python code executes in the remote container where required
by the model. Review the pinned revision before using sensitive inputs.

Contributors must keep model weights, downloaded datasets, credentials and private
request data out of commits and diagnostic attachments. The archived experiments
use synthetic fixtures and public evaluation sources. Application IDs and remote
paths in historical records are provenance, not authentication credentials.
