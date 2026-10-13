# Preparing a public research preview

The proposed version is **0.1.0**, marked as an alpha research preview. The
paper remains an unpublished review draft; release preparation does not submit
it to arXiv or certify production use.

## Build a reviewable artifact

From a committed checkout with no tracked modifications:

```sh
python -m pip install -e '.[dev,paper]'
python scripts/verify_artifacts.py --git-history --regenerate
python scripts/build_release.py --output dist/release
```

The builder archives only committed files, excluding local environments,
credentials, scratch work and untracked files. It checks the source archive's
result hashes and regenerates numerical paper inputs before building a wheel
and source distribution. The output contains the full tracked source ZIP,
Python packages, paper PDF, selected results summary, release notes, a source
commit manifest and `SHA256SUMS`. Python package building may download build
dependencies; it performs no model downloads, cloud execution or publication.
Use a new empty destination for each build.

The Python distributions contain the client, remote execution source and local
tests. The source ZIP includes the raw research archive needed for complete
paper reproduction. Model weights and full downloaded corpora are excluded.
Read [third-party notices](../THIRD_PARTY_NOTICES.md) before using the checkpoints.

## Visibility review

The pre-release scan inspected tracked files and reachable Git history,
including ZIP and gzip payloads. No token, private-key, bearer credential or
literal-secret patterns were detected. This is a pattern-based scan, not a
guarantee that arbitrary sensitive content is absent.

Historical experiment records intentionally retain Modal run/application IDs,
workspace URLs, remote paths and some local source/log paths. These identifiers
are provenance and do not authenticate account access. Some local paths contain
the author's account name. Git commits also retain the author's email address.
The numerical archives and their hashes have not been sanitized or rewritten.
Changing this repository's visibility exposes its history as well as `main`;
deleting a file in a later commit does not remove it from earlier commits.

Before making the repository public, the owners must decide whether to retain
that metadata or publish a separately sanitized snapshot. Preserve the private
originals and record new hashes and redaction mappings if choosing the latter.
Do not force-rewrite or delete research history during release preparation.

## Publication

After the release PR passes CI and is merged, build again from its exact `main`
commit. Review the asset manifest, citation metadata and notes. Enable GitHub
private vulnerability reporting when public visibility makes it available.
Then create a GitHub **pre-release** targeting the reviewed commit and attach
the verified assets. The tag, repository visibility and release publication are
separate owner actions; this preparation workflow changes none of them.

No PyPI upload is configured. Verify control of the package name and use trusted
publishing before arranging a package-index release. A DOI or arXiv identifier
must only be added to the citation metadata after one has been assigned.
