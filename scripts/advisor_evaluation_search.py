"""Lingxi Advisor's candidate-search CLI in its "evaluation" retrieval strategy.

The CLI (`lingxi-advisor-candidate-search`) always runs the "bounded" strategy:
it stops once the requested pool is full and finds an issue's fix only through
pull requests. "evaluation" is Advisor's exhaustive strategy for benchmark
targets: it loads the repository's closed-issue catalog, runs every search
phase, and also finds fixes from commit messages ("closes #123") in a
metadata-only clone. The CLI has no option for it, so this wrapper sets it and
otherwise takes the same arguments. Run it with Advisor's Python;
scripts/retrieve.py uses it for instances the bounded search left with fewer
than three candidates.
"""

import dataclasses

from lingxi_advisor.adapters.cli import search

_BoundedOptions = search.SearchRetrievalOptions


def _evaluation_options(**kwargs):
    return dataclasses.replace(_BoundedOptions(**kwargs), retrieval_strategy="evaluation")


search.SearchRetrievalOptions = _evaluation_options

if __name__ == "__main__":
    search.main()
