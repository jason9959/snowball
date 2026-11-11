"""Feature 01 page adapter.

The legacy callbacks are injected by ``app.py`` so the existing UI remains
unchanged while feature-specific code is migrated into this package.
"""


def render(page: str, conditions, results) -> None:
    if page == "return_comparison_conditions":
        conditions()
    elif page == "return_comparison_results":
        results()
