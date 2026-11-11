"""Feature 05 page adapter preserving the current screens."""


def render(page: str, conditions, results) -> None:
    if page == "monte_bootstrap_conditions":
        conditions()
    elif page == "monte_bootstrap_results":
        results()
