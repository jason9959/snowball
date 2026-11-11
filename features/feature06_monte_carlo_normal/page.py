"""Feature 04 page adapter preserving the current screens."""


def render(page: str, conditions, results) -> None:
    if page == "monte_normal_conditions":
        conditions()
    elif page == "monte_normal_results":
        results()
