"""Feature 02 page adapter; calculation migration can happen independently."""


def render(page: str, conditions, results) -> None:
    if page == "periodic_conditions":
        conditions()
    elif page == "periodic_results":
        results()
