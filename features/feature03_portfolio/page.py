"""Feature 03 page adapter preserving the current screens."""


def render(page: str, conditions, results) -> None:
    if page == "portfolio_conditions":
        conditions()
    elif page == "portfolio_results":
        results()
