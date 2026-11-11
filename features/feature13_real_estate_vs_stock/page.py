"""Page adapter for the real-estate versus stock feature."""

def render(page, conditions, results):
    if page == "real_estate_conditions":
        conditions()
    elif page == "real_estate_results":
        results()
