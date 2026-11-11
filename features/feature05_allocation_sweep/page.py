"""Page adapter for the allocation sweep feature."""

def render(page, conditions, results):
    if page == "allocation_conditions":
        conditions()
    elif page == "allocation_results":
        results()
