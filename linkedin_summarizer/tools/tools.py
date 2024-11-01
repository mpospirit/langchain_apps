from langchain_community.tools.tavily_search import TavilySearchResults


def get_linkedin_profile_url(full_name: str) -> str:
    """
    Searches for the LinkedIn or Twitter profile page.
    """
    search = TavilySearchResults(max_results=10)
    search_results = search.run(full_name)

    # Only returning the URLs
    search_results = [result["url"] for result in search_results]

    return search_results

if __name__ == "__main__":
    print(get_linkedin_profile_url("Çağrı Gökpunar"))