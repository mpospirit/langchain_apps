# LinkedIn Summarizer

This project contains scripts to search for LinkedIn profiles, scrape profile information, and generate summaries using AI.

## Scripts

- **tools/tools.py**
  - Contains the function `get_linkedin_profile_url` which searches for LinkedIn profile URLs based on a full name using the TavilySearchResults tool.

- **third_parties/linkedin.py**
  - Contains the function `scrape_linkedin_profile` which scrapes information from a LinkedIn profile URL. It can use a mock URL for testing purposes or fetch real data using the Proxycurl API.

- **main.py**
  - Contains the function `get_information` which integrates the lookup and scraping functions to get LinkedIn profile data and generate a summary using the ChatOpenAI model.

- **agents/linkedin_lookup_agent.py**
  - Contains the function `lookup` which uses a React agent to find LinkedIn profile URLs based on a full name. It utilizes the `get_linkedin_profile_url` function as a tool for the agent.