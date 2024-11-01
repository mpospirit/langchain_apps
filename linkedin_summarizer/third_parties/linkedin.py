import requests
import os
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url: str):
    """
    Scrapes information from a LinkedIn profile
    """

    proxycurl_api_key = os.getenv("PROXYCURL_API_KEY")

    api_key = proxycurl_api_key

    headers = {'Authorization': 'Bearer ' + api_key}
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    params = {
        'linkedin_profile_url': linkedin_profile_url,
        'extra': 'include',
        'github_profile_id': 'include',
        'facebook_profile_id': 'include',
        'twitter_profile_id': 'include',
        'personal_contact_number': 'include',
        'personal_email': 'include',
        'inferred_salary': 'include',
        'skills': 'include',
        'use_cache': 'if-present',
        'fallback_to_cache': 'on-error',
    }
    response = requests.get(api_endpoint,
                            params=params,
                            headers=headers)
    
    # Removing the empty values from the response
    response = response.json()
    response = {k: v for k, v in response.items() if v}

    return response

if __name__ == "__main__":
    print(scrape_linkedin_profile("https://www.linkedin.com/in/cagrigokpunar"))
