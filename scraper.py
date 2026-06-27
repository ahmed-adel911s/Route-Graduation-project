import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin
from tqdm import tqdm

BASE_URL = "https://learn.microsoft.com"
START_URL = "https://learn.microsoft.com/en-us/azure/?product=popular"
MAX_PAGES = 150 # Scrape more pages for the full dataset
OUTPUT_FILE = "dataset.json"

def get_links(url):
    print(f"Fetching links from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("/en-us/azure/") and not href.endswith(".pdf"):
                full_url = urljoin(BASE_URL, href)
                clean_url = full_url.split("?")[0].split("#")[0]
                links.add(clean_url)
        return list(links)
    except Exception as e:
        print(f"Error fetching links from {url}: {e}")
        return []

def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        main_content = soup.find("main")
        if not main_content:
            return None
            
        title_tag = main_content.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "No Title"
        
        for element in main_content(["script", "style", "nav", "footer", "aside", "button"]):
            element.extract()
            
        content = main_content.get_text(separator=" ", strip=True)
        
        if len(content) < 200:
            return None
            
        return {
            "url": url,
            "title": title,
            "content": content
        }
    except Exception:
        # Ignore individual page errors to not clog the console
        return None

def main():
    links = get_links(START_URL)
    
    # Filter out start url
    links = [link for link in links if link not in (START_URL, "https://learn.microsoft.com/en-us/azure/")]
    
    # Limit links to MAX_PAGES
    links_to_scrape = links[:MAX_PAGES]
    
    print(f"Found {len(links)} potential links. Proceeding to scrape up to {len(links_to_scrape)} pages...")
    
    dataset = []
    
    # Use tqdm for a progress bar!
    for link in tqdm(links_to_scrape, desc="Scraping Pages", unit="page"):
        data = scrape_page(link)
        if data:
            dataset.append(data)
        time.sleep(0.5) # Be polite but slightly faster
        
    print(f"\nFinished scraping. Saved {len(dataset)} records to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
