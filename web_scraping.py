import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://learn.microsoft.com"
START_URL = "https://learn.microsoft.com/en-us/azure/?product=popular"
OUTPUT_FILE = "azure_docs.txt"
MAX_PAGES = 20          
DELAY_SECONDS = 1        

HEADERS = {"User-Agent": "Mozilla/5.0 (educational RAG project scraper)"}


def get_soup(url):
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def find_azure_links(soup):
    """Pull all links on the page that point to /en-us/azure/... articles."""
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/en-us/azure"):
            links.add(BASE_URL + href)
        elif href.startswith(BASE_URL + "/en-us/azure"):
            links.add(href)
    return links


def scrape_page(url):
    """Fetch one page and return (title, main_text)."""
    soup = get_soup(url)
    title = soup.title.string.strip() if soup.title else "No title"
    main = soup.find("main") or soup
    text = main.get_text(separator="\n", strip=True)
    return title, text


def main():
    print(f"Fetching link list from: {START_URL}")
    start_soup = get_soup(START_URL)
    links = list(find_azure_links(start_soup))[:MAX_PAGES]
    print(f"Found {len(links)} pages to scrape (limited to {MAX_PAGES}).")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, url in enumerate(links, 1):
            try:
                title, text = scrape_page(url)
                f.write(f"URL: {url}\n")
                f.write(f"TITLE: {title}\n")
                f.write("-" * 60 + "\n")
                f.write(text + "\n")
                f.write("=" * 80 + "\n\n")
                print(f"[{i}/{len(links)}] Scraped: {url}")
            except Exception as e:
                print(f"[{i}/{len(links)}] FAILED {url}: {e}")
            time.sleep(DELAY_SECONDS)

    print(f"\nDone. Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
