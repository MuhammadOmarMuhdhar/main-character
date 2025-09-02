import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import time
import os

class Scraper:
    def __init__(self):
        self.url = "https://en.wikipedia.org/wiki/List_of_current_members_of_the_United_States_House_of_Representatives"
        self.congress_members = []
        
    def scrape_congress(self):
        """Scrape current US House Representatives from Wikipedia"""
        print("Fetching House data from Wikipedia...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the representatives table
        table = soup.select_one('#votingmembers')
        
        print(f"Found representatives table: {table is not None}")
        
        if not table:
            # Try finding any table with representatives
            tables = soup.find_all('table', class_=re.compile('wikitable'))
            if tables:
                table = tables[0]  # Use first wikitable
            else:
                raise Exception("Could not find representatives table")
        
        congress_names = []
        
        if table:
            # Look for bold links in the table
            bold_links = table.find_all('b')
            
            for b in bold_links:
                link = b.find('a')
                if link and '/wiki/' in link.get('href', ''):
                    name = link.text.strip()
                    # Filter to ensure it's actually a person's name
                    if (name and 
                        not name.lower() in ['vacant', 'tbd'] and
                        len(name.split()) >= 2 and
                        not any(x in name.lower() for x in ['party', 'republican', 'democratic', 'district', 'at-large'])):
                        congress_names.append(name)

        self.congress_members = congress_names
        print(f"Scraped {len(self.congress_members)} representatives")
        return self.congress_members
    
    def save_to_json(self, filename="congress_members.json"):
        """Save congress members data to JSON file"""
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.congress_members, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.congress_members)} representatives to {filepath}")
    
    def run(self):
        """Main method to run the scraper"""
        self.scrape_congress()
        self.save_to_json()
        return self.congress_members

if __name__ == "__main__":
    scraper = Scraper()
    representatives = scraper.run()
    
    # Print representative names
    print("\nRepresentative names:")
    for rep in representatives:
        print(f"  {rep}")