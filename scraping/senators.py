import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import time
import os

class Scraper:
    def __init__(self):
        self.url = "https://en.wikipedia.org/wiki/List_of_current_United_States_senators"
        self.senators = []
        
    def scrape_senators(self):
        """Scrape current US Senators from Wikipedia"""
        print("Fetching Senate data from Wikipedia...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the senators table
        table = soup.select_one('#senators')

        print(f"Found senators table: {table is not None}")
        
        if not table:
            raise Exception("Could not find senators table")
        

        senator_names = []

        if table:
            rows = table.find_all('tr')[1:]  # Skip header row

            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    # Look through cells to find the one with a person's name
                    for cell in cells:
                        link = cell.find('a')
                        if link and '/wiki/' in link.get('href', ''):
                            name = link.text.strip()
                            # Filter to ensure it's actually a person's name
                            if len(name.split()) >= 2 and not any(x in name.lower() for x in ['party', 'republican', 'democratic', 'senate', 'house']):
                                senator_names.append(name)
                                break  # Found name for this row, move to next

        self.senators = senator_names
        
       
        print(f"Scraped {len(self.senators)} senators")
        return self.senators
    
    def save_to_json(self, filename="senators.json"):
        """Save senators data to JSON file"""
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.senators, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.senators)} senators to {filepath}")
    
    def run(self):
        """Main method to run the scraper"""
        self.scrape_senators()
        self.save_to_json()
        return self.senators

if __name__ == "__main__":
    scraper = Scraper()
    senators = scraper.run()
    
    # Print senator names
    print("\nSenator names:")
    for senator in senators:
        print(f"  {senator}")