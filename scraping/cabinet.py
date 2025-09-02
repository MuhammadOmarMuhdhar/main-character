import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import time
import os

class Scraper:
    def __init__(self):
        self.url = "https://en.wikipedia.org/wiki/Cabinet_of_the_United_States"
        self.cabinet_members = []
        
    def scrape_cabinet(self):
        """Scrape current Cabinet members from Wikipedia"""
        print("Fetching Cabinet data from Wikipedia...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        tables = soup.find_all('table', {'class': 'wikitable'})
        print(f"Found {len(tables)} tables")
        
        cabinet_data = []
        
        # Process all tables
        for table_idx, table in enumerate(tables):
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cells = row.find_all('td')
                
                if len(cells) >= 2:
                    title_cell = cells[0]  # Position/Title column
                    name_cell = cells[1]   # Name column
                    
                    # Extract title
                    title = None
                    # Try data-sort-value first
                    title = title_cell.get('data-sort-value')
                    if not title:
                        # Try link text
                        title_link = title_cell.find('a')
                        if title_link:
                            title = title_link.text.strip()
                        else:
                            # Fallback to cell text
                            title = title_cell.get_text().strip().split('\n')[0]
                    
                    # Extract name
                    name = None
                    # Try data-sort-value (format: "Last, First")
                    name_sort = name_cell.get('data-sort-value')
                    if name_sort and ',' in name_sort:
                        parts = name_sort.split(', ')
                        name = f"{parts[1]} {parts[0]}"
                    else:
                        # Try finding link
                        name_link = name_cell.find('a', href=lambda x: x and '/wiki/' in x)
                        if name_link:
                            name = name_link.text.strip()
                        # Also check in <p> tags
                        if not name:
                            p_tag = name_cell.find('p')
                            if p_tag:
                                name_link = p_tag.find('a')
                                if name_link:
                                    name = name_link.text.strip()
                    
                    # Clean up title (remove extra info)
                    if title:
                        title = re.sub(r'\([^)]*\)', '', title).strip()
                        title = title.split('\n')[0].strip()
                    
                    # Add to list if both found
                    if title and name and title not in ['Position', 'Office']:
                        cabinet_data.append({
                            'title': title,
                            'name': name,
                            'table': table_idx + 1
                        })
        
        # Filter out committee names (table 1) and keep only actual cabinet members
        actual_cabinet_members = []
        for member in cabinet_data:
            # Skip table 1 (committees) and entries where name looks like a committee
            if (member['table'] != 1 and 
                'Committee' not in member['name'] and
                'Affairs' not in member['name']):
                actual_cabinet_members.append({
                    'title': member['title'],
                    'name': member['name']
                })
        
        # Remove duplicates, keeping first occurrence
        seen_titles = set()
        final_cabinet = []
        for member in actual_cabinet_members:
            if member['title'] not in seen_titles:
                seen_titles.add(member['title'])
                final_cabinet.append(member)
        
        self.cabinet_members = final_cabinet
        print(f"Scraped {len(self.cabinet_members)} cabinet members")
        return self.cabinet_members
    
    def save_to_json(self, filename="cabinet_members.json"):
        """Save cabinet members data to JSON file"""
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.cabinet_members, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.cabinet_members)} cabinet members to {filepath}")
    
    def run(self):
        """Main method to run the scraper"""
        self.scrape_cabinet()
        self.save_to_json()
        return self.cabinet_members

if __name__ == "__main__":
    scraper = Scraper()
    cabinet = scraper.run()
    
    # Print cabinet members
    print("\nCabinet Members:")
    for member in cabinet:
        print(f"  {member['title']}: {member['name']}")