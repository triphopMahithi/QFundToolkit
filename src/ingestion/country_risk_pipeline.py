import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

class CountryRiskScraper:
    def __init__(self, url: str):
        self.url = url
        self.data: List[List[Optional[Union[str, float]]]] = []
        self.columns = [
            "Country",
            "Adj. Default Spread",
            "Equity Risk Premium",
            "Country Risk Premium",
            "Corporate Tax Rate",
            "Moody's rating"
        ]

    def fetch_html(self) -> str:
        """
        Send HTTP GET request to the source URL and return HTML content.
        """
        print(f"Fetching URL: {self.url}")
        response = requests.get(self.url)
        response.raise_for_status()
        return response.text

    @staticmethod
    def parse_percent(text: str) -> Optional[float]:
        """
        Convert percentage string (e.g. '15%') to float ratio (e.g. 0.15).
        """
        try:
            return round(float(text.strip().replace('%', '')) / 100, 6)
        except:
            return None

    def extract_table_data(self, html: str) -> None:
        """
        Parse the HTML and extract country risk data from the target table.
        """
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")

        if len(tables) < 2:
            raise ValueError("Target table not found on the page.")

        table = tables[1]
        rows = table.find_all("tr")

        for row in rows:
            cells = row.find_all("td")
            if len(cells) != 6:
                continue

            country = cells[0].get_text(strip=True)
            if country.lower() == "country":
                continue  # skip duplicated header row

            row_data = [
                country,
                self.parse_percent(cells[1].get_text(strip=True)),  # Adj. Default Spread
                self.parse_percent(cells[2].get_text(strip=True)),  # Equity Risk Premium
                self.parse_percent(cells[3].get_text(strip=True)),  # Country Risk Premium
                self.parse_percent(cells[4].get_text(strip=True)),  # Corporate Tax Rate
                cells[5].get_text(strip=True)                       # Moody's rating
            ]
            self.data.append(row_data)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the extracted data to a Pandas DataFrame.
        """
        return pd.DataFrame(self.data, columns=self.columns)

    def generate_filename(self, prefix: str = "country_risk_data", ext: str = "csv") -> str:
        """
        Generate a filename with today's date inside the 'log' folder.
        Example: log/country_risk_data_2025-07-29.csv
        """
        date_str = datetime.today().strftime("%Y-%m-%d")
        folder = "log"
        os.makedirs(folder, exist_ok=True)  # Create folder if not exist
        return os.path.join(folder, f"{prefix}_{date_str}.{ext}")

    def save_to_csv(self, filename: str = None) -> None:
        """
        Save the data as a CSV file. If filename is not provided,
        it will be generated automatically using the current date.
        """
        df = self.to_dataframe()
        if filename is None:
            filename = self.generate_filename()
        df.to_csv(filename, index=False)
        print(f"File saved successfully: {filename}")


if __name__ == "__main__":
    # Example usage
    load_dotenv()
    scraper = CountryRiskScraper(
        url=os.getenv("ctryprem")
    )

    try:
        html = scraper.fetch_html()
        scraper.extract_table_data(html)
        scraper.save_to_csv()
    except Exception as e:
        print(f"Error occurred: {e}")

