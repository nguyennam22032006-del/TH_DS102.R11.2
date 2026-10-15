import csv
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

url = "https://www.scrapethissite.com/pages/forms/"
data =[]
seen = set()
for page in range(1, 25):
    response = requests.get(
        url,
        params ={"page_num": page},
        headers = {"User-Agent": "Mozilla/5.0"},
        timeout = 30
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("tr.team")

    for row in rows:
        cells = row.find_all("td")
        values = []
        for cell in cells:
            values.append(cell.get_text(strip=True))

        if len(values) != 9:
            continue

        key = (values[0], values[1])
        if key in seen:
            continue

        seen.add(key)

        data.append(values)

    time.sleep(0.5)

output = Path(__file__).with_name("hockey.csv")
with open(output, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Team_name", "Year", "Wins", "Losses", "OT_Losses",
        "Win_Pct","Goals_For","Goals_Against","Goal_Diff"])
    writer.writerows(data)
