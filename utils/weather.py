import requests, re, random
from bs4 import BeautifulSoup

class Weather:
    def __init__(self, shahar):
        self.shahar = shahar
        self.bosh_url = "https://obhavo.uz"
        self.url = f"{self.bosh_url}/{self.shahar}"
        self.session = requests.Session()

    def _request(self):
        response = self.session.get(self.url)
        response.raise_for_status()
        return response.content

    def _extract_numbers(self, text):
        return re.findall(r'\d+\.*\d*', text)

    def _parse_webpage(self):
        page = self._request()
        return BeautifulSoup(page, "html.parser")

    def _extract_weekly_data(self, soup, class_name, pop=False):
        haftalik_data = []
        haftalik_prognoz = soup.find("table", class_="weather-table")
        haftalik_elements = haftalik_prognoz.find_all("td", class_=class_name)
        for i in haftalik_elements:
            if pop:
                haftalik_data.append(self._extract_numbers(i.get_text()))
            else:
                haftalik_data.append(i.get_text().replace("\n", "").replace(" ", ""))
        return haftalik_data

    def kunlik_havo(self):
        page = self._parse_webpage()
        current_details = page.find("div", class_="current-forecast-details").get_text().replace("\n", "")
        bugun_qushimcha_malumotlar = self._extract_numbers(current_details)
        namlik, shamol, bosim, *quyosh_chiqishi_botishi = bugun_qushimcha_malumotlar
        quyosh_chiqishi, quyosh_botishi = quyosh_chiqishi_botishi[:2], quyosh_chiqishi_botishi[2:]

        return {
            "namlik": namlik,
            "shamol": shamol,
            "bosim": bosim,
            "quyosh_chiqishi": quyosh_chiqishi,
            "quyosh_botishi": quyosh_botishi
        }

    def bugungi_prognoz(self):
        page = self._parse_webpage()
        hozirgi_harorat = self._extract_numbers(page.find("div", class_="current-forecast").get_text())
        bugun = page.find("div", class_="current-day").get_text()
        hozirgi_holat = page.find("div", class_="current-forecast-desc").get_text().replace("\n", "")
        kun_prognozi = page.find("div", class_="current-forecast-day")

        tong, kun, oqshom = [self._extract_numbers(kun_prognozi.find("div", class_=f"col-{i}").get_text()) for i in range(1, 4)]

        return {
            "hozirgi harorat": hozirgi_harorat,
            "bugun": bugun,
            "hozirgi holat": hozirgi_holat,
            "tong": tong,
            "kun": kun,
            "oqshom": oqshom
        }

    def haftalik_prognoz(self):
        page = self._parse_webpage()
        haftalik_harorat = self._extract_weekly_data(page, "weather-row-forecast", pop=True)
        haftalik_holat = self._extract_weekly_data(page, "weather-row-desc")
        haftalik_yogingarchilik = self._extract_weekly_data(page, "weather-row-pop", pop=True)

        haftalik_data = {}
        days = ["ertaga", "birinchi_kun", "ikkinchi_kun", "uchinchi_kun", "tortinchi_kun", "beshinchi_kun", "oltinchi_kun"]
        for i, day in enumerate(days):
            gradus_key = f"{day}_gradus"
            holat_key = f"{day}_holat"
            yogingarchilik_key = f"{day}_yogingarchilik"
            haftalik_data[gradus_key] = haftalik_harorat[i]
            haftalik_data[holat_key] = haftalik_holat[i]
            haftalik_data[yogingarchilik_key] = haftalik_yogingarchilik[i]

        return haftalik_data

if __name__ == "__main__":
    cities = ['tashkent', 'andijan', 'bukhara', 'gulistan', 'jizzakh', 'zarafshan', 'karshi', 'navoi',
              'namangan', 'nukus', 'samarkand', 'termez', 'urgench', 'ferghana', 'khiva']

    print("Select a city by entering its corresponding number:")
    for i, city in enumerate(cities, start=1):
        print(f"{i}. {city}")

    city_choice = input("Shaharni raqam bilan tanlang: ")

    if city_choice.strip() and not city_choice.isdigit():
        print("Invalid input. Random city will be selected.")
        city_choice = ""

    if city_choice.strip():
        city_choice = int(city_choice)
        if city_choice < 1 or city_choice > len(cities):
            print("Tanlangan shahar mavjud emas.")
            exit(1)
        selected_city = cities[city_choice - 1]
    else:
        selected_city = random.choice(cities)

    print("Selected city:", selected_city)

    my_weather = Weather(selected_city)

    kunlik_havo = my_weather.kunlik_havo()
    print("Kunlik Havo:", kunlik_havo)

    bugungi_prognoz = my_weather.bugungi_prognoz()
    print("Bugungi Prognoz:", bugungi_prognoz)

    haftalik_prognoz = my_weather.haftalik_prognoz()
    print("Haftalik Prognoz:", haftalik_prognoz)