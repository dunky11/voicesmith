from bs4 import BeautifulSoup
import requests
import multiprocessing as mp
from joblib import Parallel, delayed
from pathlib import Path

ENGLISH_BLACKLIST = [
    "in",
    "do",
    "am"
]
SPANISH_BLACKLIST = []
RUSSIAN_BLACKLIST = []
GERMAN_BLACKLIST = []

class InitialismsScraper:
    jobs = [
        (
            "https://en.wiktionary.org/wiki/Category:Spanish_initialisms",
            "spanish_initialisms.txt",
            SPANISH_BLACKLIST
        ),      
        ("https://en.wiktionary.org/wiki/Category:Spanish_abbreviations", "spanish_abbreviations.txt", SPANISH_BLACKLIST),
        (
            "https://en.wiktionary.org/w/index.php?title=Category:English_initialisms",
            "english_initialisms.txt",
            ENGLISH_BLACKLIST
        ),
        ("https://en.wiktionary.org/wiki/Category:English_abbreviations", "english_abbreviations.txt", ENGLISH_BLACKLIST),
        (

            "https://en.wiktionary.org/wiki/Category:German_initialisms",
            "german_initialisms.txt",
            GERMAN_BLACKLIST
        ),
        ("https://en.wiktionary.org/wiki/Category:German_abbreviations", "german_abbreviations.txt", GERMAN_BLACKLIST),
        (
            "https://en.wiktionary.org/wiki/Category:Russian_initialisms",
            "russian_initialisms.txt",
            RUSSIAN_BLACKLIST
        ),        
        ("https://en.wiktionary.org/wiki/Category:Russian_abbreviations", "russian_abbreviations.txt", RUSSIAN_BLACKLIST),
    ]

    blacklist = [
        "um",
        "uh",
        "er",
        "ah",
        "aa",
        "ha",
        "he",
        "hu",
        "hui",
        "huii",
        "huui",
        "hi",
        "hah",
        "haha",
        "na",
        "naa",
        "nah",
        "haa",
        "aah",
        "aha",
        "hr",
        "hrr",
        "so"
    ]

    out_dir = Path(".") / "docker" / "to_copy" / "word_lists"

    def scrape_wiki_page(self, index, url, name, blacklist, words):
        print(f"Crawling page {index + 1} for language {name.split('_')[0]}")
        response = requests.get(url)
        html = BeautifulSoup(response.text, "html.parser")
        words.extend(
            [
                el.decode_contents()
                for el in html.select(".mw-category.mw-category-columns ul li a")
            ]
        )
        links = html.select('a:-soup-contains("next page")')
        if len(links) > 0:
            a = links[0]
            href = "https://en.wiktionary.org" + a["href"]
            self.scrape_wiki_page(index + 1, href, name, blacklist, words)
        else:
            self.finish_scraping(name, words, blacklist)

    def finish_scraping(self, name, words, blacklist):
        words = self.post_process(words, blacklist)
        with open(self.out_dir / name, "w", encoding="utf-8") as f:
            for word in words:
                f.write(f"{word}\n")

    def post_process(self, words, blacklist):
        words = set(list(words))
        words = [word.strip() for word in words]
        filtered = []
        for word in words:
            if " " in word:
                continue
            if len(word) == 1:
                continue
            if word.lower() in self.blacklist + blacklist:
                continue
            filtered.append(word)
        filtered.sort()
        return filtered

    def scrape(self):
        self.out_dir.mkdir(exist_ok=True)
        Parallel(n_jobs=max(mp.cpu_count() - 1, 1))(
            delayed(self.scrape_wiki_page)(0, url, name, blacklist, [])
            for url, name, blacklist in self.jobs
        )


if __name__ == "__main__":
    scraper = InitialismsScraper()
    scraper.scrape()
 