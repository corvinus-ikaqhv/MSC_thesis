import newspaper
import pandas as pd
from GoogleNews import GoogleNews
from newspaper.article import ArticleException
from newspaper.utils import BeautifulSoup
from datetime import datetime
import json


output_filename = "articles"

sites = ["finance.yahoo.com", "reuters.com", "cnbc.com","ft.com"] # "finance.yahoo.com", "reuters.com", "cnbc.com"
site_to_avoid = "support.google.com"

# news_df = pd.DataFrame()

results_list = []
for day in range(1, 32):
    print(f"Searching for news on January {day}")
    for site in sites:
        googlenews = GoogleNews(lang="en", start=f"01/{str(day).zfill(2)}/2023", end=f"01/{str(day).zfill(2)}/2023")
        googlenews.search(f"Microsoft site:{site}")
        result = googlenews.results()
        if result:
            for _article in result:
                if site_to_avoid not in _article["link"]:
                    print(f"Found article {_article['link']}")
                    # current_article_df = pd.DataFrame(
                    #     {
                    #         "source": [site],
                    #         "link": [_article["link"]],
                    #     }
                    # )
                    # news_df = pd.concat([news_df, current_article_df], ignore_index=True)
                    results_list.append(_article)
# # save as CSV
# news_df.to_csv("newsMicrosoft.csv", sep="|")
# # save as HTML
# article_html = news_df.to_html()
# text_file2 = open("newsMicrosoft.html", "w")
# text_file2.write(article_html)
# text_file2.close()

article_df = pd.DataFrame()

for item in results_list:
    url = item["link"]
    article = newspaper.Article(url=url, language="en")
    try:
        article.download()
        print(f"Downloading article from: {url}")
        article.parse()
        try:
          current_article_df = pd.DataFrame(
              {
                  "title": [item["title"]],
                  "source": [item["media"]],
                  "published_date": [str(article.meta_data["article"]["published_time"])],
                  "text": [str(article.text).rstrip("\n").replace("\n", "")],
              }
          )
          article_df = pd.concat([article_df, current_article_df], ignore_index=True)
        except KeyError:
          soup = BeautifulSoup(article.html, 'html.parser')
          soup_dict = json.loads("".join(soup.find("script", {"type":"application/ld+json"}).contents))
          date_published = [value for (key, value) in soup_dict.items() if key == 'datePublished']
          current_article_df = pd.DataFrame(
              {
                  "title": [item["title"]],
                  "source": [item["media"]],
                  "published_date": [date_published[0]],
                  "text": [str(article.text).rstrip("\n").replace("\n", "")],
              }
          )
          article_df = pd.concat([article_df, current_article_df], ignore_index=True)
    except ArticleException:
        print(f"Could not download URL: {url}")

# save as CSV
article_df.to_csv(f"{output_filename}.csv", sep="|")

# save as HTML
article_html = article_df.to_html()
text_file2 = open(f"{output_filename}.html", "w")
text_file2.write(article_html)
text_file2.close()