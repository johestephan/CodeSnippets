from newspaper import Article
import datetime
import feedparser
import time
from transformers import pipeline
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import requests

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

my_date = datetime.date.today() # if date is 01/01/2018
year, week_num, day_of_week = my_date.isocalendar()
CurrentWeek = "Week #" + str(week_num) + " of year " + str(year)

## Get the Feeds from URL
f = feedparser.parse("https://cloudblog.withgoogle.com/rss/")
f.entries[0].published_parsed
all_blogs = [entry for entry in f.entries if time.time() - time.mktime(entry.published_parsed) < (86400*7)]

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

ALL_Entries = list()

for blog in all_blogs:
    url = blog['links'][0]['href']
    article = Article(url)
    article.download()
    article.parse()
    inputs = tokenizer("summarize: " + article.text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
    inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    this_entry = '''<p><h3>%s</h3>
    %s<BR/>
    <a href='%s'>LINK</a></p>
    ''' % (blog['title'], summary, url)
    ALL_Entries.append(this_entry)
    


# End point for yout requests
url = "https://api.medium.com/v1"
userId = 'xxxx'
postURL = 'xxx'
MEDIUM_API_KEY="xxxx"

start = '''<h1> "News from Google Cloud %s"</h1>
<p> Here are the newest news from Google Cloud, summarized by our AIBot </p>
''' % (CurrentWeek)
# header requred
header = {
    "Accept":	"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding"	:"gzip, deflate, br",
    "Accept-Language"	:"en-US,en;q=0.5",
    "Connection"	:"keep-alive",
    "Host"	:"api.medium.com",
    "TE"	:"Trailers",
    "Authorization": f"Bearer {MEDIUM_API_KEY}",
    "Upgrade-Insecure-Requests":	"1",
    "User-Agent":	"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
}


data = {
    "title": "News from Google Cloud %s" % (CurrentWeek),
    "contentFormat": "html",
    "content": "%s %s" % ( "News from Google Cloud %s" % (CurrentWeek), '<BR/>'.join(ALL_Entries)),
    "tags": ["Google Cloud", "News", "TLDR", "AIBot", "AI"],
    "publishStatus": "draft"   # "public" will publish to gibubfor putting draft use value "draft"
}

response = requests.get(
    url=url + "/me", #https://api.medium.com/me
    headers=header,
    params={ "accessToken" : MEDIUM_API_KEY},
)
# checking response from server
if response.status_code ==  200:
    response_json = response.json()
    userId = response_json["data"]["id"]      
    response = requests.post(
        url= f"{postURL}",  #https://api.medium.com/me/users/{userId}/posts
        headers=header,
        data=data
    )
    print(response.text)
    if response.status_code ==  200:
        response_json = response.json()
        url = response_json["data"]["url"]
        #print(url)       # this url where you can acess your url

