import requests
url = 'https://flair-detector.herokuapp.com/'
files = {'upload_file': open('urls.txt','rb')}
r = requests.post(url, files=files)
print(r.text)