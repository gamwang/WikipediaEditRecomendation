import concurrent.futures
import requests
import os

# Retrieve a single page and report the url and contents
def load_url(url, timeout):
    r = requests.get(url)
    return r.json();

def run_threadpool(URLS):
    # We can use a with statement to ensure threads are cleaned up promptly
    direc = './data' 
    if not os.path.exists(direc):
        os.makedirs(direc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
        f = open('data/file.articles','w')
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                f.write(str(data) + '\n')
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
            else:
                print('%r page is %d bytes' % (url, len(data)))
        f.close()
def main():
    URLS = ["https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&pageids=12341&prop=extracts",
        "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&pageids=12342&prop=extracts",
        "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&pageids=12343&prop=extracts",
        "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&pageids=12344&prop=extracts",
        "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&pageids=12345&prop=extracts"]
    run_threadpool(URLS)

if __name__ == "__main__":
    main()
