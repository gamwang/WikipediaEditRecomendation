# WikipediaEditRecomendation
For CS194-16: Data Science Class

# Usage

## How to scrape random 500 active wiki users' contributions:
```
cd scrape
npm install
// To get 500 users' contributions. 
node scrape.js -con contributions.txt
// To get Categories on specific topic (In this case John Canny)
node scrape.js -cat categories.txt John\ Canny
```
## Data schema
Refer to contributions.txt and categories.txt
