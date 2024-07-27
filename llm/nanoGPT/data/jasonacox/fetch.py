#!/usr/bin/python3
"""
Pull all content from blog - jasonacox.com

Author: Jason A. Cox
14 June 2023
https://github.com/jasonacox/ProtosAI/

Credit: Simon Willison
    * Code based on project at https://til.simonwillison.net/llms/training-nanogpt-on-my-blog  
"""
import httpx
import json
import re
import string
from html import unescape

# output file
outputfile = "input.nl-json"
fp = open(outputfile, "w")

# regex to remove html tags
tag_re = re.compile('<.*?>')

# blog address - rss feed in json format
url = "https://www.jasonacox.com/wordpress/feed/json"
            
# convert non-standard punctuation into clean ASCII
translation_table = str.maketrans("…’‘‛“”«»„", ".'''\"\"\"\"\"")

# pull blog content
print(f"Pulling blog content from {url}...")
data = httpx.get(url).json()
n = 1
output = {
    "content": []
}

# parse and output content into lines for prepare.py
for item in data["items"]:
    title = item["title"]
    body = tag_re.sub('', item["content_html"])
    body = unescape(body)
    body = body.translate(translation_table)
    body = ''.join(char for char in body if char in string.printable)
    output["content"].append([title, body])
    fp.write(json.dumps([title, body]) + "\n")
    print(f"{n} : " + json.dumps([title, body]) + "\n")
    n = n + 1

# done
fp.close()
print(f"\nDone. Output written to {outputfile}")

# write output to a file
with open(outputfile, 'w') as f:
    json.dump(output, f)
