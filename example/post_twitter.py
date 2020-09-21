# coding: utf-8
"""
Tweet with images from the given arguments

args:
 1. Twitter API key (形式は，CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECREの順にカンマで繋ぐ. )
 2. ツイートするテキスト (-tまたは--textオプションで指定. ) 
 3. ツイートする画像 (-iまたは--imageオプションで指定．)
実行例：
python post_twitter.py CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_SECRE -t 'Hello world!' -i '../fig/hello_world.jpg'
"""

import sys
import json
import argparse

import requests
from requests_oauthlib import OAuth1


parser = argparse.ArgumentParser(description='tweet with images from the given arguments')
parser.add_argument('api_key', type=str,
                    help='Twitter API keys given as connecting the each keys with commas like: CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_SECRET')
parser.add_argument('-t', '--text', type=str,
                    help='Tweet text')
parser.add_argument('-i', '--image', nargs='*', type=str,
                    help='file path(s) of the images posting to twitter')
args = parser.parse_args()
key = args.api_key.split(',')
text = args.text
image_path_list = args.image

twitter_oauth = OAuth1(key[0], key[1], key[2], key[3])  # The order is Consumer Key, Consumer Secret, Access Token, Access Token Secert.
url_media = 'https://upload.twitter.com/1.1/media/upload.json'
url_text = 'https://api.twitter.com/1.1/statuses/update.json'
media_ids = ''

# Create a media ID list to post images
for image_path in image_path_list:
    files = {'media':open(image_path, 'rb')}
    # Upload a image
    request = requests.post(url_media, files=files, auth=twitter_oauth)
    # Check the response to the image uploaded 
    if request.status_code == 200:
        # Upload seccess
        pass
    else:
        # Upload failed
        sys.exit(0)

    media_id = json.loads(request.text)['media_id']
    media_id_string = json.loads(request.text)['media_id_string']

    if media_ids == '':
        media_ids += media_id_string
    else:
        media_ids = media_ids + ',' + media_id_string


params = {'status': text, 'media_ids': [media_ids]}
# Post a tweet with images
request = requests.post(url_text, params = params, auth = twitter_oauth)

# Check the response to tweet text.
if request.status_code == 200:
    # Upload seccess
    print('Posted')
else:
    # Upload failed
    print('Failed')
    print('code = '+request.status_code)
