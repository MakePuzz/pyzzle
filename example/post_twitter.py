"""
Tweet with images from the given arguments

args:
 1. Twitter API key (形式は，CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECREの順にカンマで繋ぐ．)
 2. ツイートするか，自分の最新のツイートにReplyするか (-mまたは--modeオプションで指定．デフォルトはツイート．)
 3. ツイートするテキスト (-tまたは--textオプションで指定．) 
 4. ツイートする画像 (-iまたは--imageオプションで指定．)
実行例：
python post_twitter.py CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_SECRE -t 'Hello world!' -i '../fig/hello_world.jpg'
"""
import sys
import json
import argparse
import requests
from requests_oauthlib import OAuth1


def tweet_image(text, image_path, twt_id=-1):
    """
    Tweet the given text and image(s).

    Parameters
    ----------
    text: str
        Tweet text. Note that it's up to 280 chars or 140 chars in Japanese.
    image_path : list
        Path(s) of image(s) will be posted. 
    twt_id : int
        You can reply to a tweet with the ID, if you specify it.
    
    Returns    
    -------
    request.status_code : int
    """
    # Create a media ID list to post images
    media_ids = ''
    for image in image_path:   
        image = {'media':open(image_path, 'rb')}
        # Upload a image
        request = requests.post(url_media, files=image, auth=twitter_oauth)
        # Check the response to the tweet
        if request.status_code == 200:
            pass
        else:
            return request.status_code

        media_id = json.loads(request.text)['media_id']
        media_id_string = json.loads(request.text)['media_id_string']
        if media_ids == '':
            media_ids += media_id_string
        else:
            media_ids = media_ids + ',' + media_id_string
    
    if twt_id == -1:
        params = {'status':text, 'media_ids':[media_ids]}
    else:
        params = {'status':text, 'media_ids':[media_ids], 'in_reply_to_status_id':twt_id}
    
    # Post a tweet with images
    request = requests.post(url_text, params = params, auth = twitter_oauth)
    return request.status_code

def get_my_latest_tweet():
    """
    Get pyzzle_bot latest tweet.
    
    Returns
    -------
    latest_tweet : dict
    request.status_code : int
    """
    # count: How many tweets do you want?
    params = {"count": 1, "exclude_replies": True, "include_rts": False}
    request = requests.get(url_tl, params = params, auth = twitter_oauth)

    # Check the response
    if request.status_code == 200:
        latest_tweet = json.loads(request.text)[0]
        return latest_tweet
    else:
        return request.status_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tweet with images from the given arguments')
    parser.add_argument('api_key', type=str,
                        help='Twitter API keys given as connecting the each keys with commas like: CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_SECRET')
    parser.add_argument('-m', '--method', type=str, default='tweet',
                        help='Tweet or relpy to pyzzle_bot latest tweet. Specify "tweet" or "reply".')
    parser.add_argument('-t', '--text', type=str,
                        help='Tweet text')
    parser.add_argument('-i', '--image', nargs='*', type=str,
                        help='file path(s) of the images posting to twitter')
    args = parser.parse_args()
    
    key = tuple(args.api_key.split(','))
    operation = args.method
    text = args.text
    image_path_list = args.image
    
    url_media = 'https://upload.twitter.com/1.1/media/upload.json'
    url_text = 'https://api.twitter.com/1.1/statuses/update.json'
    url_tl = 'https://api.twitter.com/1.1/statuses/user_timeline.json'

    twitter_oauth = OAuth1(key[0], key[1], key[2], key[3])  # The order is: Consumer Key, Consumer Secret, 
                                                            # Access Token, Access Token Secert.
    if operation == 'tweet':
        # Tweet
        status_code = tweet_image(text, image_path_list)
        if status_code == 200:
            print('Posted!😊')
        else:
            print('Error😩: code = '+status_code)
    elif operation == 'reply':
        # Get latest tweet id
        latest_tweet = get_my_latest_tweet()
        
        # Reply to my latest tweet
        status_code = tweet_image(text, image_path_list, latest_tweet['id'])
        if status_code == 200:
            print('Posted!😊')
        else:
            print('Error😩: code = '+status_code)
    else:
        print('Error: Check the -m option')
