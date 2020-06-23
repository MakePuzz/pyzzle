import os
import json

import requests


def read_config(path):
    config = {}
    with open(path) as f:
        for l in f.readlines():
            if ':' in l:
                k, v = l.strip().split(':', 1)
                if k in ('url', 'key', 'verify'):
                    config[k] = v.strip()
    return config


class PyzzleAPI:
    def __init__(self,
                url=os.environ.get('PYZZLEAPI_URL'),
                key=os.environ.get('PYZZLEAPI_KEY')):

        dotrc = os.environ.get('PYZZLEAPI_RC', os.path.expanduser('~/.pyzzleapirc'))

        if url is None or key is None:
            if os.path.exists(dotrc):
                config = read_config(dotrc)

                if key is None:
                    key = config.get('key')

                if url is None:
                    url = config.get('url')
    
        if url is None or key is None:
            raise Exception('Missing/incomplete configuration file: %s' % (dotrc))
        
        self.url = url
        self.key = key

        self.headers = {
            "Accept": "application/json",
            "Authorization": "Bearer " + self.key,
            'Content-Type': 'application/json'
        }
    
    def get_all_puzzles(self):
        return requests.get(self.url, headers=self.headers)
    
    def add_puzzle(self, data):
        return requests.post(self.url, headers=self.headers, data=json.dumps(data))
    
    def get_puzzle(self, id):
        url = self.url+str(id)
        return requests.post(url, headers=self.headers)
    
    def edit_puzzle(self, id, data):
        url = self.url+str(id)
        return requests.patch(url, headers=self.headers, data=json.dumps(data))
    
    def delete_puzzle(self, id):
        url = self.url+str(id)
        return requests.delete(url, headers=self.headers)
