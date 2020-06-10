import requests
import json

class PyzzleAPI:
    def __init__(self, api_token):
        self.url = 'https://pyzzle.du.r.appspot.com/api/puzzles/'
        self.headers = {
            "Accept": "application/json",
            "Authorization": "Bearer " + api_token,
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