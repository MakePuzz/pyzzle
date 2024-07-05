class Word(str):
    """Word with weight"""
    def __new__(cls, content, *args, **kwargs):
        return super(Word, cls).__new__(cls, content)
    
    def __init__(self, content, weight=0):
        self.weight = weight
