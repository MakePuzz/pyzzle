class Word(str):
    """wc value and string flags"""
    def __new__(cls, content, *args, **kwargs):
        return super(Word, cls).__new__(cls, content)
    
    def __init__(self, content, weight):
        self.weight = weight