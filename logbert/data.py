"""Data processing"""
import re

class DrainParser:
    def __init__(self, depth=4, sim_threshold=0.4, max_children=100):
        self.log_clusters = {}
        self.cluster_id = 0
        
    def parse(self, log_message):
        tokens = re.findall(r'\w+', log_message.lower())
        key = ' '.join(tokens[:5])
        if key not in self.log_clusters:
            self.log_clusters[key] = self.cluster_id
            self.cluster_id += 1
        return self.log_clusters[key]
    
    def fit(self, messages):
        for msg in messages:
            self.parse(msg)
        return self
    
    def transform(self, messages):
        return [self.parse(msg) for msg in messages]

class BalancedSessionBasedBGLLoader:
    def __init__(self, window_size=150, stride=75):
        self.window_size = window_size
        self.stride = stride