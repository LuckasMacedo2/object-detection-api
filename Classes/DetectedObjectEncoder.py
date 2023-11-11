import json
import numpy as np

class DetectedObjectEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.int64):
            return int(o)  # Converter int64 para int
        elif isinstance(o, np.float64):
            return float(o)  # Converter float64 para float
        elif isinstance(o, np.float32):
            return float(o)  # Converter float32 para float
        return super().default(o)