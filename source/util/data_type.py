# Datatype to differentiate between given images by type
from enum import Enum


class Type(Enum):
    normal = 1,
    depth = 2,
    sketch = 3,
    silhouette = 4
