import pandas as pd
import numpy as np

import DataLoading as dl

filename = "all_resources.csv"
resources_chunker = pd.read_csv(
                        dl.getDataFilePath(filename),
                        iterator=True,
                        chunksize=50000)
                        
resources = resources_chunker.get_chunk(50000)

