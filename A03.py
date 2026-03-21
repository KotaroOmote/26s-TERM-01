# A03. 再起動後のimportと動作確認
import os
import re
import json
import csv
import time
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import transformers
import torchvision
import PIL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

print("torch        :", torch.__version__)
print("transformers :", transformers.__version__)
print("torchvision  :", torchvision.__version__)
print("Pillow       :", PIL.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
