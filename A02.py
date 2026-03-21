# A02. 必要パッケージを入れ直す（このあとランタイム再起動）
!pip -q uninstall -y Pillow pillow torchvision transformers qwen-vl-utils accelerate
!pip -q install --no-cache-dir -U Pillow torchvision accelerate qwen-vl-utils
!pip -q install --no-cache-dir -U git+https://github.com/huggingface/transformers

import os
os.kill(os.getpid(), 9)
