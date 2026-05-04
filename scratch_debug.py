import os
import sys

sys.path.insert(0, os.path.abspath("skin-ml-app"))

from models.cnn import predict_cnn

# Provide a dummy image
img_path = r"C:\Users\hamma\Downloads\skin type.v1i.folder\train\normal\0001_jpg.rf.02b94a0e417726decfba917d5e4630a9.jpg"

try:
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    print("Testing predict_cnn...")
    result = predict_cnn(file_bytes)
    print("Success:", result.keys())
except Exception as e:
    import traceback
    traceback.print_exc()
