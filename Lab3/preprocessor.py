import os
from PIL import Image
import glob

def resize_images(input_directory, output_directory, dimensions=(224, 224)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)
                img = Image.open(file_path)
                resized_img = img.resize(dimensions, Image.Resampling.LANCZOS)
                
                rel_dir = os.path.relpath(root, input_directory)
                new_dir = os.path.join(output_directory, rel_dir)
                
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                
                output_file = os.path.join(new_dir, file)
                resized_img.save(output_file)

prefix = './mydataset/'
input_directories = ['Test', 'Train', 'Val']
output_directories = ['Test_resized', 'Train_resized', 'Val_resized']

for input_dir, output_dir in zip(input_directories, output_directories):
    resize_images(prefix+input_dir, prefix+output_dir)

