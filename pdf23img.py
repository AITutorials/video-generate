
from pdf2image import convert_from_path
# pip install pdf2image
# https://github.com/Belval/pdf2image
import os

file_path = './Marp.pdf'
dir_path = './img'

def pdf2image2(file_path, dir_path):
  images = convert_from_path(file_path, dpi=200)
  for image in images:
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    image.save(dir_path + f'/img_{images.index(image)}.png', 'PNG')
pdf2image2(file_path, dir_path)
