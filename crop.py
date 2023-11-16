"""
Crops plate images to identifiable size.

Usage:
    crop.py --img_dir=<img_dir> --out_dir=<out_dir>
"""
from tqdm import tqdm
from PIL import Image
from docopt import docopt
import os

if __name__ == '__main__':
    args = docopt(__doc__)
    img_dir = args['--img_dir']
    out_dir = args['--out_dir']

    for img in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img)
        raw_img = Image.open(img_path)
        cropped = raw_img.crop((1300,1700,1600,2000))
        cropped.save(os.path.join(out_dir, img))
