"""
https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python

"""
import os.path
import sys
import traceback

from PIL import Image, ImageDraw
from PIL import ImageFont


def merge_imgs(acd_imgs, omniscient_imgs, out_img = 'merge.png'):
    print(len(acd_imgs), acd_imgs)

    _acd_imgs = [Image.open(x) for x in acd_imgs]
    _omniscient_imgs = [Image.open(x) for x in omniscient_imgs]

    widths, heights = zip(*(i.size for i in _omniscient_imgs))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height*2))

    x_offset = 0
    for i in range(0, len(acd_imgs), 1):
        new_im.paste(_acd_imgs[i], (x_offset, 0))  # upper left
        new_im.paste(_omniscient_imgs[i], (x_offset, max_height))
        # draw.text((x_offset, -2), f'{i}', fill='red', fontsize=20)
        x_offset += _acd_imgs[i].size[0]

    # add text to each column
    x_offset = 300
    draw = ImageDraw.Draw(new_im)
    # https://stackoverflow.com/questions/33544897/imagefont-io-error-cannot-open-resource
    # font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 100)
    font = ImageFont.truetype('arial.ttf', size=100)
    for i in range(0, len(acd_imgs), 1):
        f = acd_imgs[i]
        tmp = f.split('/')
        t = f'{tmp[1]}-{tmp[2]}'
        draw.text((x_offset, 10), f'{i//2}:{t}', fill='red', font=font)
        x_offset += _acd_imgs[i].size[0]

    # new_im.show()
    new_im.save(out_img)

    print(f'Merge to {out_img}')

if __name__ == '__main__':


    R = 1000
    out_dir = 'out_plot'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for R in [500, 1000]:
        try:
            acd_imgs = []
            omniscient_imgs = []
            for alg_method in ['diffdim', 'diffrad', 'diffvar']:
                for init_method in ['omniscient', 'random']:
                    _out_dir = f'out/{alg_method}/{init_method}/R_{R}-S_200'
                    fs = os.listdir(_out_dir)
                    acd_imgs+=[os.path.join(_out_dir, f) for f in fs if f.startswith('acd_')]
                    omniscient_imgs += [os.path.join(_out_dir, f) for f in fs if f.startswith('misc_')]

            merge_imgs(acd_imgs, omniscient_imgs, out_img=os.path.join(out_dir, f'R_{R}.png'))
        except Exception as e:
            traceback.print_exc()


    print('finished')