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
        t = f.split('/')[1]
        draw.text((x_offset, 10), f'{i}:{t}', fill='red', font=font)
        x_offset += _acd_imgs[i].size[0]

    # new_im.show()
    new_im.save(out_img)

    print(f'Merge to {out_img}')

if __name__ == '__main__':

    # for out_dir in []:
    init_method = 'omniscient'
    R = 50
    D = 35
    prop = 1.0

    out_dir = 'out4'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # varies radius: fixed K and noise_cov
    for noise_cov in [25]: #[1, 9, 25, 100, 250,]:
        for K in range(5, 30 + 1, 5):
            try:
                acd_imgs = []
                omniscient_imgs = []
                for r in [1, 2, 3, 4, 5, 6]:
                    _out_dir = f'out/R_{R}-D_{D}-K_{K}-r_{r}-cov_{noise_cov}-p_{prop}/{init_method}'
                    fs = os.listdir(_out_dir)
                    acd_imgs+=[os.path.join(_out_dir, f) for f in fs if f.endswith('_acd.png')]
                    omniscient_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_misc.png')]

                merge_imgs(acd_imgs, omniscient_imgs, out_img=os.path.join(out_dir, f'R_{R}-D_{D}-K_{K}-cov_{noise_cov}-K_{prop}_{init_method}.png'))
            except Exception as e:
                traceback.print_exc()

    # # varies K: fixed r and noise_cov
    # for noise_cov in [1, 9, 25, 100, 250,]:
    #     for r in [1, 2, 3, 4, 5, 6]:
    #         try:
    #             acd_imgs = []
    #             omniscient_imgs = []
    #             for K in range(5, 35 + 1, 5):
    #                 _out_dir = f'out/R_{R}-D_{D}-r_{r}-cov_{noise_cov}-pxK_{prop}/{init_method}'
    #                 fs = os.listdir(_out_dir)
    #                 acd_imgs+=[os.path.join(_out_dir, f) for f in fs if f.endswith('_acd.png')]
    #                 omniscient_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_misc.png')]
    #
    #             merge_imgs(acd_imgs, omniscient_imgs, out_img=os.path.join(out_dir, f'R_{R}-D_{D}-r_{r}-cov_{noise_cov}-pxK_{prop}_{init_method}.png'))
    #         except Exception as e:
    #             traceback.print_exc()
    #
    # # varies cov: fixed r and K
    # for K in range(5, 35 + 1, 5):
    #     for r in [1, 2, 3, 4, 5, 6]:
    #         try:
    #             acd_imgs = []
    #             omniscient_imgs = []
    #             for noise_cov in [1, 9, 25, 100, 250,]:
    #                 _out_dir = f'out/R_{R}-D_{D}-K_{K}-r_{r}-cov_{noise_cov}-pxK_{prop}'
    #                 fs = os.listdir(_out_dir)
    #                 acd_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_acd.png')]
    #                 omniscient_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_misc.png')]
    #
    #             merge_imgs(acd_imgs, omniscient_imgs,
    #                        out_img=os.path.join(out_dir, f'R_{R}-D_{D}-K_{K}-r_{r}-pxK_{prop}.png'))
    #         except Exception as e:
    #             traceback.print_exc()

    # varies radius: fixed K and noise_cov
    noise_cov = 25
    # for prop in [0.05, 0.1, 0.25, 0.4, 0.5]:
    #     for K in range(5, 35 + 1, 5):
    #         try:
    #             acd_imgs = []
    #             omniscient_imgs = []
    #             for r in [1, 2, 3, 4, 5, 6]:
    #                 _out_dir = f'out/R_{R}-D_{D}-K_{K}-r_{r}-cov_{noise_cov}-pxK_{prop}/{init_method}'
    #                 fs = os.listdir(_out_dir)
    #                 acd_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_acd.png')]
    #                 omniscient_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_misc.png')]
    #
    #             merge_imgs(acd_imgs, omniscient_imgs, out_img=os.path.join(out_dir,
    #                                                                        f'R_{R}-D_{D}-K_{K}-cov_{noise_cov}-pxK_{prop}_{init_method}.png'))
    #         except Exception as e:
    #             traceback.print_exc()

    # for r in [1, 2, 3, 4, 5, 6]:
    #     for K in range(5, 35 + 1, 5):
    #         try:
    #             acd_imgs = []
    #             omniscient_imgs = []
    #             for prop in [0.05, 0.1, 0.25, 0.4, 0.5]:
    #                 _out_dir = f'out/R_{R}-D_{D}-K_{K}-r_{r}-cov_{noise_cov}-pxK_{prop}/{init_method}'
    #                 fs = os.listdir(_out_dir)
    #                 acd_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_acd.png')]
    #                 omniscient_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_misc.png')]
    #
    #             merge_imgs(acd_imgs, omniscient_imgs, out_img=os.path.join(out_dir,
    #                                                                        f'R_{R}-D_{D}-K_{K}-r_{r}-cov_{noise_cov}_{init_method}.png'))
    #         except Exception as e:
    #             traceback.print_exc()


    # noise_covs = [1, 9]  # [1, 9, 25, 100, 250]
    # R=50
    # noise_cov = 9
    # for r in [1, 3]:
    #     for K in range(5, 35 + 1, 5):
    #         try:
    #             acd_imgs = []
    #             omniscient_imgs = []
    #             for noise_mean in [1, 5, 10, 15, 25]:  # [1, 9, 25, 100, 250]
    #                 prop = 0.2
    #                 _out_dir = f'out_noise/R_{R}-D_{D}-K_{K}-r_{r}-mu_{noise_mean}-cov_{noise_cov}-pxK_{prop}/{init_method}'
    #                 fs = os.listdir(_out_dir)
    #                 acd_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_acd.png')]
    #                 omniscient_imgs += [os.path.join(_out_dir, f) for f in fs if f.endswith('_misc.png')]
    #
    #             merge_imgs(acd_imgs, omniscient_imgs, out_img=os.path.join(out_dir,
    #                                                                        f'R_{R}-D_{D}-K_{K}-r_{r}-noise_mean-cov_{noise_cov}_{init_method}.png'))
    #         except Exception as e:
    #             traceback.print_exc()

    print('finished')