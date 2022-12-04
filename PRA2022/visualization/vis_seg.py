import os
import json

import numpy as np
from pycocotools.coco import COCO

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


with open("../results/result.json", "r") as f:
    json_result = json.load(f)['result']



for data in json_result:
    segs = data['segmentation']
    if len(segs) != 0:
        im = Image.open(os.path.join("../datasets/my_dataset/images", str(data['image_id']).zfill(5) + ".jpg"))
        # Save image and its labeled version.
        plt.axis("off")
        plt.imshow(np.asarray(im))
        ax = plt.gca()
        ax.set_autoscale_on(False)

        if data['type'] == 8:
            c = [0.4, 0.4, 0.4]
        elif data['type'] == 9:
            c = [0.6, 0.6, 0.6]
        else:
            c = [0.8, 0.8, 0.8]

        # c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        # print(c)
        polygons = []
        color = []

        for seg in segs:
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
            polygons.append(Polygon(poly))
            color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        plt.show()
        # break

