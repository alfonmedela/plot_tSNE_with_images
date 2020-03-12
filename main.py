import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import glob

def change_colors(img,width, color):

    a,b,c = img.shape
    x_min = 0
    x_max = b
    y_min = 0
    y_max = a
    for num in range(width):
        for channel in range(3):
            img[y_min+num, x_min:x_max, channel] = color[channel]
            img[y_max - 1 - num, x_min:x_max, channel] = color[channel]

            img[y_min:y_max, x_min + num, channel] = color[channel]
            img[y_min:y_max, x_max - 1 - num, channel] = color[channel]
    return img

def visualize_scatter_with_images(X_2d_data, images, figsize=(45, 45), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.savefig('tsne.png')

def load_data():

    root_path = 'imgs/'
    dirs = glob.glob(root_path + '*')

    colors = [[255, 0, 0],
              [0, 128, 0],
              [255, 255, 0]]

    imgs = []
    labels = []
    n_class = 0
    for dir in dirs:
        path = dir + '\\*'
        images = glob.glob(path)
        color = colors[n_class]
        for image in images:
            image = cv2.imread(image)
            image = cv2.resize(image, (224, 224))
            image = change_colors(image, 5, color)
            imgs.append(image)
            labels.append(n_class)
        n_class += 1

    x_tsne = np.array([[-10, -8], [-9.5, -10], [-8, -8], [0, 0], [2, 0], [1, 4], [-10, 8], [-9, 8], [-8, 10]])

    return  x_tsne, labels, imgs

if __name__ == '__main__':

    x_tsne, labels, imgs = load_data()

    label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    label_ids = np.array([label_to_id_dict[x] for x in labels])

    visualize_scatter_with_images(x_tsne, images=imgs, image_zoom=2.5)















