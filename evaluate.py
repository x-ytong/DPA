import argparse
import os
from PIL import Image
import numpy as np
np.set_printoptions(suppress=True)

Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser(description="Evaluation")
parser.add_argument('--labelpath', default='None', type=str,
                    help='path of ground truth')
parser.add_argument('--resultpath', default='None', type=str,
                    help='path of results')

color = [[  0,   0,   0],
         [200,   0,   0],       # industrial area
         [250,   0, 150],       # urban residential
         [200, 150, 150],       # rural residential
         [250, 200, 150],       # stadium
         [150, 150,   0],       # square
         [250, 150, 150],       # road
         [250, 150,   0],       # overpass
         [250, 200, 250],       # railway station
         [200, 150,   0],       # airport
         [  0, 200,   0],       # paddy field
         [150, 250,   0],       # irrigated field
         [150, 200, 150],       # dry cropland
         [200,   0, 200],       # garden land
         [150,   0, 250],       # arbor forest
         [150, 150, 250],       # shrub forest
         [200, 150, 200],       # park land
         [250, 200,   0],       # natural meadow
         [200, 200,   0],       # artificial meadow
         [  0,   0, 200],       # river
         [  0, 150, 200],       # lake
         [  0, 200, 250],       # pond
         [150, 200, 250],       # fish pond
         [250, 250, 250],       # snow
         [200, 200, 200]]       # bareland


def main():
    args = parser.parse_args()
    confusion_matrix = np.zeros((len(color) - 1, len(color) - 1))
    lb_class = []
    for lb_name, rt_name in zip(os.listdir(args.labelpath), os.listdir(args.resultpath)):
        label = np.array(Image.open(args.labelpath + lb_name).convert('RGB'))
        result = np.array(Image.open(args.resultpath + rt_name).convert('RGB'))
        label = encode_annotation(label, color)
        result = encode_annotation(result, color)
        cm = cm_generation(label, result, len(color) - 1)
        confusion_matrix = confusion_matrix + cm
        lb_class = np.concatenate((lb_class, np.unique(label)), axis=0)
    # kappa = kappa_coefficient(confusion_matrix)
    num = len(np.unique(lb_class)) - 1
    oa = overall_accuracy(confusion_matrix)
    mf1 = f1_score(confusion_matrix) / num
    miou = mean_intersection_over_union(confusion_matrix) / num
    ua = users_accuracy(confusion_matrix)
    pa = producers_accuracy(confusion_matrix)
    print('{}: OA: {}, mf1: {}, mIoU: {}.'.format(
        args.resultpath, round(oa * 10000), round(mf1 * 10000), round(miou * 10000)))
    print('users_ac: {}.'.format(np.around(ua*10000)))
    print('producers_ac: {}.'.format(np.around(pa*10000)))


def encode_annotation(mask, cr_bar):
    mask = mask.astype(int)
    index = np.zeros((mask.shape[0], mask.shape[1]))
    index = index.astype(int)
    for ii, cr in enumerate(np.array(cr_bar)):
        index[np.where(np.all(mask == cr, axis=-1))] = ii
    index = index.astype(int)
    return index


def cm_generation(label, result, cl_num):
    mask = label > 0
    count = cl_num * (label[mask] - 1) + (result[mask] - 1)
    count = np.bincount(count, minlength=cl_num**2)
    cm = count.reshape(cl_num, cl_num)
    return cm


def kappa_coefficient(confusion_matrix):
    po = np.diag(confusion_matrix).sum()
    po = po / confusion_matrix.sum()
    pe = np.inner(confusion_matrix.sum(axis=1), confusion_matrix.sum(axis=0))
    pe = pe / confusion_matrix.sum()**2
    kappa = (po - pe) / (1 - pe)
    return kappa


def overall_accuracy(confusion_matrix):
    oa = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return oa


def mean_intersection_over_union(confusion_matrix):
    inter = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    iou = np.sum(np.divide(inter, union, out=np.zeros_like(union, dtype=float), where=union != 0))
    return iou


def users_accuracy(confusion_matrix):
    pre = confusion_matrix.sum(axis=0)
    ua = np.divide(np.diag(confusion_matrix), pre, out=np.zeros_like(pre, dtype=float), where=pre != 0)
    return ua


def producers_accuracy(confusion_matrix):
    tru = confusion_matrix.sum(axis=1)
    pa = np.divide(np.diag(confusion_matrix), tru, out=np.zeros_like(tru, dtype=float), where=tru != 0)
    return pa


def f1_score(confusion_matrix):
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    fra = tp + (fp + fn) / 2
    f1 = np.sum(np.divide(tp, fra, out=np.zeros_like(fra, dtype=float), where=fra != 0))
    return f1


if __name__ == '__main__':
    main()



