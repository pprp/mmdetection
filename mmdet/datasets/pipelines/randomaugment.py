import collections
import os
import random
import warnings

import cv2
import mmcv
import numpy
from PIL import Image, ImageOps, ImageEnhance
from mmcv.parallel import DataContainer
from mmcv.utils import build_from_cfg
from pycocotools import mask as mask_util

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES


def bbox2fields():
    bbox2label = {'gt_bboxes': 'gt_labels', 'gt_bboxes_ignore': 'gt_labels_ignore'}
    bbox2mask = {'gt_bboxes': 'gt_masks', 'gt_bboxes_ignore': 'gt_masks_ignore'}
    return bbox2label, bbox2mask


def invert(image, _):
    return ImageOps.invert(image)


def equalize(image, _):
    return ImageOps.equalize(image)


def solar1(image, magnitude):
    return ImageOps.solarize(image, int((magnitude / 10.) * 256))


def solar2(image, magnitude):
    return ImageOps.solarize(image, 256 - int((magnitude / 10.) * 256))


def solar3(image, magnitude):
    lut = []
    for i in range(256):
        if i < 128:
            lut.append(min(255, i + int((magnitude / 10.) * 110)))
        else:
            lut.append(i)
    if image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        return image


def poster1(image, magnitude):
    magnitude = int((magnitude / 10.) * 4)
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def poster2(image, magnitude):
    magnitude = 4 - int((magnitude / 10.) * 4)
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def poster3(image, magnitude):
    magnitude = int((magnitude / 10.) * 4) + 4
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def contrast1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    magnitude = 1.0 + -magnitude if random.random() > 0.5 else magnitude
    return ImageEnhance.Contrast(image).enhance(magnitude)


def contrast2(image, magnitude):
    return ImageEnhance.Contrast(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def contrast3(image, _):
    return ImageOps.autocontrast(image)


def color1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    magnitude = 1.0 + -magnitude if random.random() > 0.5 else magnitude
    return ImageEnhance.Color(image).enhance(magnitude)


def color2(image, magnitude):
    return ImageEnhance.Color(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def brightness1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    magnitude = 1.0 + -magnitude if random.random() > 0.5 else magnitude
    return ImageEnhance.Brightness(image).enhance(magnitude)


def brightness2(image, magnitude):
    return ImageEnhance.Brightness(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def sharpness1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    magnitude = 1.0 + -magnitude if random.random() > 0.5 else magnitude

    return ImageEnhance.Sharpness(image).enhance(magnitude)


def sharpness2(image, magnitude):
    return ImageEnhance.Sharpness(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def random_hsv(img, h_gain=0.015, s_gain=0.700, v_gain=0.400):
    random_gain = numpy.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=numpy.int16)
    lut_hue = ((x * random_gain[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * random_gain[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * random_gain[2], 0, 255).astype('uint8')

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype('uint8')
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


class Shear:
    def __init__(self, min_val=0.002, max_val=0.2):
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def _shear_img(results, magnitude, direction):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sheared = mmcv.imshear(img,
                                       magnitude,
                                       direction)
            results[key] = img_sheared.astype(img.dtype)

    @staticmethod
    def _shear_boxes(results, magnitude, direction):
        h, w, c = results['img_shape']
        if direction == 'horizontal':
            shear_matrix = numpy.stack([[1, magnitude], [0, 1]]).astype(numpy.float32)  # [2, 2]
        else:
            shear_matrix = numpy.stack([[1, 0], [magnitude, 1]]).astype(numpy.float32)
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])  # [4, 2, nb_box, 1]
            coordinates = coordinates[..., 0].transpose((2, 1, 0)).astype(numpy.float32)  # [nb_box, 2, 4]
            new_coords = numpy.matmul(shear_matrix[None, :, :], coordinates)  # [nb_box, 2, 4]
            min_x = numpy.min(new_coords[:, 0, :], axis=-1)
            min_y = numpy.min(new_coords[:, 1, :], axis=-1)
            max_x = numpy.max(new_coords[:, 0, :], axis=-1)
            max_y = numpy.max(new_coords[:, 1, :], axis=-1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _shear_masks(results, magnitude, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.shear((h, w),
                                       magnitude,
                                       direction)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            return results
        magnitude = numpy.random.uniform(self.min_val, self.max_val)
        if numpy.random.rand() > 0.5:
            magnitude *= -1
        direction = numpy.random.choice(['horizontal', 'vertical'])
        self._shear_img(results, magnitude, direction)
        self._shear_boxes(results, magnitude, direction)
        self._shear_masks(results, magnitude, direction)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


class Rotate:
    def __init__(self, min_val=1, max_val=45):
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def _rotate_img(results, angle, center, scale):
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(img, angle, center, scale)
            results[key] = img_rotated.astype(img.dtype)

    @staticmethod
    def _rotate_boxes(results, rotate_matrix):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])
            coordinates = numpy.concatenate((coordinates,
                                             numpy.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                                            axis=1)
            coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = numpy.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x = numpy.min(rotated_coords[:, :, 0], axis=1)
            min_y = numpy.min(rotated_coords[:, :, 1], axis=1)
            max_x = numpy.max(rotated_coords[:, :, 0], axis=1)
            max_y = numpy.max(rotated_coords[:, :, 1], axis=1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _rotate_masks(results, angle, center, scale):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, 0)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            return results
        h, w = results['img'].shape[:2]
        angle = numpy.random.randint(self.min_val, self.max_val)
        if numpy.random.rand() > 0.5:
            angle *= -1
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1)
        self._rotate_img(results, angle, center, 1)
        self._rotate_boxes(results, matrix)
        self._rotate_masks(results, angle, center, 1)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


class Translate:
    def __init__(self, min_val=1, max_val=256):
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def _translate_image(results, offset, direction):
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            results[key] = mmcv.imtranslate(img, offset, direction).astype(img.dtype)

    @staticmethod
    def _translate_boxes(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            if direction == 'horizontal':
                min_x = numpy.maximum(0, min_x + offset)
                max_x = numpy.minimum(w, max_x + offset)
            elif direction == 'vertical':
                min_y = numpy.maximum(0, min_y + offset)
                max_y = numpy.minimum(h, max_y + offset)

            results[key] = numpy.concatenate([min_x, min_y, max_x, max_y], axis=-1)

    @staticmethod
    def _translate_masks(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, 0)

    @staticmethod
    def _filter_invalid(results):
        bbox2label, bbox2mask = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > 0) & (bbox_h > 0)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]
        return results

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            return results
        offset = numpy.random.randint(self.min_val, self.max_val)
        if numpy.random.rand() > 0.5:
            offset *= -1
        direction = numpy.random.choice(['horizontal', 'vertical'])
        self._translate_image(results, offset, direction)
        self._translate_boxes(results, offset, direction)
        self._translate_masks(results, offset, direction)
        self._filter_invalid(results)
        return results


@PIPELINES.register_module()
class RandomAugment:
    def __init__(self):
        self.color_transforms = [color1, color2, solar1, solar2, solar3, invert,
                                 poster1, poster2, poster3, equalize, contrast1,
                                 contrast2, contrast3, sharpness1, sharpness2,
                                 brightness1, brightness2]
        self.geo_transforms = [Shear(), Rotate(), Translate()]

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            random_hsv(results['img'])
        else:
            image = results['img']
            image = Image.fromarray(image[:, :, ::-1])
            for transform in numpy.random.choice(self.color_transforms, 2):
                magnitude = min(10., max(0., random.gauss(9, 0.5)))
                image = transform(image, magnitude)
            results['img'] = numpy.array(image)[:, :, ::-1]

        transform = numpy.random.choice(self.geo_transforms)
        return transform(results)

    def __repr__(self):
        return self.__class__.__name__
