import torch
from torch.nn.functional import interpolate

from isegm.inference.clicker import Click
from .base import BaseTransform
from ...utils.misc import get_bbox_from_mask, expand_bbox, clamp_bbox


class ZoomOut(BaseTransform):
    def __init__(self,
                 target_size=400,
                 start_size=180,
                 expansion_ratio=1.4,
                 min_crop_size=200,
                 recompute_thresh_iou=0.5,
                 prob_thresh=0.50):
        super().__init__()
        self.start_size = start_size
        self.target_size = target_size
        self.min_crop_size = min_crop_size
        self.expansion_ratio = expansion_ratio
        self.recompute_thresh_iou = recompute_thresh_iou
        self.prob_thresh = prob_thresh

        self._input_image_shape = None
        self._prev_probs = None
        self._object_roi = None
        self._roi_image = None

    def transform(self, image_nd, clicks_lists):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = False

        clicks_list = clicks_lists[0]
        last_click = clicks_list[-1]

        self._input_image_shape = image_nd.shape
        _, _, height, width = image_nd.shape

        y, x = last_click.coords
        xmin = 0 if x - self.start_size < 0 else x - self.start_size
        xmax = width - 1 if x + self.start_size > width else x + self.start_size
        ymin = 0 if y - self.start_size < 0 else y - self.start_size
        ymax = height - 1 if y + self.start_size > height else y + self.start_size
        # TODO: Look over datasets: Does the imgs have to be resized?? Can i be org size?
        # TODO: If we click outside of current self._object_roi -> Enlarge it by the click + margin.
        # TODO: If we are still within boundaries -> return image_nd, clicks_lists

        self._object_roi = (ymin, ymax, xmin, xmax)

        print("Clicked: x:", x, " y:", y)
        print("x:", xmin, " xmax:", xmax)
        print("y:", ymin, " ymax:", ymax)

        self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)
        self.image_changed = True

        tclicks_lists = [self._transform_clicks(clicks_list)]
        return self._roi_image.to(image_nd.device), tclicks_lists

    def inv_transform(self, prob_map):
        print("inv_transform - ZOOM OUT")

        if self._object_roi is None:
            self._prev_probs = prob_map.cpu().numpy()
            return prob_map

        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        prob_map = interpolate(prob_map,
                               size=(rmax - rmin + 1, cmax - cmin + 1),
                               mode='bilinear',
                               align_corners=True)

        new_prob_map = torch.zeros(*self._input_image_shape, device=prob_map.device, dtype=prob_map.dtype)
        new_prob_map[:, :, rmin:rmax + 1, cmin:cmax + 1] = prob_map

        self._prev_probs = new_prob_map.cpu().numpy()
        return new_prob_map

    def get_state(self):
        roi_image = self._roi_image.cpu() if self._roi_image is not None else None
        return self._input_image_shape, self._object_roi, self._prev_probs, roi_image, self.image_changed

    def set_state(self, state):
        self._input_image_shape, self._object_roi, self._prev_probs, self._roi_image, self.image_changed = state

    def reset(self):
        self._input_image_shape = None
        self._object_roi = None
        self._prev_probs = None
        self._roi_image = None
        self.image_changed = False

    def _transform_clicks(self, clicks_list):
        if self._object_roi is None:
            return clicks_list

        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]

        transformed_clicks = []

        for click in clicks_list:
            new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
            transformed_clicks.append(Click(is_positive=click.is_positive, coords=(new_r, new_c)))

        return transformed_clicks


def get_object_roi(pred_mask, clicks_list, expansion_ratio, min_crop_size):
    pred_mask = pred_mask.copy()

    for click in clicks_list:
        if click.is_positive:
            pred_mask[int(click.coords[0]), int(click.coords[1])] = 1

    bbox = get_bbox_from_mask(pred_mask)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    return bbox


def get_roi_image_nd(image_nd, object_roi, target_size):
    rmin, rmax, cmin, cmax = object_roi

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    if isinstance(target_size, tuple):
        new_height, new_width = target_size
    else:
        scale = target_size / max(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

    with torch.no_grad():
        roi_image_nd = image_nd[:, :, rmin:rmax + 1, cmin:cmax + 1]
        roi_image_nd = torch.nn.functional.interpolate(roi_image_nd, size=(new_height, new_width), mode='bilinear',
                                                       align_corners=True)

    return roi_image_nd


def check_object_roi(object_roi, clicks_list):
    for click in clicks_list:
        if click.is_positive:
            if click.coords[0] < object_roi[0] or click.coords[0] >= object_roi[1]:
                return False
            if click.coords[1] < object_roi[2] or click.coords[1] >= object_roi[3]:
                return False

    return True