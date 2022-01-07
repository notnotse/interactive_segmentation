import cv2
import torch

from interactive_demo.controller import InteractiveController
from interactive_demo.democontroller import ImageController
from isegm.utils import exp
from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp
from timer import time_this

add_point = None


def show_cv2_img(img_name, img, mouse_callback=None):
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)

    if mouse_callback is not None:
        cv2.setMouseCallback(img_name, mouse_callback)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_points_callback(event, x, y, flags, params):
    global add_point

    if event == cv2.EVENT_LBUTTONDOWN:
        crop_img(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        crop_img(x, y)

@time_this
def crop_img(x, y, size=350):
    global img
    global predictor
    size = size // 2
    small_img = img[y - size:y + size, x - size:x + size]

    predictor.set_image(small_img)
    res_img = predictor.add_click(size, size, True)

    img[y - size:y + size, x - size:x + size] = res_img

    #small_img = cv2.circle(small_img, (size, size), radius=100, color=(0, 0, 255), thickness=-1)


def load_model():
    MODEL_NAME = "010"
    DEVICE_NAME = "cuda:0"
    torch.backends.cudnn.deterministic = True
    cfg = exp.load_config_file("config.yml", return_edict=True)
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, MODEL_NAME)
    model = utils.load_is_model(checkpoint_path, DEVICE_NAME, cpu_dist_maps=True)

    return ImageController(model, DEVICE_NAME,
                           predictor_params={'brs_mode': 'NoBRS'})


predictor = load_model()
img = cv2.imread("bild.jpg")
show_cv2_img("Img", img, add_points_callback)
show_cv2_img("Result", img, add_points_callback)

# load_model()
