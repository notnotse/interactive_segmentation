from isegm.data.datasets.mask import MaskDataset
from isegm.utils.exp_imports.default import *


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (320, 480)
    model_cfg.num_max_points = 24

    model = HRNetModel(width=18, ocr_width=64, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=5,
                       with_prev_mask=True)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4

    train_augmentator = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(scale_limit=(0.0, 0.2), border_mode=cv2.BORDER_CONSTANT, rotate_limit=(45, 45), p=1),
        Transpose(),
        Resize(450, 450),
        RandomBrightnessContrast(p=0.5, brightness_limit=(0.0, 0.1), contrast_limit=(0.0, 0.15)),
        RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        Sharpen(p=0.5),
        HueSaturationValue(p=0.5)
    ])

    points_sampler = MultiPointSampler(model_cfg.num_max_points,
                                       prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)
    """
    trainset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.30
    )
    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)
    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )
    """

    trainset = MaskDataset(
        dataset_path=cfg.MASK_PATH_TRAIN,
        min_object_area=500,
        epoch_len=5000,
        points_sampler=points_sampler,
        augmentator=train_augmentator)

    valset = MaskDataset(dataset_path=cfg.MASK_PATH_VAL,
                         min_object_area=500,
                         epoch_len=5000,
                         points_sampler=points_sampler)

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[200, 220], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (200, 1)],
                        image_dump_interval=3000,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)

    trainer.run(num_epochs=1000)
