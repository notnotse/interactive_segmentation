from albumentations import ImageOnlyTransform
from albumentations.core.serialization import SERIALIZABLE_REGISTRY


def remove_image_only_transforms(sdict):
    if not 'transforms' in sdict:
        return sdict

    keep_transforms = []
    for tdict in sdict['transforms']:
        cls = SERIALIZABLE_REGISTRY[tdict['__class_fullname__']]
        if 'transforms' in tdict:
            keep_transforms.append(remove_image_only_transforms(tdict))
        elif not issubclass(cls, ImageOnlyTransform):
            keep_transforms.append(tdict)
    sdict['transforms'] = keep_transforms

    return sdict
