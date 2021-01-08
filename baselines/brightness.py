from loading.datasets import make_att_specs, SingleImageDataset
from loading.loaders import get_data_for_spec, make_basic_transform
from utils.experiment_utils import get_default_set_spec_dict
from utils.utils import compute_normalization_parameters


def compute_brightness(set_name, device_name):
    """A method for computing the average brightness of a set of images. """
    att_spec_dict = make_att_specs()
    set_spec_dict = get_default_set_spec_dict()
    # e.g. "ESAOTE_6100_train"
    set_spec = set_spec_dict[set_name]
    images = get_data_for_spec(set_spec, loader_type='image', dropna_values=False)
    # e.g. "ESAOTE_6100"
    transform = make_basic_transform(device_name, limit_image_size=False, to_tensor=True)

    ds = SingleImageDataset(image_frame=images, root_dir=set_spec.img_root_path, attribute_specs=[att_spec_dict['Sex']],
                            return_attribute_dict=False, transform=transform,
                            use_one_channel=True)


    mean, std  = compute_normalization_parameters(ds, 1)

    print(mean)
    print(std)

# Average brightness values determined using the compute_brightness method
brightness_dict = {'Philips_iU22': 0.1693, 'ESAOTE_6100': 0.2328, 'GE_Logiq_E': 0.2314}


def get_brightness_factor(source_set,target_set):
    """Compute a simple brightness adjustment factor based on the average brightness values of the two devices."""
    source_bright = brightness_dict[source_set]
    target_bright = brightness_dict[target_set]
    factor = source_bright / target_bright
    return factor