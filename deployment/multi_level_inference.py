import torch

from loading.loaders import make_basic_transform
from models.multi_input import MultiInputNet

def load_model_from_checkpoint(checkpoint_path):
    combined_dict = torch.load(checkpoint_path)

    model_param_dict = combined_dict['model_param_dict']
    model = MultiInputNet(**model_param_dict)
    state_dict = combined_dict['model_state_dict']
    model.load_state_dict(state_dict)
    return model

def load_transform_from_checkpoint(checkpoint_path, ignore_resize=True):
    combined_dict = torch.load(checkpoint_path)
    transform_dict = combined_dict['transform_dict']
    # decide whether to override the resize
    if ignore_resize:
        transform_dict['resize_option_name'] = None
    transform = make_basic_transform(**transform_dict)
    return transform

if __name__ == '__main__':
    # path to param dict
   #  model_param_dict = torch.load('multi_input_test_params.pth')

    # path to state dict
  #  state_dict = torch.load('multi_input_test.pth')


    model = load_model_from_checkpoint('checkpoints/pref_checkpoint_14.pt')
    model.eval()

    # data loading
    # todo work out details

    ds = torch.load('test_dataset.pth')

    imgs, y = ds[0]
    n_img = torch.tensor([imgs.shape[0]])

    pred_output = model((imgs, n_img))


