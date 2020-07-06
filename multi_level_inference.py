import torch

from baselines import get_default_set_spec_dict
from loading.loading_utils import make_set_specs
from models.multi_input import MultiInputNet

if __name__ == '__main__':
    # path to param dict
   #  model_param_dict = torch.load('multi_input_test_params.pth')

    # path to state dict
  #  state_dict = torch.load('multi_input_test.pth')

    combined_dict = torch.load('checkpoints/pref_checkpoint_13.pt')

    model_param_dict = combined_dict['model_param_dict']
    model = MultiInputNet(**model_param_dict)
    state_dict = combined_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # data loading
    # todo work out details

    ds = torch.load('test_dataset.pth')

    imgs, y = ds[0]
    n_img = torch.tensor([imgs.shape[0]])

    pred_output = model((imgs, n_img))


