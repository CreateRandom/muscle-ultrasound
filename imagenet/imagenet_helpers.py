import ast
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def get_transform():
    image_net_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return image_net_transform


def get_imagenet_class_dict():
    dictionary = {}
    with open("imagenet/image_idx_to_label.txt", "r") as f:
        contents= f.read()
        dictionary = ast.literal_eval(contents)
    return dictionary
from torch import nn

class NetworkPart(nn.Module):
    def __init__(self, original_model, n_to_omit):
        super(NetworkPart, self).__init__()
        max_to_omit = len(list(original_model.children()))
        n_to_omit = min(max_to_omit, n_to_omit)
        self.features = nn.Sequential(*list(original_model.children())[:-n_to_omit])

    def forward(self, x):
        x = self.features(x)
        return x.flatten()