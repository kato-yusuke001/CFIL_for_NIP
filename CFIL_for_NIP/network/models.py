import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()],
        ["sigmoid", nn.Sigmoid()]
    ])[activation]


def addCoordinates(image):

    batch_size, _, image_height, image_width = image.size()

    y_coords = 2.0 * torch.arange(image_height).unsqueeze(
        1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
    x_coords = 2.0 * torch.arange(image_width).unsqueeze(
        0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

    coords = torch.stack((y_coords, x_coords), dim=0)



    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)

    image = torch.cat((coords.to(image.device), image), dim=1)

    return image



class CNNet(nn.Module):
  def __init__(self, in_dim = 32, in_chans = 3, out_feat = 4, filters = [32, 128, 128, 256, 64], lin_features = 128, kernels = [5, 3, 3, 3, 1], dim_s = [1,1,2,2,1], activation = "relu", drop_p = 0.1):
    super().__init__()
    self.convs = nn.ModuleList([Conv2dAuto(in_channels = in_chans, out_channels = filters[0], stride = dim_s[0], kernel_size = kernels[0])] +
                               [Conv2dAuto(in_channels = filters[i-1], out_channels = filters[i], stride = dim_s[i], kernel_size = kernels[i]) for i in range(1, len(filters))])
    self.batch_norms = nn.ModuleList([nn.BatchNorm2d(num_features= filters[i]) for i in range(len(filters))])
    self.setup(torch.rand(1,in_chans, in_dim, in_dim))
    self.denses = nn.ModuleList([nn.Linear(in_features = self.lin_in_shape, out_features=lin_features),
                                 nn.Linear(in_features = lin_features, out_features=out_feat)])
    self.act = activation_func(activation)
    self.drop = nn.Dropout(p=drop_p)

  def setup(self, x):
    for l in self.convs:
      x = l(x)
      print("setup x shape", x.size())
    self.lin_in_shape = x.shape[-1]*x.shape[-2]*x.shape[-3]
    print("lin shape", self.lin_in_shape)

  def forward(self, x):
    batch = x.size()[0]
    for l, b in zip(self.convs, self.batch_norms):
      x = self.drop(self.act(b(l(x))))

    x = x.view(batch, -1)
    x = self.denses[0](x)
    x = self.act(x)
    x = self.drop(x)
    x = self.denses[1](x)
    return x


class CordCNNet(nn.Module):
  def __init__(self, in_dim = 32, in_chans = 3, out_feat = 4, filters = [32, 128, 128, 256, 64], kernels = [5, 3, 3, 3, 1], dim_s = [1,1,2,2,1], activation = "relu", drop_p = 0.1):
    super().__init__()
    self.convs = nn.ModuleList([Conv2dAuto(in_channels = in_chans + 2, out_channels = filters[0], stride = dim_s[0], kernel_size = kernels[0])] +
                               [Conv2dAuto(in_channels = filters[i-1] + 2, out_channels = filters[i], stride = dim_s[i], kernel_size = kernels[i]) for i in range(1, len(filters))])
    self.batch_norms = nn.ModuleList([nn.BatchNorm2d(num_features= filters[i]) for i in range(len(filters))])
    self.setup(torch.rand(1,in_chans, in_dim, in_dim))
    self.denses = nn.ModuleList([nn.Linear(in_features = self.lin_in_shape, out_features=128),
                                 nn.Linear(in_features = 128, out_features=out_feat)])
    self.act = activation_func(activation)
    self.drop = nn.Dropout(p=drop_p)

  def setup(self, x):
    for l in self.convs:
      x = addCoordinates(x)
      x = l(x)
      print("setup x shape", x.size())
    self.lin_in_shape = x.shape[-1]*x.shape[-2]*x.shape[-3]
    print("lin shape", self.lin_in_shape)

  def forward(self, x):
    batch = x.size()[0]

    for l, b in zip(self.convs, self.batch_norms):
      x = addCoordinates(x)
      x = self.drop(self.act(b(l(x))))

    x = x.view(batch, -1)
    x = self.denses[0](x)
    x = self.act(x)
    x = self.drop(x)
    x = self.denses[1](x)
    return x

class FCNet(nn.Module):
    def __init__(self, in_dim = 32, in_chans = None, layer_dims = [128, 64, 4], activation = "leaky_relu", final_activation = "none", drop_p = 0.2):
        super().__init__()
        self.layer_dims = [in_dim] + layer_dims
        self.denses = nn.ModuleList([nn.Linear(in_features = self.layer_dims[i], out_features = self.layer_dims[i+1]) for i in range(len(layer_dims))])
        self.bns = nn.ModuleList([nn.LayerNorm(self.layer_dims[i + 1]) for i in range(len(layer_dims) - 1)])
        self.act = activation_func(activation)
        self.f_act = activation_func(final_activation)
        self.drop = nn.Dropout(p=drop_p)

    def forward(self, x):
        for i, l in enumerate(self.denses):
            x = l(x)
            if not i == len(self.denses) - 1:
                x = self.bns[i](x)
                x = self.act(x)
            if not i == len(self.denses) - 1:
                x = self.drop(x)
        return self.f_act(x)

nets = {"cnn" : CNNet, "fcn" : FCNet}
