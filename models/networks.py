import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torchvision import transforms
import random


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = LayerNorm()
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='layer', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_1blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=1)
    elif netG == 'convnext':
        net = ConvnextGenerator(input_nc, output_nc, ngf)
    elif netG == 'invnext':
        net = InvnextGenerator(input_nc, output_nc, ngf)
    elif netG == 'Convnext_Interpolation':
        net = ConvnextBIGenerator(input_nc, output_nc, ngf)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='layer', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'binary':
        net = Discriminator_Classify(input_nc)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """

        super(GANLoss, self).__init__()
        target_real_label = random.randint(7, 12) * 0.1
        target_fake_label = random.randint(0, 3) * 0.1
        target_A_label = torch.tensor([1, 1, 1, 1])
        target_B_label = torch.tensor([0, 0, 0, 0])

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.register_buffer('A_label', torch.tensor(target_A_label))
        self.register_buffer('B_label', torch.tensor(target_B_label))

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'binary':
            self.loss = nn.CrossEntropyLoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)


    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def get_target_tensorAB(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.A_label
        else:
            target_tensor = self.B_label
        return target_tensor
        #return target_tensor.expand_as(prediction)


    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)

        elif self.gan_mode == 'binary':
            target_tensor = self.get_target_tensorAB(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=96, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),#通道数为96,矩阵大小不变
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers,这里i=0,1
            mult = 2 ** i#**时乘方
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),#kernel=3,str=2,pad=1时,输出为(n+2-3)/2+1,0.5的话退位,比如3.5=3
                      nn.ReLU(True)] #+=：代表追加操作，即in-place操作，在原地把另一个对象的内容追加到对象中。

        mult = 2 ** n_downsampling#i=0,1,2,3,4,5
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),#nn.ConvTranspose2d反卷积
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      #output_padding的值默认为stride-1,想要还原单数输出,欲要output_padding=0
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # groups=dim意为depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers.equal to MLP
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # nn.Identity()恒等映射

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x  # 缩放向量,给不同channel乘上不同的数值（可训练）来帮助网络更快更精确的收敛。
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)  # groups=dim意为depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.conv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers.equal to MLP
        self.act = nn.GELU()
        self.conv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # nn.Identity()恒等映射

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

from involution import Involution2d
class InvBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = Involution2d(in_channels=dim, out_channels=dim)  # groups=dim意为depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers.equal to MLP
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # nn.Identity()恒等映射

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x  # 缩放向量,给不同channel乘上不同的数值（可训练）来帮助网络更快更精确的收敛。
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


from torch.nn import Tanh
class ConvnextGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=96, drop_path_rate=0.,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], layer_scale_init_value=1e-6):
        super(ConvnextGenerator, self).__init__()

        features = ngf

        self.Convnextblock = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                     sum(depths))]  # torch.linspace表示线性划分,从0到drop_path_rate,划分个数为sum(depths)
        # 作用是随着深度的增加,dropout的概率会变大
        cur = 0  # cur用来记录深度
        for i in range(4):  # range(4)表示4个stage
            Convnextblock = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],  # 加个*表示列表可以展开,即在sequential中叠加block
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                # range(depths[i])表示每个stage中的深度,即block数
            )
            self.Convnextblock.append(Convnextblock)
            cur += depths[i]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_nc, features, kernel_size=4, stride=4),
            LayerNorm(features, eps=1e-6, data_format="channels_first")
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=2, stride=2),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first")
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=2, stride=2),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first")
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=2, stride=2),
            LayerNorm(features * 8, eps=1e-6, data_format="channels_first")
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first"),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2),
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(features, output_nc, kernel_size=4, stride=4),
            LayerNorm(output_nc, eps=1e-6, data_format="channels_first"),
        )

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first"),
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )
        self.act = Tanh()

    def forward(self, x):
        en1 = self.encoder1(x)
        enx1 = self.Convnextblock[0](en1)
        en2 = self.encoder2(enx1)
        enx2 = self.Convnextblock[1](en2)
        en3 = self.encoder3(enx2)
        enx3 = self.Convnextblock[2](en3)
        en4 = self.encoder4(enx3)
        bottle_x = self.Convnextblock[3](en4)

        de4 = self.decoder4(bottle_x)
        dex4 = torch.cat((de4, enx3), dim=1)
        dex4 = self.upconv4(dex4)
        de3 = self.decoder3(dex4)
        dex3 = torch.cat((de3, enx2), dim=1)
        dex3 = self.upconv3(dex3)
        de2 = self.decoder2(dex3)
        dex2 = torch.cat((de2, enx1), dim=1)
        dex2 = self.upconv2(dex2)
        de1 = self.decoder1(dex2)
        return self.act(de1)
class ConvnextBIGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=96, drop_path_rate=0.,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], layer_scale_init_value=1e-6):
        super(ConvnextBIGenerator, self).__init__()

        features = ngf

        self.Convnextblock = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                     sum(depths))]  # torch.linspace表示线性划分,从0到drop_path_rate,划分个数为sum(depths)
        # 作用是随着深度的增加,dropout的概率会变大
        cur = 0  # cur用来记录深度
        for i in range(4):  # range(4)表示4个stage
            Convnextblock = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],  # 加个*表示列表可以展开,即在sequential中叠加block
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                # range(depths[i])表示每个stage中的深度,即block数
            )
            self.Convnextblock.append(Convnextblock)
            cur += depths[i]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_nc, features, kernel_size=4, stride=4),
            LayerNorm(features, eps=1e-6, data_format="channels_first")
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=2, stride=2),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first")
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=2, stride=2),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first")
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=2, stride=2),
            LayerNorm(features * 8, eps=1e-6, data_format="channels_first")
        )
        # step 1
        self.decoder4 = nn.Sequential(
            #transforms.Resize(size=(14, 14), interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.Resize(size=(14, 14), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(size=(14, 14), interpolation=transforms.InterpolationMode.BICUBIC),
            nn.Conv2d(features * 8, features * 4, kernel_size=1),
            #nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )
        #step 2
        self.decoder3 = nn.Sequential(
            #transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.BICUBIC),
            nn.Conv2d(features * 4, features * 2, kernel_size=1),
            #nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first"),
        )
        # step 3
        self.decoder2 = nn.Sequential(
            #transforms.Resize(size=(56, 56), interpolation=transforms.InterpolationMode.BICUBIC),
            #nn.Conv2d(features * 2, features, kernel_size=1),
            nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2),#反卷积
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )

        self.decoder1 = nn.Sequential(
            #transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            #nn.Conv2d(features, output_nc, kernel_size=1),
            nn.ConvTranspose2d(features, output_nc, kernel_size=4, stride=4),
            LayerNorm(output_nc, eps=1e-6, data_format="channels_first"),
        )
        self.decoder0 = nn.Sequential(
            transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            nn.ConvTranspose2d(output_nc, output_nc, kernel_size=1),
            LayerNorm(output_nc, eps=1e-6, data_format="channels_first"),
        )
        self.upconv4 = nn.Sequential(
            nn.Conv2d(features * 8, features * 4,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )
        self.upconv3 = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first"),
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )
        self.upconv1 = nn.Sequential(
            nn.Conv2d(features * 8, features * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )
        self.act = Tanh()

    def forward(self, x):
        en1 = self.encoder1(x)
        enx1 = self.Convnextblock[0](en1)
        en2 = self.encoder2(enx1)
        enx2 = self.Convnextblock[1](en2)
        en3 = self.encoder3(enx2)
        enx3 = self.Convnextblock[2](en3)
        en4 = self.encoder4(enx3)
        bottle_x = self.Convnextblock[3](en4)

        de4 = self.decoder4(bottle_x)
        dex4 = torch.cat((de4, enx3), dim=1)
        dex4 = self.upconv4(dex4)
        de3 = self.decoder3(dex4)
        dex3 = torch.cat((de3, enx2), dim=1)
        dex3 = self.upconv3(dex3)
        de2 = self.decoder2(dex3)
        dex2 = torch.cat((de2, enx1), dim=1)
        dex2 = self.upconv2(dex2)
        de1 = self.decoder1(dex2)
        return self.act(de1)


class InvnextGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=96, drop_path_rate=0.,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], layer_scale_init_value=1e-6):
        super(InvnextGenerator, self).__init__()

        features = ngf

        self.InvnextBlock = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                     sum(depths))]  # torch.linspace表示线性划分,从0到drop_path_rate,划分个数为sum(depths)
        # 作用是随着深度的增加,dropout的概率会变大
        cur = 0  # cur用来记录深度
        for i in range(4):  # range(4)表示4个stage
            InvnextBlock = nn.Sequential(
                *[InvBlock(dim=dims[i], drop_path=dp_rates[cur + j],  # 加个*表示列表可以展开,即在sequential中叠加block
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                # range(depths[i])表示每个stage中的深度,即block数
            )
            self.InvnextBlock.append(InvnextBlock)
            cur += depths[i]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=features, kernel_size=4, stride=4),
            LayerNorm(features, eps=1e-6, data_format="channels_first")
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=2, stride=2),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first")
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=2, stride=2),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first")
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=features * 4, out_channels=features * 8, kernel_size=2, stride=2),
            LayerNorm(features * 8, eps=1e-6, data_format="channels_first")
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first"),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2),
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(features, output_nc, kernel_size=4, stride=4),
            LayerNorm(output_nc, eps=1e-6, data_format="channels_first"),
        )

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features * 2, eps=1e-6, data_format="channels_first"),
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False),
            LayerNorm(features * 4, eps=1e-6, data_format="channels_first"),
        )
        self.act = Tanh()

    def forward(self, x):
        en1 = self.encoder1(x)
        enx1 = self.InvnextBlock[0](en1)
        en2 = self.encoder2(enx1)
        enx2 = self.InvnextBlock[1](en2)
        en3 = self.encoder3(enx2)
        enx3 = self.InvnextBlock[2](en3)
        en4 = self.encoder4(enx3)
        bottle_x = self.InvnextBlock[3](en4)

        de4 = self.decoder4(bottle_x)
        dex4 = torch.cat((de4, enx3), dim=1)
        dex4 = self.upconv4(dex4)
        de3 = self.decoder3(dex4)
        dex3 = torch.cat((de3, enx2), dim=1)
        dex3 = self.upconv3(dex3)
        de2 = self.decoder2(dex3)
        dex2 = torch.cat((de2, enx1), dim=1)
        dex2 = self.upconv2(dex2)
        de1 = self.decoder1(dex2)
        return self.act(de1)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=96, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=96, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8) #**幂运算
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Discriminator_Classify(nn.Module):
    def __init__(self, input_nc, ndf=96):
        super(Discriminator_Classify, self).__init__()

        self.conv = nn.Sequential(
            # input is (input_nc) x 224 x 224
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, True),
            # state size. (ndf) x 112 x 112
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # state size. (ndf*2) x 56 x 56
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # state size. (ndf*4) x 28 x 28
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # state size. (ndf*8) x 14 x 14
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, True),
            # state size. (ndf*16) x 7 x 7
            nn.Conv2d(ndf * 16, ndf * 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, True),
            # state size. (ndf*32) x 4 x 4
        )

        self.classify = nn.Sequential(
            nn.Linear(ndf * 32 * 16, 2),
        )

    def forward(self, input):
        conv_out = self.conv(input)
        conv_out = conv_out.view(4, -1)
        classify_out = self.classify(conv_out)
        return classify_out

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=96, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
