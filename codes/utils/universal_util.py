import os
import math
import torch
import pickle
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageFont, ImageDraw
from ruamel import yaml

from utils.ssim import SSIM


def save_yaml(opt, yaml_path):
    f = open(yaml_path, 'w', encoding='utf-8')
    yaml.dump(opt, f, Dumper=yaml.RoundTripDumper)
    f.close()


def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    data = yaml.load(f.read(), Loader=yaml.Loader)
    f.close()
    return data


def add_poisson_gaussian_noise(img, level=1000.0):
    if torch.max(img) == 0.0:
        poisson = torch.poisson(torch.zeros(*img.shape)).to(img.device)
    else:
        poisson = torch.poisson(img / torch.max(img) * level).to(img.device)
    gaussian = torch.normal(mean=torch.ones(*img.shape) * 100.0, std=torch.ones(*img.shape) * 4.5).to(img.device)
    img_noised = poisson + gaussian
    assert torch.max(img_noised) - torch.min(img_noised) != 0.0
    img_noised = (img_noised - torch.min(img_noised)) / (torch.max(img_noised) - torch.min(img_noised))
    if torch.max(img) != 0.0:
        img_noised = img_noised * (torch.max(img) - torch.min(img)) + torch.min(img)
    else:
        # raise RuntimeWarning('occur purely dark img')
        print('occur purely dark img')
    return img_noised


def random_rotate_crop_flip(img, new_size, fill):
    transformer = transforms.Compose([
        # bicubic may result in negative value of some pixels
        #TODO change

        transforms.RandomRotation(degrees=(-180, +180), interpolation=transforms.InterpolationMode.BILINEAR, fill=fill),
        transforms.RandomCrop(size=new_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    return transformer(img)


def get_the_latest_file(root, postfix='_net.pth'):
    """
    target files should be named as '1234567_net.pth'
    """
    all_files = [file for file in os.listdir(root) if (file.endswith(postfix) and file.replace(postfix, '').isdigit())]
    assert len(all_files) > 0, 'empty directory'
    latest_file = all_files[0]
    get_step = lambda x: int(x.replace(postfix, ''))
    for certain_file in all_files:
        certain_step = get_step(certain_file)
        if certain_step > get_step(latest_file):
            latest_file = certain_file
    return {'path': os.path.join(root, latest_file),
            'step': get_step(latest_file)}


def remove_excess(root, keep, postfix='_model.pth'):
    all_files = [file for file in os.listdir(root) if file.endswith(postfix)]
    num_removed = 0
    for file in all_files:
        is_redundant = True
        for s in keep:
            if s in file:
                is_redundant = False
        if is_redundant:
            os.remove(os.path.join(root, file))
            num_removed += 1
    return num_removed


def calculate_PSNR(img1, img2, border=0, max_val=None):
    """
    input image shape should be torch.Tensor(..., H, W)
    border mean how many pixels of the edge will be abandoned. default: 0
    """
    assert len(img1.shape) >= 2 and len(img2.shape) >= 2, 'Input images must be in the shape of (..., H, W).'
    assert img1.shape == img2.shape, f'input images should have the same dimensions, ' \
                                     f'but got {img1.shape} and {img2.shape}'
    if max_val == 'auto':
        max_val = max(torch.max(img1).item(), torch.max(img2).item())
    elif max_val is None:
        raise ValueError('unspecified max_val')
    H, W = img1.shape[-2:]
    img1 = img1[..., border:H - border, border:W - border].type(torch.float32)
    img2 = img2[..., border:H - border, border:W - border].type(torch.float32)
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse <= 0.0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def normalization(tensor, v_min=0.0, v_max=1.0, batch=False):
    """
    :param tensor:
    :param v_min:
    :param v_max:
    :param batch: if False, regard all dims as data needed normalization
                  if True, regard the 1st dim as batch, the last 2 dims as data needed normalization
    :return:
    """
    if not batch:
        if torch.max(tensor) - torch.min(tensor) == 0.0:
            return torch.clamp(tensor, max=v_max, min=v_min)
        return ((tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))) * (v_max - v_min) + v_min
    else:
        assert len(tensor.shape) >= 2
        b_max = torch.max(torch.max(tensor, dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        b_min = torch.min(torch.min(tensor, dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
        e = torch.argwhere((b_max - b_min).squeeze(-1).squeeze(-1) == 0.0)
        if len(e) > 0:
            t = tensor.clone()
            p = torch.clamp(tensor[e, ...], max=v_max, min=v_min)
            t[e, ...] = torch.rand(size=t[0, ...].shape, device=t.device)
            b_max = torch.max(torch.max(t, dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
            b_min = torch.min(torch.min(t, dim=-1, keepdim=True).values, dim=-2, keepdim=True).values
            t = ((t - b_min) / (b_max - b_min)) * (v_max - v_min) + v_min
            t[e, ...] = p
        else:
            t = ((tensor - b_min) / (b_max - b_min)) * (v_max - v_min) + v_min
        return t


def get_phaseZ(opt=None, batch_size=1, device=torch.device('cpu')):
    """
    opt: default = {'idx_start': 4, 'num_idx': 11, 'mode': 'gaussian', 'std': 0.125, 'bound': 1.0}
    """
    if opt is None:
        opt = {'idx_start': [4], 'num_idx': [15], 'mode': 'gaussian', 'std': 0.125, 'bound': 1.0}
    phaseZ = torch.zeros(size=(batch_size, 25))
    if opt['mode'] == 'gaussian':
        for i, n, b in zip(opt['idx_start'], opt['num_idx'], opt['bound']):
            phaseZ[:, i:i + n] = torch.normal(mean=0.0, std=opt['std'],size=(batch_size, n))
            phaseZ = torch.clamp(phaseZ, min=-b, max=b)
    elif opt['mode'] == 'uniform':
        for i, n, b in zip(opt['idx_start'], opt['num_idx'], opt['bound']):
            phaseZ[:, i:i + n] = torch.rand(size=(batch_size, n)) * 2.0 * b - b
    else:
        raise NotImplementedError
    return phaseZ.to(device)


def pickle_dump(obj, file_path):
    with open(file_path, "xb") as f:
        pickle.dump(obj, f)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def rectangular_closure(x):
    """返回矩形闭包的上下左右边界"""
    assert x.dtype == torch.bool
    assert len(x.shape) == 2
    a_h = torch.argwhere(torch.sum(x, dim=1) >= 1)
    a_w = torch.argwhere(torch.sum(x, dim=0) >= 1)
    return a_h[0].item(), a_h[-1].item(), a_w[0].item(), a_w[-1].item()


def scale_and_translation(x, v_max, v_min):
    x = normalization(x)
    return x * (v_max - v_min) + v_min


def complex_to_reals(x, mode='ri'):
    """
    element of x should be complex number
    mode = 'ri' | 'ap'
    """
    if mode == 'ri':
        return x.real, x.imag
    elif mode == 'ap':
        a = torch.angle(x)
        # a = a.detach().cpu().numpy()
        # a = np.unwrap(np.unwrap(a, axis=-1), axis=-2)
        # a = torch.from_numpy(a).to(x.device)
        return torch.abs(x), a
    else:
        raise NotImplementedError


def one_plus_log(x, base='e'):
    if base == 'e':
        return torch.sign(x) * torch.log(1.0 + torch.abs(x))
    elif base == '10':
        return torch.sign(x) * torch.log10(1.0 + torch.abs(x))
    else:
        raise NotImplementedError


def save_gray_img(x, path, norm=True):
    assert len(x.shape) == 2
    if norm:
        x = normalization(x)
    x = (x * 65535.0).to(torch.int32)
    transforms.ToPILImage()(x).save(path)


class PCA_Encoder(object):
    def __init__(self, weight, mean):
        self.weight = weight  # l**2 x h
        self.mean = mean  # 1 x l**2
        self.size = self.weight.size()

    def __call__(self, batch_kernel):
        """
        :param batch_kernel: shape (B, l, l)
        :return: shape (B, h)
        """
        B, H, W = batch_kernel.size()  # (B, l, l)
        return torch.bmm(batch_kernel.view((B, 1, H * W)) - self.mean,
                         self.weight.expand((B, ) + self.size)).view((B, -1))


class PCA_Decoder(object):
    def __init__(self, weight, mean):
        self.weight = weight  # l**2 x h
        self.mean = mean  # 1 x l**2
        self.l = int(self.weight.shape[0] ** 0.5)
        self.size = weight.T.size()
        assert self.l * self.l == self.weight.shape[0]

    def __call__(self, batch_kernel_code):
        """
        :param batch_kernel_code: shape (B, h)
        :return: shape (B, l, l)
        """
        B, h = batch_kernel_code.shape
        return (torch.bmm(batch_kernel_code.view((B, 1, h)), self.weight.T.expand((B, ) + self.size)) + self.mean).view(
            (B, self.l, self.l))


def nearest_itpl(x, size, norm=False):
    """nearest interpolation"""
    assert len(size) == 2
    if len(x.shape) == 4:
        y = F.interpolate(x, size, mode='nearest')
    elif len(x.shape) == 3:
        y = F.interpolate(x.unsqueeze(0), size, mode='nearest').squeeze(0)
    elif len(x.shape) == 2:
        y = F.interpolate(x.unsqueeze(0).unsqueeze(0), size, mode='nearest').squeeze(0).squeeze(0)
    else:
        raise ValueError
    if norm:
        y = normalization(y)
    return y


def overlap(x, y, pos):
    """put x on y, and x[..., 0, 0] at pos"""
    y = y.clone().detach()
    y[..., pos[-2]:pos[-2] + x.shape[-2], pos[-1]:pos[-1] + x.shape[-1]] = x
    return y


def draw_text_on_image(img, text: str, pos: tuple, font_size: int, color):
    """
    :param img: PIL Image
    :param text:
    :param pos: (width, height)
    :param font_size:
    :param color: int or tuple[int, ...]
    :return: PIL Image
    """
    draw = ImageDraw.Draw(img)
    if os.name == 'posix':  # Ubuntu
        font = ImageFont.truetype('./abyssinica/AbyssinicaSIL-Regular.ttf', size=font_size)
    elif os.name == 'nt':  # Windows
        font = ImageFont.truetype(r'C:\Windows\Fonts\times.ttf', size=font_size) #todo change path
    else:
        raise Exception
    draw.text(pos, text, font=font, fill=color)


def concat(tensors, row, col):
    assert row * col == len(tensors)
    return torch.cat([torch.cat([tensors[i * col + j] for j in range(col)], dim=-1) for i in range(row)], dim=-2)


def calculate_SSIM(img1, img2, rescale=False):
    """
    img1, img2: tensor(1, C, H, W)/(C, H, W)/(H, W), [0, 1]
    rescale: if True, rescale images such that max=1.0
    """
    def expand4d(x):
        if len(x.shape) == 2:
            return x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            return x.unsqueeze(0)
        elif len(x.shape) == 4:
            return x
        else:
            raise ValueError
    img1, img2 = expand4d(img1), expand4d(img2)
    if rescale:
        v = max(torch.max(img1).item(), torch.max(img2).item())
        img1, img2 = img1 / v, img2 / v
    ssim_loss = SSIM(window_size=11)
    return ssim_loss(img1, img2).item()


def scan_pos(H, W, h, w, s=2):
    """
    scan big picture (H, W) with small picture (h, w)
    :param H:
    :param W:
    :param h:
    :param w:
    :param s:
    :return:
    """
    assert H >= h and W >= w
    assert h % s == 0 and w % s == 0
    y, x = 0, 0
    ys, xs = [], []
    positions = []
    while True:
        ys.append(y)
        if y + h >= H:
            ys[-1] = H - h
            break
        y += (h // s)
    while True:
        xs.append(x)
        if x + w >= W:
            xs[-1] = W - w
            break
        x += (w // s)
    for y in ys:
        for x in xs:
            positions.append((y, x))
    return positions


def main():
    opt = read_yaml('../../options/trains/train_UNetBased.yaml')
    print(opt)
    pass


if __name__ == '__main__':
    main()
