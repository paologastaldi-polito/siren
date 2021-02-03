import scipy
import numpy as np
import math
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale, ToPILImage # , GaussianBlur
import matplotlib.pyplot as plt
import skimage
from skimage import metrics

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    # grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]   # original code
    grad = torch.autograd.grad(
        y,                              # predicted image => a continuous function
        [x],                            # ground truth => sample a continuous function
        grad_outputs=grad_outputs,      #
        retain_graph=True,              # keep the graph after its usage
        create_graph=True               # graph for higher order derivative products
    )[0]

    return grad

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        # div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True, allow_unused=True)[0][..., i:i+1] # original code
        div += torch.autograd.grad(
            y[..., i],                  # predicted image
            x,                          # ground truth
            torch.ones_like(y[..., i]), # 
            retain_graph=True,          # keep the graph after its usage
            create_graph=True           # graph for higher order derivative products
        )[0][..., i:i+1]
    return div

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def _psnr(pred, gt, max_pixel=1.):
    '''peak signal to noise ratio formula'''
    mse = skimage.metrics.mean_squared_error(pred, gt)
    # mse = ((pred - gt)**2).mean().item()
    # rmse = math.sqrt(mse)
    psnr = float('inf')
    if mse != 0.:
    #     psnr = 20 * math.log10(max_pixel/rmse)
        # psnr = skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min())
        psnr = skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=max_pixel)
    return psnr

def _ssim(pred, gt,  max_pixel=1.):
    '''structural similarity formula'''
    # ssim_noise = skimage.metrics.structural_similarity(gt, pred, data_range=gt.max() - gt.min())
    ssim_noise = skimage.metrics.structural_similarity(gt, pred, data_range=max_pixel)
    return ssim_noise

def caption(pred, gt=None, type=None, sidelength=256):
    '''Generate a caption automatically'''
    label = 'PSNR: {:.2f}, SSIM: {:.2f}'
    if gt is not None:
        psnr = -999.
        ssim = -999.
        if type == 'img':
            pred = torch.from_numpy(_init_img_psnr(pred))
            pred = pred.cpu().view(sidelength, sidelength).detach().numpy()
            gt = torch.from_numpy(_init_img_psnr(gt))
            gt = gt.cpu().view(sidelength, sidelength).detach().numpy()
        elif type == 'grads':
            pred = torch.from_numpy(_init_grads_psnr(pred))
            pred_x = pred[..., 0]
            pred_y = pred[..., 1]
            pred = torch.sqrt(pred_x**2 + pred_y**2)
            pred = pred.cpu().view(sidelength, sidelength).detach().numpy()
            # pred = pred.cpu().norm(dim=-1).view(sidelength, sidelength).detach().numpy()
            # pred_x = pred_x.cpu().view(sidelength, sidelength).detach().numpy()
            # pred_y = pred_y.cpu().view(sidelength, sidelength).detach().numpy()
            gt = torch.from_numpy(_init_grads_psnr(gt))
            gt_x = gt[..., 0]
            gt_y = gt[..., 1]
            gt = torch.sqrt(gt_x**2 + gt_y**2)
            gt = gt.cpu().view(sidelength, sidelength).detach().numpy()
            # gt = gt.cpu().norm(dim=-1).view(sidelength, sidelength).detach().numpy()
            # gt_x = gt_x.cpu().view(sidelength, sidelength).detach().numpy()
            # gt_y = gt_y.cpu().view(sidelength, sidelength).detach().numpy()
        elif type == 'laplace':
            pred = torch.from_numpy(_init_laplace_psnr(pred))
            pred = pred.cpu().view(sidelength, sidelength).detach().numpy()
            gt = torch.from_numpy(_init_laplace_psnr(gt))
            gt = gt.cpu().view(sidelength, sidelength).detach().numpy()
        psnr = _psnr(pred, gt)
        ssim = _ssim(pred, gt)
        label = label.format(psnr, ssim)
    else:
        label = None
    return label

def plot_all(img, gt, sidelength=256, img_caption=None, save=False, fname='figure.png'):
    '''Plot image, gradients and laplacian all at the same time (only for the generated image)'''
    n_images = 3
    fig, axes = plt.subplots(1, n_images, figsize=(18,6))
    for a in axes:
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])
    if img_caption is None:
        img_caption = {
            'img' : caption(img['img'], gt['img'], 'img', sidelength=sidelength),
            'grads' : caption(img['grads'], gt['grads'], 'grads', sidelength=sidelength),
            'laplace' : caption(img['laplace'], gt['laplace'], 'laplace', sidelength=sidelength)
        }
    if save:
        color='k'
    else:
        color='w'
    axes[0].imshow(img['img'].cpu().view(sidelength, sidelength).detach().numpy())
    axes[0].set_xlabel(img_caption['img'], color=color)
    axes[1].imshow(img['grads'].cpu().norm(dim=-1).view(sidelength, sidelength).detach().numpy())
    axes[1].set_xlabel(img_caption['grads'], color=color)
    axes[2].imshow(img['laplace'].cpu().view(sidelength, sidelength).detach().numpy())
    axes[2].set_xlabel(img_caption['laplace'], color=color)
    plt.axis('off')
    if save:
        plt.savefig(fname=fname, format='png')
    plt.show()

# --- IMAGE ---

def monocromatic_img(color=255, sidelength=256):
    '''Generate a squared monocromatic image'''
    img = np.zeros([sidelength,sidelength,1],dtype=np.uint8)
    img[:] = color
    transform = Compose([
        ToPILImage(),
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

# def blur_img(img):
#     '''Hard blur an image (with presets)'''
#     return GaussianBlur(99., (0.1, 99.))(img)

def _init_img_psnr(in_img, sidelength=256, silent=True):
    if torch.is_tensor(in_img):
        out_img = in_img.cpu().detach().numpy() # tensors do not have to be attached to the graph and running on the GPU anymore
    else:
        out_img = in_img
    out_img = out_img.transpose(1, 2, 0) # only for the image
    # why 1. and 2.?
    # original image tensor: [-1., +1.]
    # output: [0., +1.]
    out_img = (out_img + 1.) / 2. # move [-1, 1] in [0, 1]
    if not silent and (np.min(out_img) < 0. or np.max(out_img) > 1.):
        print('WARNING: clipping the image tensor, min %.2f max %.2f' % (np.min(out_img), np.max(out_img)))
    out_img = np.clip(out_img, a_min=0., a_max=1.)
    return out_img

def img_psnr(img, gt, silent=True):
    '''Compute PSNR over image'''
    if type(img) is dict:
        img = img['img']
    if type(gt) is dict:
        gt = gt['img']
    img = _init_img_psnr(img, silent=silent)
    gt = _init_img_psnr(gt, silent=silent)
    return _psnr(img, gt)

def plot_img(img, gt=None, sidelength=256, img_caption=None, silent=True, save=False, fname='figure.png'):
    img = torch.from_numpy(_init_img_psnr(img))
    img = img.cpu().view(sidelength, sidelength).detach().numpy()
    gt = torch.from_numpy(_init_img_psnr(gt))
    gt = gt.cpu().view(sidelength, sidelength).detach().numpy()

    n_images = 1
    _, axes = plt.subplots(1,n_images, figsize=(18,6))
    if img_caption is None:
        img_caption = caption(img, gt, 'img', sidelength=sidelength, silent=silent)
    if save:
        color='k'
    else:
        color='w'
    axes.imshow(img.cpu().view(sidelength, sidelength).detach().numpy())
    axes.set_xlabel(img_caption, color=color)
    if save:
        plt.savefig(fname=fname, format='png')
    plt.show()

# --- GRADIENTS ---

def merge_grads(grads_x, grads_y):
    if torch.is_tensor(grads_x):
        grads = torch.sqrt(torch.pow(grads_x, 2) + torch.pow(grads_y, 2))
    else:
        grads = np.sqrt(np.power(grads_x, 2) + np.power(grads_y, 2))
    return grads

def sobel_filter(img, scale_fact=1.):
    '''Generate gradients with the sobel operator'''
    img = img.cpu().detach()
    img *= scale_fact
    grads_x = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
    grads_y = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]   
    grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
    grads = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)
    # grads = merge_grads(grads_x, grads_y) 
    # return { 'grads' : torch.from_numpy(grads), 'grads_x' : torch.from_numpy(grads_x), 'grads_y' : torch.from_numpy(grads_y) }
    return grads

def _init_grads_psnr(in_grads, sidelength=256, silent=True):
    if torch.is_tensor(in_grads):
        out_grads = in_grads.cpu().detach().numpy() # tensors do not have to be attached to the graph and running on the GPU anymore
    else:
        out_grads = in_grads
    # why 16. and 32.?
    # original image tensor: [-1., +1.]
    # gradient matrix: [-32., +32.]
    # output: [0., +1.]
    out_grads = (out_grads + 16.) / 32.
    if not silent and (np.min(out_grads) < 0. or np.max(out_grads) > 1.):
        print('WARNING: clipping the gradients tensor, min %.2f max %.2f' % (np.min(out_grads), np.max(out_grads)))
    out_grads = np.clip(out_grads, a_min=0., a_max=1.)
    return out_grads

def grads_psnr(img_grads, gt_grads, silent=True):
    '''Compute PSNR over gradients'''
    if type(img_grads) is dict:
        img_grads = img_grads['grads']
    if type(gt_grads) is dict:
        gt_grads = gt_grads['grads']
    # if len(img_grads.shape) != 2:
    #     img_grads = merge_grads(img_grads[..., 0].unsqueeze(-1), img_grads[..., 1].unsqueeze(-1))
    # if len(gt_grads.shape) != 2:
    #     gt_grads = merge_grads(gt_grads[..., 0].unsqueeze(-1), gt_grads[..., 1].unsqueeze(-1))
    img_grads = _init_grads_psnr(img_grads, silent=silent)
    gt_grads = _init_grads_psnr(gt_grads, silent=silent)
    return _psnr(img_grads, gt_grads)

def plot_grads(img_grads, gt_grads=None, sidelength=256, img_caption=None, silent=True, save=False, fname='figure.png'):
    img_grads = torch.from_numpy(_init_grads_psnr(img_grads))
    img_grads_x = img_grads[..., 0]
    img_grads_y = img_grads[..., 1]
    img_grads = torch.sqrt(img_grads_x**2 + img_grads_y**2)
    img_grads = img_grads.cpu().view(sidelength, sidelength).detach().numpy()
    # pred = pred.cpu().norm(dim=-1).view(sidelength, sidelength).detach().numpy()
    # pred_x = pred_x.cpu().view(sidelength, sidelength).detach().numpy()
    # pred_y = pred_y.cpu().view(sidelength, sidelength).detach().numpy()
    gt_grads = torch.from_numpy(_init_grads_psnr(gt_grads))
    gt_grads_x = gt_grads[..., 0]
    gt_grads_y = gt_grads[..., 1]
    gt_grads = torch.sqrt(gt_grads_x**2 + gt_grads_y**2)
    gt_grads = gt.cpu().view(sidelength, sidelength).detach().numpy()
    # gt = gt.cpu().norm(dim=-1).view(sidelength, sidelength).detach().numpy()
    # gt_x = gt_x.cpu().view(sidelength, sidelength).detach().numpy()
    # gt_y = gt_y.cpu().view(sidelength, sidelength).detach().numpy()

    n_images = 1
    # if gt_grads is not None:
    #     n_images += 1
    _, axes = plt.subplots(1,n_images, figsize=(18,6))
    if img_caption is None:
        img_caption = caption(img_grads, gt_grads, 'grads', sidelength=sidelength, silent=silent)
    if save:
        color='k'
    else:
        color='w'
    axes.imshow(img_grads.cpu().norm(dim=-1).view(sidelength, sidelength).detach().numpy())
    # axes.set_xlabel(img_caption, color=color)
    # if gt_grads is not None:
    #     axes[1].imshow(gt_grads.cpu().norm(dim=-1).view(sidelength, sidelength).detach().numpy())
    if save:
        plt.savefig(fname=fname, format='png')
    plt.show()

# --- LAPLACIAN ---

def laplace_filter(img, scale_fact=1.):
    '''Get laplacian with ND laplacian operator'''
    img = img.cpu().detach()
    img *= scale_fact
    img_laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
    img_laplace = torch.from_numpy(img_laplace).view(-1, 1)
    return img_laplace

def _init_laplace_psnr(in_laplace, sidelength=256, silent=True):
    if torch.is_tensor(in_laplace):
        out_laplace = in_laplace.cpu().detach().numpy() # tensors do not have to be attached to the graph and running on the GPU anymore
    else:
        out_laplace = in_laplace
    # why 8. and 16.?
    # original image tensor: [-1., +1.]
    # laplacian matrix: [-8., +8]
    out_laplace = (out_laplace + 8.) / 16.
    if not silent and (np.min(out_laplace) < 0. or np.max(out_laplace) > 1.):
        print('WARNING: clipping the laplacian tensor, min %.2f max %.2f' % (np.min(out_laplace), np.max(out_laplace)))
    out_laplace = np.clip(out_laplace, a_min=0., a_max=1.)
    return out_laplace

def laplace_psnr(img_laplace, gt_laplace, silent=True):
    '''Compute PSNR over laplacian'''
    if type(img_laplace) is dict:
        img_laplace = img_laplace['laplace']
    if type(gt_laplace) is dict:
        gt_laplace = gt_laplace['laplace']
    img_laplace = _init_laplace_psnr(img_laplace, silent=silent)
    gt_laplace = _init_laplace_psnr(gt_laplace, silent=silent)
    return _psnr(img_laplace, gt_laplace)

def plot_laplace(img_laplace, gt_laplace=None, sidelength=256, img_caption=None, silent=True, save=False, fname='figure.png'):
    img_laplace = torch.from_numpy(_init_laplace_psnr(img_laplace))
    img_laplace = img_laplace.cpu().view(sidelength, sidelength).detach().numpy()
    gt_laplace = torch.from_numpy(_init_laplace_psnr(gt_laplace))
    gt_laplace = gt_laplace.cpu().view(sidelength, sidelength).detach().numpy()

    n_images = 1
    _, axes = plt.subplots(1,n_images, figsize=(18,6))
    if img_caption is None:
        img_caption = caption(img_laplace, gt_laplace, 'laplace', sidelength=sidelength, silent=silent)
    if save:
        color='k'
    else:
        color='w'
    axes.imshow(img_laplace.cpu().view(sidelength, sidelength).detach().numpy())
    if save:
        plt.savefig(fname=fname, format='png')
    plt.show()

def image_mse(model_output, coords, gt_image):
    image_loss = ((model_output - gt_image)**2).mean()
    return image_loss

def gradients_mse(model_output, coords, gt_gradients):
    # compute gradients on the model
    gradients = gradient(model_output, coords)
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))
    return gradients_loss
 
def laplace_mse(model_output, coords, gt_laplace):
    laplacian = laplace(model_output, coords)
    # compare them with the ground truth
    laplace_loss = torch.mean((laplacian - gt_laplace)**2)
    return laplace_loss

def psnr(pred, gt, sidelength=256, silent=True):
    pred = torch.from_numpy(_init_img_psnr(pred, silent=silent))
    pred = pred.cpu().view(sidelength, sidelength).detach().numpy()
    gt = torch.from_numpy(_init_img_psnr(gt, silent=silent))
    gt = gt.cpu().view(sidelength, sidelength).detach().numpy()
    return _psnr(pred, gt)

def plot_psnrs(psnrs, total_steps, title, color='w', save=False, fname='figure.png'):
    '''psnrs: dict
    total_steps: int
    title: str
    color: matplotlib color type'''
    if save:
        color='k'
    x = [i for i in range(total_steps+1)]
    plt.figure(figsize=(12, 6))
    plt.title(title, color=color, fontsize=16)
    plt.xlabel('iterations', color=color, fontsize=14)
    plt.ylabel('PSNR', color=color, fontsize=14)
    plt.xticks(color=color)
    plt.yticks(color=color)
    plt.xlim(0, total_steps)
    plt.grid('--')
    for s, y in psnrs.items():
        if s == '30':
            plt.plot(x, y, label=s, linewidth=2.5)
        else:
            plt.plot(x, y, label=s)
    plt.legend(loc='upper left')
    if save:
        plt.savefig(fname=fname, format='png')
    plt.show()
