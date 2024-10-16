import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class Subplots:
    def __init__(self, figsize = (40, 5)):
        self.fig = plt.figure(figsize=figsize)
        
    def plot_img_list(self, img_list, savedir='figs/test', 
                    nrows = 1, rownum = 0, 
                    hold = False, coltitles=[], rowtitle=''):
        
        for i, img in enumerate(img_list):
            try:
                npimg = img.clone().detach().cpu().numpy()
            except:
                npimg = img
            tpimg = np.transpose(npimg, (1, 2, 0))
            lenrow = int((len(img_list)))
            ax = self.fig.add_subplot(nrows, lenrow, i+1+(rownum*lenrow))
            if len(coltitles) > i:
                ax.set_title(coltitles[i])
            if i == 0:
                ax.annotate(rowtitle, xy=((-0.06 * len(rowtitle)), 0.4),# xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
                # ax.set_ylabel(rowtitle, rotation=90)
            ax.imshow(tpimg)
            ax.axis('off')

        if not hold:
            self.fig.tight_layout()
            plt.savefig(f'{savedir}.png')
            plt.clf()
            plt.close('all')
            
                    
def VisualizeNumpyImageGrayscale(image_3d):
    r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
    """
    vmin = np.min(image_3d)
    image_2d = image_3d - vmin
    vmax = np.max(image_2d)
    return (image_2d / vmax)

def normalize_numpy(image_3d):
    r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
    """
    vmin = np.min(image_3d)
    image_2d = image_3d - vmin
    vmax = np.max(image_2d)
    return (image_2d / vmax)

# def normalize_tensor(image_3d): 
#     r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
#     """
#     vmin = torch.min(image_3d)
#     image_2d = image_3d - vmin
#     vmax = torch.max(image_2d)
#     return (image_2d / vmax)

def normalize_tensor(image_3d): 
    r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
    """
    image_2d = (image_3d - torch.min(image_3d))
    return (image_2d / torch.max(image_2d))

def format_img(img_):
    np_img = img_.numpy()
    tp_img = np.transpose(np_img, (1, 2, 0))
    return tp_img

def imshow(img, save_path=None):
    try:
        npimg = img.clone().detach().cpu().numpy()
    except:
        npimg = img
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg)
    # plt.axis('off')
    plt.tight_layout()
    if save_path != None:
        plt.savefig(str(str(save_path) + ".png"))
    #plt.show()a

def imshow_img(img, imsave_path):
    # works for tensors and numpy arrays
    try:
        npimg = VisualizeNumpyImageGrayscale(img.numpy())
    except:
        npimg = VisualizeNumpyImageGrayscale(img)
    npimg = np.transpose(npimg, (2, 0, 1))
    imshow(npimg, save_path=imsave_path)
    print("Saving image as ", imsave_path)
    
def returnGrad(img, labels, model, compute_loss, loss_metric, augment=None, device = 'cpu'):
    model.train()
    model.to(device)
    img = img.to(device)
    img.requires_grad_(True)
    labels.to(device).requires_grad_(True)
    model.requires_grad_(True)
    cuda = device.type != 'cpu'
    scaler = amp.GradScaler(enabled=cuda)
    pred = model(img)
    # out, train_out = model(img, augment=augment)  # inference and training outputs
    loss, loss_items = compute_loss(pred, labels, metric=loss_metric)#[1][:3]  # box, obj, cls
    # loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]).to(device))
    # loss = torch.sum(loss).requires_grad_(True)
    
    with torch.autograd.set_detect_anomaly(True):
        scaler.scale(loss).backward(inputs=img)
    # loss.backward()
    
#    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
    model.eval()
    Sc_dx = torch.tensor(Sc_dx, dtype=torch.float32)
    return Sc_dx

def calculate_snr(img, attr, dB=True):
    try:
        img_np = img.detach().cpu().numpy()
        attr_np = attr.detach().cpu().numpy()
    except:
        img_np = img
        attr_np = attr
    
    # Calculate the signal power
    signal_power = np.mean(img_np**2)

    # Calculate the noise power
    noise_power = np.mean(attr_np**2)

    if dB == True:
        # Calculate SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        # Calculate SNR
        snr = signal_power / noise_power

    return snr

def overlay_mask(img, mask, colormap: str = "jet", alpha: float = 0.7):
    
    cmap = plt.get_cmap(colormap)
    npmask = np.array(mask.clone().detach().cpu().squeeze(0))
    # cmpmask = ((255 * cmap(npmask)[:, :, :3]).astype(np.uint8)).transpose((2, 0, 1))
    cmpmask = (cmap(npmask)[:, :, :3]).transpose((2, 0, 1))
    overlayed_imgnp = ((alpha * (np.asarray(img.clone().detach().cpu())) + (1 - alpha) * cmpmask))
    overlayed_tensor = torch.tensor(overlayed_imgnp, device=img.device)
    
    return overlayed_tensor