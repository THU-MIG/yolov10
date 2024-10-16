import torch
import numpy as np
# from plot_functs import * 
from .plot_functs import normalize_tensor, overlay_mask, imshow
import math   
import time
import matplotlib.path as mplPath
from matplotlib.path import Path
# from utils.general import non_max_suppression, xyxy2xywh, scale_coords
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh, non_max_suppression
from .metrics import bbox_iou
import torchvision.transforms as T

def plaus_loss_fn(grad, smask, pgt_coeff):
    ################## Compute the PGT Loss ##################
    # Positive regularization term for incentivizing pixels near the target to have high attribution
    dist_attr_pos = attr_reg(grad, (1.0 - smask)) # dist_reg = seg_mask
    # Negative regularization term for incentivizing pixels far from the target to have low attribution
    dist_attr_neg = attr_reg(grad, smask)
    # Calculate plausibility regularization term
    # dist_reg = dist_attr_pos - dist_attr_neg
    dist_reg = ((dist_attr_pos / torch.mean(grad)) - (dist_attr_neg / torch.mean(grad)))
    plaus_reg = (((1.0 + dist_reg) / 2.0))
    # Calculate plausibility loss
    plaus_loss = (1 - plaus_reg) * pgt_coeff
    return plaus_loss

def get_dist_reg(images, seg_mask):
    seg_mask = T.Resize((images.shape[2], images.shape[3]), antialias=True)(seg_mask).to(images.device)
    seg_mask = seg_mask.to(dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    seg_mask[seg_mask > 0] = 1.0
    
    smask = torch.zeros_like(seg_mask)
    sigmas = [20.0 + (i_sig * 20.0) for i_sig in range(8)]
    for k_it, sigma in enumerate(sigmas):
        # Apply Gaussian blur to the mask
        kernel_size = int(sigma + 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        seg_mask1 = T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma)(seg_mask)
        if torch.max(seg_mask1) > 1.0:
            seg_mask1 = (seg_mask1 - seg_mask1.min()) / (seg_mask1.max() - seg_mask1.min())
        smask = torch.max(smask, seg_mask1)
    return smask

def get_gradient(img, grad_wrt, norm=False, absolute=True, grayscale=False, keepmean=False):
    """
    Compute the gradient of an image with respect to a given tensor.

    Args:
        img (torch.Tensor): The input image tensor.
        grad_wrt (torch.Tensor): The tensor with respect to which the gradient is computed.
        norm (bool, optional): Whether to normalize the gradient. Defaults to True.
        absolute (bool, optional): Whether to take the absolute values of the gradients. Defaults to True.
        grayscale (bool, optional): Whether to convert the gradient to grayscale. Defaults to True.
        keepmean (bool, optional): Whether to keep the mean value of the attribution map. Defaults to False.

    Returns:
        torch.Tensor: The computed attribution map.

    """
    if (grad_wrt.shape != torch.Size([1])) and (grad_wrt.shape != torch.Size([])):
        grad_wrt_outputs = torch.ones_like(grad_wrt).clone().detach()#.requires_grad_(True)#.retains_grad_(True)
    else:
        grad_wrt_outputs = None
    attribution_map = torch.autograd.grad(grad_wrt, img, 
                                    grad_outputs=grad_wrt_outputs, 
                                    create_graph=True, # Create graph to allow for higher order derivatives but slows down computation significantly
                                    )[0]
    if absolute:
        attribution_map = torch.abs(attribution_map) # attribution_map ** 2 # Take absolute values of gradients
    if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
        attribution_map = torch.sum(attribution_map, 1, keepdim=True)
    if norm:
        if keepmean:
            attmean = torch.mean(attribution_map)
            attmin = torch.min(attribution_map)
            attmax = torch.max(attribution_map)
        attribution_map = normalize_batch(attribution_map) # Normalize attribution maps per image in batch
        if keepmean:
            attribution_map -= attribution_map.mean()
            attribution_map += (attmean / (attmax - attmin))
        
    return attribution_map

def get_gaussian(img, grad_wrt, norm=True, absolute=True, grayscale=True, keepmean=False):
    """
    Generate Gaussian noise based on the input image.

    Args:
        img (torch.Tensor): Input image.
        grad_wrt: Gradient with respect to the input image.
        norm (bool, optional): Whether to normalize the generated noise. Defaults to True.
        absolute (bool, optional): Whether to take the absolute values of the gradients. Defaults to True.
        grayscale (bool, optional): Whether to convert the noise to grayscale. Defaults to True.
        keepmean (bool, optional): Whether to keep the mean of the noise. Defaults to False.

    Returns:
        torch.Tensor: Generated Gaussian noise.
    """
    
    gaussian_noise = torch.randn_like(img)
    
    if absolute:
        gaussian_noise = torch.abs(gaussian_noise) # Take absolute values of gradients
    if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
        gaussian_noise = torch.sum(gaussian_noise, 1, keepdim=True)
    if norm:
        if keepmean:
            attmean = torch.mean(gaussian_noise)
            attmin = torch.min(gaussian_noise)
            attmax = torch.max(gaussian_noise)
        gaussian_noise = normalize_batch(gaussian_noise) # Normalize attribution maps per image in batch
        if keepmean:
            gaussian_noise -= gaussian_noise.mean()
            gaussian_noise += (attmean / (attmax - attmin))
        
    return gaussian_noise
    

def get_plaus_score(targets_out, attr, debug=False, corners=False, imgs=None, eps = 1e-7):
    # TODO: Remove imgs from this function and only take it as input if debug is True
    """
    Calculates the plausibility score based on the given inputs.

    Args:
        imgs (torch.Tensor): The input images.
        targets_out (torch.Tensor): The output targets.
        attr (torch.Tensor): The attribute tensor.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        torch.Tensor: The plausibility score.
    """
    # # if imgs is None:
    # #     imgs = torch.zeros_like(attr)
    # # with torch.no_grad():
    # target_inds = targets_out[:, 0].int()
    # xyxy_batch = targets_out[:, 2:6]# * pre_gen_gains[out_num]
    # num_pixels = torch.tile(torch.tensor([attr.shape[2], attr.shape[3], attr.shape[2], attr.shape[3]], device=attr.device), (xyxy_batch.shape[0], 1))
    # # num_pixels = torch.tile(torch.tensor([1.0, 1.0, 1.0, 1.0], device=imgs.device), (xyxy_batch.shape[0], 1))
    # xyxy_corners = (corners_coords_batch(xyxy_batch) * num_pixels).int()
    # co = xyxy_corners
    # if corners:
    #     co = targets_out[:, 2:6].int()
    # coords_map = torch.zeros_like(attr, dtype=torch.bool)
    # # rows = np.arange(co.shape[0])
    # x1, x2 = co[:,1], co[:,3]
    # y1, y2 = co[:,0], co[:,2]
    
    # for ic in range(co.shape[0]): # potential for speedup here with torch indexing instead of for loop
    #     coords_map[target_inds[ic], :,x1[ic]:x2[ic],y1[ic]:y2[ic]] = True

    if torch.isnan(attr).any():
        attr = torch.nan_to_num(attr, nan=0.0)
    
    coords_map = get_bbox_map(targets_out, attr)
    plaus_score = ((torch.sum((attr * coords_map))) / (torch.sum(attr)))

    if debug:
        for i in range(len(coords_map)):
            coords_map3ch = torch.cat([coords_map[i][:1], coords_map[i][:1], coords_map[i][:1]], dim=0)
            test_bbox = torch.zeros_like(imgs[i])
            test_bbox[coords_map3ch] = imgs[i][coords_map3ch]
            imshow(test_bbox, save_path='figs/test_bbox')
            if imgs is None:
                imgs = torch.zeros_like(attr)
            imshow(imgs[i], save_path='figs/im0')
            imshow(attr[i], save_path='figs/attr')
    
    # with torch.no_grad():
    # # att_select = attr[coords_map]
    # att_select = attr * coords_map.to(torch.float32)
    # att_total = attr
    
    # IoU_num = torch.sum(att_select)
    # IoU_denom = torch.sum(att_total)
    
    # IoU_ = (IoU_num / IoU_denom)
    # plaus_score = IoU_

    # # plaus_score = ((torch.sum(attr[coords_map])) / (torch.sum(attr)))
    
    return plaus_score

def get_attr_corners(targets_out, attr, debug=False, corners=False, imgs=None, eps = 1e-7):
    # TODO: Remove imgs from this function and only take it as input if debug is True
    """
    Calculates the plausibility score based on the given inputs.

    Args:
        imgs (torch.Tensor): The input images.
        targets_out (torch.Tensor): The output targets.
        attr (torch.Tensor): The attribute tensor.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        torch.Tensor: The plausibility score.
    """
    # if imgs is None:
    #     imgs = torch.zeros_like(attr)
    # with torch.no_grad():
    target_inds = targets_out[:, 0].int()
    xyxy_batch = targets_out[:, 2:6]# * pre_gen_gains[out_num]
    num_pixels = torch.tile(torch.tensor([attr.shape[2], attr.shape[3], attr.shape[2], attr.shape[3]], device=attr.device), (xyxy_batch.shape[0], 1))
    # num_pixels = torch.tile(torch.tensor([1.0, 1.0, 1.0, 1.0], device=imgs.device), (xyxy_batch.shape[0], 1))
    xyxy_corners = (corners_coords_batch(xyxy_batch) * num_pixels).int()
    co = xyxy_corners
    if corners:
        co = targets_out[:, 2:6].int()
    coords_map = torch.zeros_like(attr, dtype=torch.bool)
    # rows = np.arange(co.shape[0])
    x1, x2 = co[:,1], co[:,3]
    y1, y2 = co[:,0], co[:,2]
    
    for ic in range(co.shape[0]): # potential for speedup here with torch indexing instead of for loop
        coords_map[target_inds[ic], :,x1[ic]:x2[ic],y1[ic]:y2[ic]] = True

    if torch.isnan(attr).any():
        attr = torch.nan_to_num(attr, nan=0.0)
    if debug:
        for i in range(len(coords_map)):
            coords_map3ch = torch.cat([coords_map[i][:1], coords_map[i][:1], coords_map[i][:1]], dim=0)
            test_bbox = torch.zeros_like(imgs[i])
            test_bbox[coords_map3ch] = imgs[i][coords_map3ch]
            imshow(test_bbox, save_path='figs/test_bbox')
            imshow(imgs[i], save_path='figs/im0')
            imshow(attr[i], save_path='figs/attr')
    
    # att_select = attr[coords_map]
    # with torch.no_grad():
    # IoU_num = (torch.sum(attr[coords_map]))
    # IoU_denom = torch.sum(attr)
    # IoU_ = (IoU_num / (IoU_denom))
    
    # IoU_ = torch.max(attr[coords_map]) - torch.max(attr[~coords_map])
    co = (xyxy_batch * num_pixels).int()
    x1 = co[:,1] + 1
    y1 = co[:,0] + 1
    # with torch.no_grad():
    attr_ = torch.sum(attr, 1, keepdim=True)
    corners_attr = None #torch.zeros(len(xyxy_batch), 4, device=attr.device)
    for ic in range(co.shape[0]):
        attr0 = attr_[target_inds[ic], :,:x1[ic],:y1[ic]]
        attr1 = attr_[target_inds[ic], :,:x1[ic],y1[ic]:]
        attr2 = attr_[target_inds[ic], :,x1[ic]:,:y1[ic]]
        attr3 = attr_[target_inds[ic], :,x1[ic]:,y1[ic]:]

        x_0, y_0 = max_indices_2d(attr0[0])
        x_1, y_1 = max_indices_2d(attr1[0])
        x_2, y_2 = max_indices_2d(attr2[0])
        x_3, y_3 = max_indices_2d(attr3[0])

        y_1 += y1[ic]
        x_2 += x1[ic]
        x_3 += x1[ic]
        y_3 += y1[ic]

        max_corners = torch.cat([torch.min(x_0, x_2).unsqueeze(0) / attr_.shape[2],
                                    torch.min(y_0, y_1).unsqueeze(0) / attr_.shape[3],
                                    torch.max(x_1, x_3).unsqueeze(0) / attr_.shape[2],
                                    torch.max(y_2, y_3).unsqueeze(0) / attr_.shape[3]])
        if corners_attr is None:
            corners_attr = max_corners
        else:
            corners_attr = torch.cat([corners_attr, max_corners], dim=0)
        # corners_attr[ic] = max_corners
        # corners_attr = attr[:,0,:4,0]
    corners_attr = corners_attr.view(-1, 4)
    # corners_attr = torch.stack(corners_attr, dim=0)
    IoU_ = bbox_iou(corners_attr.T, xyxy_batch, x1y1x2y2=False, metric='CIoU')
    plaus_score = IoU_.mean()

    return plaus_score

def max_indices_2d(x_inp):
    # values, indices = x.reshape(x.size(0), -1).max(dim=-1)
    torch.max(x_inp,)
    index = torch.argmax(x_inp)
    x = index // x_inp.shape[1]
    y = index % x_inp.shape[1]
    # x, y = divmod(index.item(), x_inp.shape[1])

    return torch.cat([x.unsqueeze(0), y.unsqueeze(0)])


def point_in_polygon(poly, grid):
    # t0 = time.time()
    num_points = poly.shape[0]
    j = num_points - 1
    oddNodes = torch.zeros_like(grid[..., 0], dtype=torch.bool)
    for i in range(num_points):
        cond1 = (poly[i, 1] < grid[..., 1]) & (poly[j, 1] >= grid[..., 1])
        cond2 = (poly[j, 1] < grid[..., 1]) & (poly[i, 1] >= grid[..., 1])
        cond3 = (grid[..., 0] - poly[i, 0]) < (poly[j, 0] - poly[i, 0]) * (grid[..., 1] - poly[i, 1]) / (poly[j, 1] - poly[i, 1])
        oddNodes = oddNodes ^ (cond1 | cond2) & cond3
        j = i
    # t1 = time.time()
    # print(f'point in polygon time: {t1-t0}')
    return oddNodes
    
def point_in_polygon_gpu(poly, grid):
    num_points = poly.shape[0]
    i = torch.arange(num_points)
    j = (i - 1) % num_points
    # Expand dimensions
    # t0 = time.time()
    poly_expanded = poly.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, grid.shape[0], grid.shape[0])
    # t1 = time.time()
    cond1 = (poly_expanded[i, 1] < grid[..., 1]) & (poly_expanded[j, 1] >= grid[..., 1])
    cond2 = (poly_expanded[j, 1] < grid[..., 1]) & (poly_expanded[i, 1] >= grid[..., 1])
    cond3 = (grid[..., 0] - poly_expanded[i, 0]) < (poly_expanded[j, 0] - poly_expanded[i, 0]) * (grid[..., 1] - poly_expanded[i, 1]) / (poly_expanded[j, 1] - poly_expanded[i, 1])
    # t2 = time.time()
    oddNodes = torch.zeros_like(grid[..., 0], dtype=torch.bool)
    cond = (cond1 | cond2) & cond3
    # t3 = time.time()
    # efficiently perform xor using gpu and avoiding cpu as much as possible
    c = []
    while len(cond) > 1: 
        if len(cond) % 2 == 1: # odd number of elements
            c.append(cond[-1])
            cond = cond[:-1]
        cond = torch.bitwise_xor(cond[:int(len(cond)/2)], cond[int(len(cond)/2):])
    for c_ in c:
        cond = torch.bitwise_xor(cond, c_)
    oddNodes = cond
    # t4 = time.time()
    # for c in cond:
    #     oddNodes = oddNodes ^ c
    # print(f'expand time: {t1-t0} | cond123 time: {t2-t1} | cond logic time: {t3-t2} |  bitwise xor time: {t4-t3}')
    # print(f'point in polygon time gpu: {t4-t0}')
    # oddNodes = oddNodes ^ (cond1 | cond2) & cond3
    return oddNodes


def bitmap_for_polygon(poly, h, w):
    y = torch.arange(h).to(poly.device).float()
    x = torch.arange(w).to(poly.device).float()
    grid_y, grid_x = torch.meshgrid(y, x)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    bitmap = point_in_polygon(poly, grid)
    return bitmap.unsqueeze(0)


def corners_coords(center_xywh):
    center_x, center_y, w, h = center_xywh
    x = center_x - w/2
    y = center_y - h/2
    return torch.tensor([x, y, x+w, y+h])

def corners_coords_batch(center_xywh):
    center_x, center_y = center_xywh[:,0], center_xywh[:,1]
    w, h = center_xywh[:,2], center_xywh[:,3]
    x = center_x - w/2
    y = center_y - h/2
    return torch.stack([x, y, x+w, y+h], dim=1)
    
def normalize_batch(x):
    """
    Normalize a batch of tensors along each channel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
    Returns:
        torch.Tensor: Normalized tensor of the same shape as the input.
    """
    mins = torch.zeros((x.shape[0], *(1,)*len(x.shape[1:])), device=x.device)
    maxs = torch.zeros((x.shape[0], *(1,)*len(x.shape[1:])), device=x.device)
    for i in range(x.shape[0]):
        mins[i] = x[i].min()
        maxs[i] = x[i].max()
    x_ = (x - mins) / (maxs - mins)
    
    return x_

def get_detections(model_clone, img):
    """
    Get detections from a model given an input image and targets.

    Args:
        model (nn.Module): The model to use for detection.
        img (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The detected bounding boxes.
    """
    model_clone.eval() # Set model to evaluation mode
    # Run inference
    with torch.no_grad():
        det_out, out = model_clone(img)
    
    # model_.train()
    del img 
    
    return det_out, out

def get_labels(det_out, imgs, targets, opt):
    ###################### Get predicted labels ###################### 
    nb, _, height, width = imgs.shape  # batch size, channels, height, width 
    targets_ = targets.clone() 
    targets_[:, 2:] = targets_[:, 2:] * torch.Tensor([width, height, width, height]).to(imgs.device)  # to pixels
    lb = [targets_[targets_[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
    o = non_max_suppression(det_out, conf_thres=0.001, iou_thres=0.6, labels=lb, multi_label=True)
    pred_labels = [] 
    for si, pred in enumerate(o):
        labels = targets_[targets_[:, 0] == si, 1:]
        nl = len(labels) 
        predn = pred.clone()
        # Get the indices that sort the values in column 5 in ascending order
        sort_indices = torch.argsort(pred[:, 4], dim=0, descending=True)
        # Apply the sorting indices to the tensor
        sorted_pred = predn[sort_indices]
        # Remove predictions with less than 0.1 confidence
        n_conf = int(torch.sum(sorted_pred[:,4]>0.1)) + 1
        sorted_pred = sorted_pred[:n_conf]
        new_col = torch.ones((sorted_pred.shape[0], 1), device=imgs.device) * si
        preds = torch.cat((new_col, sorted_pred[:, [5, 0, 1, 2, 3]]), dim=1)
        preds[:, 2:] = xyxy2xywh(preds[:, 2:])  # xywh
        gn = torch.tensor([width, height])[[1, 0, 1, 0]]  # normalization gain whwh
        preds[:, 2:] /= gn.to(imgs.device)  # from pixels
        pred_labels.append(preds)
    pred_labels = torch.cat(pred_labels, 0).to(imgs.device)
    
    return pred_labels
    ##################################################################

from torchvision.utils import make_grid

def get_center_coords(attr):
    img_tensor = img_tensor / img_tensor.max()

    # Define a brightness threshold
    threshold = 0.95

    # Create a binary mask of the bright pixels
    mask = img_tensor > threshold

    # Get the coordinates of the bright pixels
    y_coords, x_coords = torch.where(mask)

    # Calculate the centroid of the bright pixels
    centroid_x = x_coords.float().mean().item()
    centroid_y = y_coords.float().mean().item()

    print(f'The central bright point is at ({centroid_x}, {centroid_y})')
    
    return


def get_distance_grids(attr, targets, imgs=None, focus_coeff=0.5, debug=False):
    """
    Compute the distance grids from each pixel to the target coordinates.

    Args:
        attr (torch.Tensor): Attribution maps.
        targets (torch.Tensor): Target coordinates.
        focus_coeff (float, optional): Focus coefficient, smaller means more focused. Defaults to 0.5.
        debug (bool, optional): Whether to visualize debug information. Defaults to False.

    Returns:
        torch.Tensor: Distance grids.
    """
    
    # Assign the height and width of the input tensor to variables
    height, width = attr.shape[-1], attr.shape[-2]
    
    # attr = torch.abs(attr) # Take absolute values of gradients
    # attr = normalize_batch(attr) # Normalize attribution maps per image in batch

    # Create a grid of indices
    xx, yy = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width))).to(attr.device)
    idx_grid = torch.stack((xx, yy), dim=-1).float()
    
    # Expand the grid to match the batch size
    idx_batch_grid = idx_grid.expand(attr.shape[0], -1, -1, -1)
    
    # Initialize a list to store the distance grids
    dist_grids_ = [[]] * attr.shape[0]

    # Loop over batches
    for j in range(attr.shape[0]):
        # Get the rows where the first column is the current unique value
        rows = targets[targets[:, 0] == j]
        
        if len(rows) != 0: 
            # Create a tensor for the target coordinates
            xy = rows[:,2:4] # y, x
            # Flip the x and y coordinates and scale them to the image size
            xy[:, 0], xy[:, 1] = xy[:, 1] * width, xy[:, 0] * height # y, x to x, y
            xy_center = xy.unsqueeze(1).unsqueeze(1)#.requires_grad_(True) 
            
            # Compute the Euclidean distance from each pixel to the target coordinates
            dists = torch.norm(idx_batch_grid[j].expand(len(xy_center), -1, -1, -1) - xy_center, dim=-1)

            # Pick the closest distance to any target for each pixel 
            dist_grid_ = torch.min(dists, dim=0)[0].unsqueeze(0) 
            dist_grid = torch.cat([dist_grid_, dist_grid_, dist_grid_], dim=0) if attr.shape[1] == 3 else dist_grid_
        else:
            # Set grid to zero if no targets are present
            dist_grid = torch.zeros_like(attr[j])
            
        dist_grids_[j] = dist_grid
    # Convert the list of distance grids to a tensor for faster computation
    dist_grids = normalize_batch(torch.stack(dist_grids_)) ** focus_coeff
    if torch.isnan(dist_grids).any():
        dist_grids = torch.nan_to_num(dist_grids, nan=0.0)

    if debug:
        for i in range(len(dist_grids)):
            if ((i % 8) == 0):
                grid_show = torch.cat([dist_grids[i][:1], dist_grids[i][:1], dist_grids[i][:1]], dim=0)
                imshow(grid_show, save_path='figs/dist_grids')
                if imgs is None:
                    imgs = torch.zeros_like(attr)
                imshow(imgs[i], save_path='figs/im0')
                img_overlay = (overlay_mask(imgs[i], dist_grids[i][0], alpha = 0.75))
                imshow(img_overlay, save_path='figs/dist_grid_overlay')
                weighted_attr = (dist_grids[i] * attr[i])
                imshow(weighted_attr, save_path='figs/weighted_attr')
                imshow(attr[i], save_path='figs/attr')

    return dist_grids

def attr_reg(attribution_map, distance_map):

    # dist_attr = distance_map * attribution_map 
    dist_attr = torch.mean(distance_map * attribution_map)#, dim=(1, 2, 3)) 
    # del distance_map, attribution_map
    return dist_attr

def get_bbox_map(targets_out, attr, corners=False):
    target_inds = targets_out[:, 0].int()
    xyxy_batch = targets_out[:, 2:6]# * pre_gen_gains[out_num]
    num_pixels = torch.tile(torch.tensor([attr.shape[2], attr.shape[3], attr.shape[2], attr.shape[3]], device=attr.device), (xyxy_batch.shape[0], 1))
    # num_pixels = torch.tile(torch.tensor([1.0, 1.0, 1.0, 1.0], device=imgs.device), (xyxy_batch.shape[0], 1))
    xyxy_corners = (corners_coords_batch(xyxy_batch) * num_pixels).int()
    co = xyxy_corners
    if corners:
        co = targets_out[:, 2:6].int()
    coords_map = torch.zeros_like(attr, dtype=torch.bool)
    # rows = np.arange(co.shape[0])
    x1, x2 = co[:,1], co[:,3]
    y1, y2 = co[:,0], co[:,2]
    
    for ic in range(co.shape[0]): # potential for speedup here with torch indexing instead of for loop
        coords_map[target_inds[ic], :,x1[ic]:x2[ic],y1[ic]:y2[ic]] = True
    
    bbox_map = coords_map.to(torch.float32)

    return bbox_map
######################################## BCE #######################################
def get_plaus_loss(targets, attribution_map, opt, imgs=None, debug=False, only_loss=False):
    # if imgs is None:
    #     imgs = torch.zeros_like(attribution_map)
    # Calculate Plausibility IoU with attribution maps
    # attribution_map.retains_grad = True
    if not only_loss:
        plaus_score = get_plaus_score(targets_out = targets, attr = attribution_map.clone().detach().requires_grad_(True), imgs = imgs)
    else:
        plaus_score = torch.tensor(0.0)
    
    # attribution_map = normalize_batch(attribution_map) # Normalize attribution maps per image in batch

    # Calculate distance regularization
    distance_map = get_distance_grids(attribution_map, targets, imgs, opt.focus_coeff)
    # distance_map = torch.ones_like(attribution_map)
    
    if opt.dist_x_bbox:
        bbox_map = get_bbox_map(targets, attribution_map).to(torch.bool)
        distance_map[bbox_map] = 0.0
        # distance_map = distance_map * (1 - bbox_map)

    # Positive regularization term for incentivizing pixels near the target to have high attribution
    dist_attr_pos = attr_reg(attribution_map, (1.0 - distance_map))
    # Negative regularization term for incentivizing pixels far from the target to have low attribution
    dist_attr_neg = attr_reg(attribution_map, distance_map)
    # Calculate plausibility regularization term
    # dist_reg = dist_attr_pos - dist_attr_neg
    dist_reg = ((dist_attr_pos / torch.mean(attribution_map)) - (dist_attr_neg / torch.mean(attribution_map)))
    # dist_reg = torch.mean((dist_attr_pos / torch.mean(attribution_map, dim=(1, 2, 3))) - (dist_attr_neg / torch.mean(attribution_map, dim=(1, 2, 3)))) 
    # dist_reg = (torch.mean(torch.exp((dist_attr_pos / torch.mean(attribution_map, dim=(1, 2, 3)))) + \
    #                             torch.exp(1 - (dist_attr_neg / torch.mean(attribution_map, dim=(1, 2, 3)))))) \
    #                             / 2.5

    if opt.bbox_coeff != 0.0:
        bbox_map = get_bbox_map(targets, attribution_map)
        attr_bbox_pos = attr_reg(attribution_map, bbox_map)
        attr_bbox_neg = attr_reg(attribution_map, (1.0 - bbox_map))
        bbox_reg = attr_bbox_pos - attr_bbox_neg
        # bbox_reg = (attr_bbox_pos / torch.mean(attribution_map)) - (attr_bbox_neg / torch.mean(attribution_map))
    else:
        bbox_reg = 0.0

    bbox_map = get_bbox_map(targets, attribution_map)
    plaus_score = ((torch.sum((attribution_map * bbox_map))) / (torch.sum(attribution_map)))
    # iou_loss = (1.0 - plaus_score)

    if not opt.dist_reg_only:
        dist_reg_loss = (((1.0 + dist_reg) / 2.0))
        plaus_reg = (plaus_score * opt.iou_coeff) + \
                    (((dist_reg_loss * opt.dist_coeff) + \
                      (bbox_reg * opt.bbox_coeff))\
                    # ((((((1.0 + dist_reg) / 2.0) - 1.0) * opt.dist_coeff) + ((((1.0 + bbox_reg) / 2.0) - 1.0) * opt.bbox_coeff))\
                    # / (plaus_score) \
                    )
    else:
        plaus_reg = (((1.0 + dist_reg) / 2.0))
        # plaus_reg = dist_reg 
    # Calculate plausibility loss
    plaus_loss = (1 - plaus_reg) * opt.pgt_coeff
    # plaus_loss = (plaus_reg) * opt.pgt_coeff
    if only_loss:
        return plaus_loss
    if not debug:
        return plaus_loss, (plaus_score, dist_reg, plaus_reg,)
    else:
        return plaus_loss, (plaus_score, dist_reg, plaus_reg,), distance_map

####################################################################################
#### ALL FUNCTIONS BELOW ARE DEPRECIATED AND WILL BE REMOVED IN FUTURE VERSIONS ####
####################################################################################

def generate_vanilla_grad(model, input_tensor, loss_func = None, 
                          targets_list=None, targets=None, metric=None, out_num = 1, 
                          n_max_labels=3, norm=True, abs=True, grayscale=True, 
                          class_specific_attr = True, device='cpu'):    
    """
    Generate vanilla gradients for the given model and input tensor.

    Args:
        model (nn.Module): The model to generate gradients for.
        input_tensor (torch.Tensor): The input tensor for which gradients are computed.
        loss_func (callable, optional): The loss function to compute gradients with respect to. Defaults to None.
        targets_list (list, optional): The list of target tensors. Defaults to None.
        metric (callable, optional): The metric function to evaluate the loss. Defaults to None.
        out_num (int, optional): The index of the output tensor to compute gradients with respect to. Defaults to 1.
        n_max_labels (int, optional): The maximum number of labels to consider. Defaults to 3.
        norm (bool, optional): Whether to normalize the attribution map. Defaults to True.
        abs (bool, optional): Whether to take the absolute values of gradients. Defaults to True.
        grayscale (bool, optional): Whether to convert the attribution map to grayscale. Defaults to True.
        class_specific_attr (bool, optional): Whether to compute class-specific attribution maps. Defaults to True.
        device (str, optional): The device to use for computation. Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: The generated vanilla gradients.
    """
    # Set model.train() at the beginning and revert back to original mode (model.eval() or model.train()) at the end
    train_mode = model.training
    if not train_mode:
        model.train()
    
    input_tensor.requires_grad = True # Set requires_grad attribute of tensor. Important for computing gradients
    model.zero_grad() # Zero gradients
    inpt = input_tensor
    # Forward pass
    train_out = model(inpt) # training outputs (no inference outputs in train mode)
    
    # train_out[1] = torch.Size([4, 3, 80, 80, 7]) HxWx(#anchorxC) cls (class probabilities)
    # train_out[0] = torch.Size([4, 3, 160, 160, 7]) HxWx(#anchorx4) box or reg (location and scaling)
    # train_out[2] = torch.Size([4, 3, 40, 40, 7]) HxWx(#anchorx1) obj (objectness score or confidence)
    
    if class_specific_attr:
        n_attr_list, index_classes = [], []
        for i in range(len(input_tensor)):
            if len(targets_list[i]) > n_max_labels:
                targets_list[i] = targets_list[i][:n_max_labels]
            if targets_list[i].numel() != 0:
                # unique_classes = torch.unique(targets_list[i][:,1])
                class_numbers = targets_list[i][:,1]
                index_classes.append([[0, 1, 2, 3, 4, int(uc)] for uc in class_numbers])
                num_attrs = len(targets_list[i])
                # index_classes.append([0, 1, 2, 3, 4] + [int(uc + 5) for uc in unique_classes])
                # num_attrs = 1 #len(unique_classes)# if loss_func else len(targets_list[i])
                n_attr_list.append(num_attrs)
            else:
                index_classes.append([0, 1, 2, 3, 4])
                n_attr_list.append(0)
    
        targets_list_filled = [targ.clone().detach() for targ in targets_list]
        labels_len = [len(targets_list[ih]) for ih in range(len(targets_list))]
        max_labels = np.max(labels_len)
        max_index = np.argmax(labels_len)
        for i in range(len(targets_list)):
            # targets_list_filled[i] = targets_list[i]
            if len(targets_list_filled[i]) < max_labels:
                tlist = [targets_list_filled[i]] * math.ceil(max_labels / len(targets_list_filled[i]))
                targets_list_filled[i] = torch.cat(tlist)[:max_labels].unsqueeze(0)
            else:
                targets_list_filled[i] = targets_list_filled[i].unsqueeze(0)
        for i in range(len(targets_list_filled)-1,-1,-1):
            if targets_list_filled[i].numel() == 0:
                targets_list_filled.pop(i)
        targets_list_filled = torch.cat(targets_list_filled)
    
    n_img_attrs = len(input_tensor) if class_specific_attr else 1
    n_img_attrs = 1 if loss_func else n_img_attrs
    
    attrs_batch = []
    for i_batch in range(n_img_attrs):
        if loss_func and class_specific_attr:
            i_batch = max_index
        # inpt = input_tensor[i_batch].unsqueeze(0)
        # ##################################################################
        # model.zero_grad() # Zero gradients
        # train_out = model(inpt)  # training outputs (no inference outputs in train mode)
        # ##################################################################
        n_label_attrs = n_attr_list[i_batch] if class_specific_attr else 1
        n_label_attrs = 1 if not class_specific_attr else n_label_attrs
        attrs_img = []
        for i_attr in range(n_label_attrs):
            if loss_func is None:
                grad_wrt = train_out[out_num]
                if class_specific_attr:
                    grad_wrt = train_out[out_num][:,:,:,:,index_classes[i_batch][i_attr]]
                grad_wrt_outputs = torch.ones_like(grad_wrt)
            else:
                # if class_specific_attr:
                #     targets = targets_list[:][i_attr]
                # n_targets = len(targets_list[i_batch])
                if class_specific_attr:
                    target_indiv = targets_list_filled[:,i_attr] # batch image input
                else:
                    target_indiv = targets
                # target_indiv = targets_list[i_batch][i_attr].unsqueeze(0) # single image input
                # target_indiv[:,0] = 0 # this indicates the batch index of the target, should be 0 since we are only doing one image at a time
                    
                try:
                    loss, loss_items = loss_func(train_out, target_indiv, inpt, metric=metric)  # loss scaled by batch_size
                except:
                    target_indiv = target_indiv.to(device)
                    inpt = inpt.to(device)
                    for tro in train_out:
                        tro = tro.to(device)
                    print("Error in loss function, trying again with device specified")
                    loss, loss_items = loss_func(train_out, target_indiv, inpt, metric=metric)
                grad_wrt = loss
                grad_wrt_outputs = None
            
            model.zero_grad() # Zero gradients
            gradients = torch.autograd.grad(grad_wrt, inpt, 
                                                grad_outputs=grad_wrt_outputs, 
                                                retain_graph=True, 
                                                # create_graph=True, # Create graph to allow for higher order derivatives but slows down computation significantly
                                                )

            # Convert gradients to numpy array and back to ensure full separation from graph
            # attribution_map = torch.tensor(torch.sum(gradients[0], 1, keepdim=True).clone().detach().cpu().numpy())
            attribution_map = gradients[0]#.clone().detach() # without converting to numpy
            
            if grayscale: # Convert to grayscale, saves vram and computation time for plaus_eval
                attribution_map = torch.sum(attribution_map, 1, keepdim=True)
            if abs:
                attribution_map = torch.abs(attribution_map) # Take absolute values of gradients
            if norm:
                attribution_map = normalize_batch(attribution_map) # Normalize attribution maps per image in batch
            attrs_img.append(attribution_map)
        if len(attrs_img) == 0:
            attrs_batch.append((torch.zeros_like(inpt).unsqueeze(0)).to(device))
        else:
            attrs_batch.append(torch.stack(attrs_img).to(device))

    # out_attr = torch.tensor(attribution_map).unsqueeze(0).to(device) if ((loss_func) or (not class_specific_attr)) else torch.stack(attrs_batch).to(device)
    # out_attr = [attrs_batch[0]] * len(input_tensor) if ((loss_func) or (not class_specific_attr)) else attrs_batch
    out_attr = attrs_batch
    # Set model back to original mode
    if not train_mode:
        model.eval()
    
    return out_attr

class RVNonLinearFunc(torch.nn.Module):
    """
    Custom Bayesian ReLU activation function for random variables.

    Attributes:
        None
    """
    def __init__(self, func):
        super(RVNonLinearFunc, self).__init__()
        self.func = func

    def forward(self, mu_in, Sigma_in):
        """
        Forward pass of the Bayesian ReLU activation function.

        Args:
            mu_in (torch.Tensor): A tensor of shape (batch_size, input_size),
                representing the mean input to the ReLU activation function.
            Sigma_in (torch.Tensor): A tensor of shape (batch_size, input_size, input_size),
                representing the covariance input to the ReLU activation function.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors,
                including the mean of the output and the covariance of the output.
        """
        # Collect stats
        batch_size = mu_in.size(0)
       
        # Mean
        mu_out = self.func(mu_in)
        
        # Compute the derivative of the ReLU activation function with respect to the input mean
        gradi = torch.autograd.grad(mu_out, mu_in, grad_outputs=torch.ones_like(mu_out), create_graph=True)[0].view(batch_size,-1)

        # add an extra dimension to gradi at position 2 and 1
        grad1 = gradi.unsqueeze(dim=2)
        grad2 = gradi.unsqueeze(dim=1)
       
        # compute the outer product of grad1 and grad2
        outer_product = torch.bmm(grad1, grad2)
       
        # element-wise multiply Sigma_in with the outer product
        # and return the result
        Sigma_out = torch.mul(Sigma_in, outer_product)

        return mu_out, Sigma_out

