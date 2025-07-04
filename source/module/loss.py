import torch.nn.functional as F
import torchlight.nn.loss as tl
import torch

def landmark_loss(landmarks, flow, reduce='mean'):
    """ 
        landmarks: list of cords of each paired landmarks [B] -> [N,4]
        flow: offset of the flow [B,2,W,H]
        reduce: mean or batch
    """
    assert reduce in ['mean', 'batch']
    if reduce == 'batch':
        reduce = 'sum'

    loss = 0
    for idx, lm in enumerate(landmarks):
        source_lm = lm[:, 2:4]
        target_lm = lm[:, 0:2]
        loss += tl.landmark_loss(source_lm.unsqueeze(0),
                                 target_lm.unsqueeze(0), flow[idx:idx+1],
                                 reduce=reduce)
    return loss / (2*len(landmarks))


def warping_loss(output, target, reduce='mean'):
    assert reduce in ['mean', 'batch']
    if reduce == 'batch':
        return F.mse_loss(output, target, reduction='sum') / (2 * output.shape[0])
    else:
        return F.mse_loss(output, target, reduction='mean')


def charbonnier_loss(x, y, reduce='mean', eps=1e-3):
    diff = x - y
    loss = torch.sqrt((diff * diff) + (eps*eps))
    if reduce == 'mean':
        loss = torch.mean(loss)
    elif reduce == 'batch':
        loss = torch.sum(loss) / loss.shape[0]
    else:
        raise ValueError('Invalid reduce mode, choose from [mean, batch]')
    return loss


def feature_warping_loss(warped_feats, target, encoder):
    target_feats = encoder(target)
    loss = 0
    for f1, f2 in zip(warped_feats, list(target_feats)):
        loss += F.mse_loss(f1, f2)
    return loss


from torchlight.nn.ops.gradient import image_gradients

def gradient_weighted_loss(pred, target, reduce='mean', eps=1e-3):
    diff = pred - target
    loss = torch.sqrt((diff * diff) + (eps*eps))
    
    dx, dy = image_gradients(target)
    d = torch.sqrt(dx**2+dy**2)
    weight = d / torch.max(d)
    loss = loss * weight
    
    if reduce == 'mean':
        loss = torch.mean(loss)
    elif reduce == 'batch':
        loss = torch.sum(loss) / loss.shape[0]
    else:
        raise ValueError('Invalid reduce mode, choose from [mean, batch]')
    return loss + charbonnier_loss(pred, target, reduce, eps)

def gradient_warping_loss(output, target, reduce='mean'):
    assert reduce in ['mean', 'batch']
    dx1, dy1 = image_gradients(output)
    dx2, dy2 = image_gradients(target)
    distance = (dx1-dx2)**2 + (dy1-dy2)**2
    return torch.mean(distance)