import torch.nn.functional as F

def apply_mask_func(x, mask):
    mask_hard = mask.hard
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    return mask_hard.float().expand_as(x) * x

def apply_mask(x, mask):
    channel_mask = mask.get('channel', None)
    spatial_mask = mask.get('spatial', None)
    if channel_mask is not None:
        assert channel_mask.hard.shape[0] == x.shape[0]
        assert channel_mask.hard.shape[1] == x.shape[1], (channel_mask.hard.shape, x.shape)
        x = channel_mask.hard.float() * x
    if spatial_mask is not None:
        assert spatial_mask.hard.shape[0] == x.shape[0]
        assert spatial_mask.hard.shape[2:4] == x.shape[2:4], (spatial_mask.hard.shape, x.shape)
        x = spatial_mask.hard.float() * x
    return x

def ponder_cost_map(masks):
    """ takes in the mask list and returns a 2D image of ponder cost """
    assert isinstance(masks, list)
    out = None
    for mask in masks:
        m = mask['std'].hard
        assert m.dim() == 4
        m = m[0]  # only show the first image of the batch
        if out is None:
            out = m
        else:
            out += F.interpolate(m.unsqueeze(0),
                                 size=(out.shape[1], out.shape[2]), mode='nearest').squeeze(0)
    return out.squeeze(0).cpu().numpy()
