import math

import torch
import torch.nn as nn

from utils.logger import logger,recoder
from utils.config import BaseConfig

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SparsityCriterion(nn.Module):
    ''' 
    Defines the sparsity loss, consisting of two parts:
    - network loss: MSE between computational budget used for whole network and target 
    - block loss: sparsity (percentage of used FLOPS between 0 and 1) in a block must lie between upper and lower bound. 
    This loss is annealed.
    '''

    def __init__(self, sparsity_target, num_epochs, type="spatial"):
        super(SparsityCriterion, self).__init__()
        # assert type in ["spatial", "channel", "mix"], "not support type:{}".format(type)
        # 传入的是保留的百分比，但是我们需要的是删除的百分比
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs
        self.type = type
        self.counter = 0

    def forward_spatial(self, meta):

        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        upper_bound = (1 - progress * (1 - self.sparsity_target))
        lower_bound = progress * self.sparsity_target

        loss_block = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            m_dil = mask['spatial']['dilate']
            m = mask['spatial']['std']

            c = m_dil.active_positions * m_dil.flops_per_position + \
                m.active_positions * m.flops_per_position
            t = m_dil.total_positions * m_dil.flops_per_position + \
                m.total_positions * m.flops_per_position

            layer_perc = c / t
            recoder.add('block_perc_' + str(i), layer_perc.item())
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc
            loss_block += max(0, layer_perc - upper_bound) ** 2  # upper bound
            loss_block += max(0, lower_bound - layer_perc) ** 2  # lower bound

            cost += c
            total += t

        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block /= len(meta['masks'])
        loss_network = (perc - self.sparsity_target) ** 2

        recoder.add('upper_bound', upper_bound)
        recoder.add('lower_bound', lower_bound)
        recoder.add('cost_perc', perc.item())
        recoder.add('loss_sp_block', loss_block.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block

    def forward_channel(self, meta):
        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        upper_bound = (1 - progress * (1 - self.sparsity_target))
        lower_bound = progress * self.sparsity_target

        loss_block = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            m = mask['channel']

            c = m.active_channels * m.flops_per_channel
            t = m.total_channels * m.flops_per_channel

            layer_perc = c / t
            recoder.add('layer_perc_' + str(i), layer_perc.item())
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc
            loss_block += max(0, layer_perc - upper_bound) ** 2  # upper bound
            loss_block += max(0, lower_bound - layer_perc) ** 2  # lower bound

            cost += c
            total += t

        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block /= len(meta['masks'])
        loss_network = (perc - self.sparsity_target) ** 2

        recoder.add('upper_bound', upper_bound)
        recoder.add('lower_bound', lower_bound)
        recoder.add('cost_perc', perc.item())
        recoder.add('loss_sp_block', loss_block.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block

    def forward_mix(self, meta):
        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        upper_bound = (1 - progress * (1 - self.sparsity_target))
        lower_bound = progress * self.sparsity_target

        loss_block = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        count = 0
        for i, mask in enumerate(meta['masks']):
            spatial_masks = mask['spatial']
            channel_masks = mask['channel']

            tempT = torch.tensor(.0).to(device=meta['device'])
            tempS = torch.tensor(.0).to(device=meta['device'])

            for j in range(len(spatial_masks)):
                spatial_mask = spatial_masks[j]
                channel_mask = channel_masks[j]
                channel_inte_state = channel_mask.channel_inte_state
                spatial_inte_state = spatial_mask.spatial_inte_state
                # channel_inte_state = channel_mask.channel_inte_state if self.counter % 2 == 0 else channel_mask.channel_inte_state.detach()
                # spatial_inte_state = spatial_mask.spatial_inte_state if self.counter % 2 == 1 else spatial_mask.spatial_inte_state.detach()
                tempS = spatial_mask.kernel_size * (channel_inte_state * spatial_inte_state).sum()
                tempT = spatial_mask.total_positions * spatial_mask.flops_per_position

            layer_perc = tempS / tempT
            recoder.add('layer_perc_' + str(i), layer_perc.item())
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc
            loss_block += max(0, layer_perc - upper_bound) ** 2  # upper bound
            loss_block += max(0, lower_bound - layer_perc) ** 2  # lower bound

            cost += tempS
            total += tempT

        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block /= len(meta['masks'])
        loss_network = (perc - self.sparsity_target) ** 2

        recoder.add('upper_bound', upper_bound)
        recoder.add('lower_bound', lower_bound)
        recoder.add('cost_perc', perc.item())
        recoder.add('loss_sp_block', loss_block.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block

    def forward_mix1(self, meta):
        sparsity_target = 1-self.sparsity_target
        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2

        upper_bound_channel = (1 - progress * (1 - sparsity_target/2))
        lower_bound_channel = progress * sparsity_target/2

        upper_bound_spatial = upper_bound_channel
        lower_bound_spatial = lower_bound_channel

        # upper_bound = (1 - progress * (1 - sparsity_target))
        # lower_bound = progress * sparsity_target

        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        # loss_block = torch.tensor(.0).to(device=meta['device'])

        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            spatial_mask = mask['spatial']
            channel_mask = mask['channel']
            # 需不需要让最后K轮联合训练
            # channel_inte_state = channel_mask.channel_inte_state if self.counter % 2 == 0 else channel_mask.channel_inte_state.detach()
            # spatial_inte_state = spatial_mask.spatial_inte_state if self.counter % 2 == 1 else spatial_mask.spatial_inte_state.detach()
            # channel_inte_state = channel_mask.channel_inte_state
            # spatial_inte_state = spatial_mask.spatial_inte_state
            self.counter += 1
            # 正常 -> spatial_mask.total_positions
            Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                  spatial_mask.total_positions * \
                  spatial_mask.kernel_size)/spatial_mask.batch_size
            # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
            Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) *\
                  channel_mask.active_channels *\
                  spatial_mask.kernel_size)/spatial_mask.batch_size

            t = spatial_mask.total_positions * spatial_mask.flops_per_position
            # print(Sci,t)
            layer_perc_channel = Sci / t
            layer_perc_spatial = Ssi / t
            layer_perc = layer_perc_channel + layer_perc_spatial

            recoder.add('cha_layer_perc_' + str(i), 1 - layer_perc_channel.item())
            recoder.add('spa_layer_perc_' + str(i), 1 - layer_perc_spatial.item())
            recoder.add('layer_perc_' + str(i), 1 - layer_perc.item())

            assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc

            loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
            loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound

            loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
            loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound

            # loss_block += max(0, layer_perc - upper_bound) ** 2  # upper bound
            # loss_block += max(0, lower_bound - layer_perc) ** 2  # lower bound

            cost += Sci + Ssi
            total += t

        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block_channel /= len(meta['masks'])
        loss_block_spatial /= len(meta['masks'])
        loss_network = (perc - sparsity_target) ** 2

        recoder.add('upper_bound', upper_bound_channel)
        recoder.add('lower_bound', lower_bound_channel)
        recoder.add('cost_perc', perc.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block_spatial*0.1 + loss_block_channel

    def forward_mix2(self, meta):
        sparsity_target = 1-self.sparsity_target
        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        # 1-sqrt(2)/2
        upper_bound_channel = (1 - progress * (1 -  0.293))
        lower_bound_channel = progress * 0.293

        upper_bound_spatial = upper_bound_channel
        lower_bound_spatial = lower_bound_channel

        # upper_bound = (1 - progress * (1 - sparsity_target))
        # lower_bound = progress * sparsity_target

        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        # loss_block = torch.tensor(.0).to(device=meta['device'])

        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            spatial_mask = mask['spatial']
            channel_mask = mask['channel']
            # 需不需要让最后K轮联合训练
            # channel_inte_state = channel_mask.channel_inte_state if self.counter % 2 == 0 else channel_mask.channel_inte_state.detach()
            # spatial_inte_state = spatial_mask.spatial_inte_state if self.counter % 2 == 1 else spatial_mask.spatial_inte_state.detach()
            # channel_inte_state = channel_mask.channel_inte_state
            # spatial_inte_state = spatial_mask.spatial_inte_state
            self.counter += 1
            # 正常 -> spatial_mask.total_positions
            Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                   spatial_mask.total_positions * \
                   spatial_mask.kernel_size)/spatial_mask.batch_size
            # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
            Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
                   channel_mask.total_channels * \
                   spatial_mask.kernel_size)/spatial_mask.batch_size

            Si = spatial_mask.kernel_size * \
                 (channel_mask.channel_inte_state * spatial_mask.spatial_inte_state).sum()

            t = spatial_mask.total_positions * spatial_mask.flops_per_position
            # print(Sci,t)
            layer_perc_channel = Sci / t
            layer_perc_spatial = Ssi / t
            layer_perc = 1 - (Si / t)

            recoder.add('cha_layer_perc_' + str(i), 1 - layer_perc_channel.item())
            recoder.add('spa_layer_perc_' + str(i), 1 - layer_perc_spatial.item())
            recoder.add('layer_perc_' + str(i), 1 - layer_perc.item())

            assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc

            loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
            loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound

            loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
            loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound

            # loss_block += max(0, layer_perc - upper_bound) ** 2  # upper bound
            # loss_block += max(0, lower_bound - layer_perc) ** 2  # lower bound

            cost += Si
            total += t

        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block_channel /= len(meta['masks'])
        loss_block_spatial /= len(meta['masks'])
        loss_network = (perc - sparsity_target) ** 2

        recoder.add('upper_bound', upper_bound_channel)
        recoder.add('lower_bound', lower_bound_channel)
        recoder.add('cost_perc', perc.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block_spatial + loss_block_channel

    def forward_mix3(self, meta):
        sparsity_target = 1-self.sparsity_target
        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        # 1-sqrt(2)/2
        upper_bound_channel = (1 - progress * (1 - sparsity_target * 0.5))
        lower_bound_channel = progress * sparsity_target * 0.25

        upper_bound_spatial = upper_bound_channel
        lower_bound_spatial = lower_bound_channel

        # upper_bound = (1 - progress * (1 - sparsity_target))
        # lower_bound = progress * sparsity_target

        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        # loss_block = torch.tensor(.0).to(device=meta['device'])

        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            spatial_mask = mask['spatial']
            channel_mask = mask['channel']
            # 需不需要让最后K轮联合训练
            # channel_inte_state = channel_mask.channel_inte_state if self.counter % 2 == 0 else channel_mask.channel_inte_state.detach()
            # spatial_inte_state = spatial_mask.spatial_inte_state if self.counter % 2 == 1 else spatial_mask.spatial_inte_state.detach()
            # channel_inte_state = channel_mask.channel_inte_state
            # spatial_inte_state = spatial_mask.spatial_inte_state
            self.counter += 1
            # 正常 -> spatial_mask.total_positions
            Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                   spatial_mask.total_positions * \
                   spatial_mask.kernel_size)/spatial_mask.batch_size
            # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
            Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
                   channel_mask.total_channels * \
                   spatial_mask.kernel_size)/spatial_mask.batch_size

            Si = spatial_mask.kernel_size * \
                 (channel_mask.channel_inte_state * spatial_mask.spatial_inte_state).sum()

            t = spatial_mask.total_positions * spatial_mask.flops_per_position
            # print(Sci,t)
            layer_perc_channel = Sci / t
            layer_perc_spatial = Ssi / t
            layer_perc = 1 - (Si / t)

            recoder.add('cha_layer_perc_' + str(i), 1 - layer_perc_channel.item())
            recoder.add('spa_layer_perc_' + str(i), 1 - layer_perc_spatial.item())
            recoder.add('layer_perc_' + str(i), 1 - layer_perc.item())

            assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc

            loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
            loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound

            loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
            loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound

            # loss_block += max(0, layer_perc - upper_bound) ** 2  # upper bound
            # loss_block += max(0, lower_bound - layer_perc) ** 2  # lower bound

            cost += Si
            total += t

        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block_channel /= len(meta['masks'])
        loss_block_spatial /= len(meta['masks'])
        loss_network = (perc - sparsity_target) ** 2

        recoder.add('upper_bound', upper_bound_channel)
        recoder.add('lower_bound', lower_bound_channel)
        recoder.add('cost_perc', perc.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block_spatial + loss_block_channel


    def forward_mix4(self, meta):
        sparsity_target = 1-math.sqrt(self.sparsity_target)
        # 通过控制progress的下界来调整最终约束的强度，当progress的最终值为0则比表明约束的范围是0——1，
        # 如果progress最终等于1，则约束的范围是theta
        # min(max(p, 0), 1) * my_pi  ~ [0,my_pi]
        # cos(min(max(p, 0), 1) * my_pi) ~ [1, cos(my_pi)]
        # my_pi =  math.pi / 2
        # p = meta['epoch'] / (0.33 * self.num_epochs)
        # progress = math.cos(min(max(p, 0), 1) * my_pi) ** 2
        progress = 1
        # 1-sqrt(2)/2
        upper_bound_channel = (1 - progress * (1 -  sparsity_target))
        lower_bound_channel = progress * sparsity_target

        upper_bound_spatial = upper_bound_channel
        lower_bound_spatial = lower_bound_channel

        # upper_bound = (1 - progress * (1 - sparsity_target))
        # lower_bound = progress * sparsity_target

        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        # loss_block = torch.tensor(.0).to(device=meta['device'])

        channel_cost,spatial_cost, total = torch.tensor(.0).to(device=meta['device']),\
                                           torch.tensor(.0).to(device=meta['device']),\
                                           torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            spatial_mask = mask['spatial']
            channel_mask = mask['channel']
            # 正常 -> spatial_mask.total_positions
            Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                   spatial_mask.total_positions * \
                   spatial_mask.kernel_size)/spatial_mask.batch_size
            # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
            Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
                   channel_mask.total_channels * \
                   spatial_mask.kernel_size)/spatial_mask.batch_size

            # Si = spatial_mask.kernel_size * \
            #      (channel_mask.channel_inte_state * spatial_mask.spatial_inte_state).sum()

            t = spatial_mask.total_positions * spatial_mask.flops_per_position
            # print(Sci,t)
            layer_perc_channel = Sci / t
            layer_perc_spatial = Ssi / t
            # layer_perc = 1 - (Si / t)

            recoder.add('cha_layer_perc_' + str(i), 1 - layer_perc_channel.item())
            recoder.add('spa_layer_perc_' + str(i), 1 - layer_perc_spatial.item())
            # recoder.add('layer_perc_' + str(i), 1 - layer_perc.item())

            assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            # assert layer_perc >= 0 and layer_perc <= 1, layer_perc

            loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
            loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound

            loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
            loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound

            # loss_block += max(0, layer_perc - upper_bound) ** 2  # upper bound
            # loss_block += max(0, lower_bound - layer_perc) ** 2  # lower bound

            channel_cost += Sci
            spatial_cost += Ssi
            total += t

        perc_channel = channel_cost / total
        prec_spatial = spatial_cost / total
        assert perc_channel >= 0 and perc_channel <= 1, perc_channel
        assert prec_spatial >= 0 and prec_spatial <= 1, prec_spatial
        loss_block_channel /= len(meta['masks'])
        loss_block_spatial /= len(meta['masks'])
        loss_channel_network = (perc_channel - sparsity_target) ** 2
        loss_spatial_network = (prec_spatial - sparsity_target) ** 2

        recoder.add('upper_bound', upper_bound_channel)
        recoder.add('lower_bound', lower_bound_channel)
        recoder.add('cost_perc_channel', perc_channel.item())
        recoder.add('cost_perc_spatial', prec_spatial.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_total_ch', loss_channel_network.item())
        recoder.add('loss_sp_total_sp', loss_spatial_network.item())
        return loss_channel_network + loss_spatial_network + loss_block_spatial + loss_block_channel

    # def forward_mix5(self, meta):
    #     sparsity_target = 1-math.sqrt(self.sparsity_target)
    #     # 通过控制progress的下界来调整最终约束的强度，当progress的最终值为0则比表明约束的范围是0——1，
    #     # 如果progress最终等于1，则约束的范围是theta
    #     # min(max(p, 0), 1) * my_pi  ~ [0,my_pi]
    #     # cos(min(max(p, 0), 1) * my_pi) ~ [1, cos(my_pi)]
    #     my_pi =  math.pi * 0.1845
    #     p = meta['epoch'] / (0.33 * self.num_epochs)
    #     progress = math.cos(min(max(p, 0), 1) * my_pi) ** 2
    #     # progress = 1
    #     # 1-sqrt(2)/2
    #     upper_bound_channel = (1 - progress * (1 -  sparsity_target))
    #     lower_bound_channel = progress * sparsity_target
    #
    #     upper_bound_spatial = upper_bound_channel
    #     lower_bound_spatial = lower_bound_channel
    #
    #     # upper_bound = (1 - progress * (1 - sparsity_target))
    #     # lower_bound = progress * sparsity_target
    #
    #     loss_block_channel = torch.tensor(.0).to(device=meta['device'])
    #     loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
    #     # loss_block = torch.tensor(.0).to(device=meta['device'])
    #
    #     channel_cost,spatial_cost, total = torch.tensor(.0).to(device=meta['device']), \
    #                                        torch.tensor(.0).to(device=meta['device']), \
    #                                        torch.tensor(.0).to(device=meta['device'])
    #
    #     for i, mask in enumerate(meta['masks']):
    #         spatial_masks = mask['spatial']
    #         channel_masks = mask['channel']
    #
    #         tempT = torch.tensor(.0).to(device=meta['device'])
    #         tempS = torch.tensor(.0).to(device=meta['device'])
    #         tempC = torch.tensor(.0).to(device=meta['device'])
    #         for i in range(len(spatial_masks)):
    #             spatial_mask = spatial_masks[i]
    #             channel_mask = channel_masks[i]
    #             # 正常 -> spatial_mask.total_positions
    #             Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
    #                    spatial_mask.total_positions * \
    #                    spatial_mask.kernel_size)/spatial_mask.batch_size
    #             # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
    #             Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
    #                    channel_mask.total_channels * \
    #                    spatial_mask.kernel_size)/spatial_mask.batch_size
    #
    #             # Si = spatial_mask.kernel_size * \
    #             #      (channel_mask.channel_inte_state * spatial_mask.spatial_inte_state).sum()
    #
    #             t = spatial_mask.total_positions * spatial_mask.flops_per_position
    #             tempT += t
    #             tempS += Ssi
    #             tempC += Sci
    #             # print(Sci,t)
    #             # layer_perc_channel = Sci / t
    #             # layer_perc_spatial = Ssi / t
    #             # layer_perc = 1 - (Si / t)
    #
    #             # recoder.add('layer_perc_channel_' + str(i), 1 - layer_perc_channel.item())
    #             # recoder.add('layer_perc_spatial_' + str(i), 1 - (Ssi / t).item())
    #             # recoder.add('layer_perc_' + str(i), 1 - layer_perc.item())
    #
    #             # assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
    #             # assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
    #             # assert layer_perc >= 0 and layer_perc <= 1, layer_perc
    #
    #             # loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
    #             # loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound
    #
    #             # loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
    #             # loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound
    #
    #             # loss_block += max(0, layer_perc - upper_bound) ** 2  # upper bound
    #             # loss_block += max(0, lower_bound - layer_perc) ** 2  # lower bound
    #
    #             channel_cost += Sci
    #             spatial_cost += Ssi
    #             total += t
    #
    #         layer_perc_spatial = tempS / tempT
    #         layer_perc_channel = tempC / tempT
    #         recoder.add('layer_perc_channel_' + str(i), 1 - layer_perc_channel.item())
    #         recoder.add('layer_perc_spatial_' + str(i), 1 - layer_perc_spatial.item())
    #         assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
    #         assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
    #         loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
    #         loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound
    #         loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
    #         loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound
    #
    #     perc_channel = channel_cost / total
    #     prec_spatial = spatial_cost / total
    #     assert perc_channel >= 0 and perc_channel <= 1, perc_channel
    #     assert prec_spatial >= 0 and prec_spatial <= 1, prec_spatial
    #     loss_block_channel /= len(meta['masks'])
    #     loss_block_spatial /= len(meta['masks'])
    #     loss_channel_network = (perc_channel - sparsity_target) ** 2
    #     loss_spatial_network = (prec_spatial - sparsity_target) ** 2
    #
    #     recoder.add('cs_upper_bound', upper_bound_channel)
    #     recoder.add('cs_lower_bound', lower_bound_channel)
    #     recoder.add('cs_cost_perc_channel', perc_channel.item())
    #     recoder.add('cs_cost_perc_spatial', prec_spatial.item())
    #     recoder.add('cs_loss_sp_channel', loss_block_channel.item())
    #     recoder.add('cs_loss_sp_spatial', loss_block_spatial.item())
    #     recoder.add('cs_loss_sp_network_channel', loss_channel_network.item())
    #     recoder.add('cs_loss_sp_network_spatial', loss_spatial_network.item())
    #     return loss_channel_network + loss_spatial_network + loss_block_spatial + loss_block_channel

    def forward_mix6(self, meta):
        sparsity_target = 1-math.sqrt(self.sparsity_target)
        # 通过控制progress的下界来调整最终约束的强度，当progress的最终值为0则比表明约束的范围是0——1，
        # 如果progress最终等于1，则约束的范围是theta
        # min(max(p, 0), 1) * my_pi  ~ [0,my_pi]
        # cos(min(max(p, 0), 1) * my_pi) ~ [1, cos(my_pi)]
        my_pi =  math.pi * 0.1845
        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * my_pi) ** 2
        # progress = 1
        # 1-sqrt(2)/2
        upper_bound_channel = (1 - progress * (1 -  sparsity_target))
        lower_bound_channel = progress * sparsity_target

        upper_bound_spatial = upper_bound_channel
        lower_bound_spatial = lower_bound_channel

        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), \
                      torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            spatial_masks = mask['spatial']
            channel_masks = mask['channel']

            tempT = torch.tensor(.0).to(device=meta['device'])
            tempS = torch.tensor(.0).to(device=meta['device'])
            tempC = torch.tensor(.0).to(device=meta['device'])
            for i in range(len(spatial_masks)):
                spatial_mask = spatial_masks[i]
                channel_mask = channel_masks[i]
                # 正常 -> spatial_mask.total_positions
                Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                       spatial_mask.total_positions * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size
                # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
                Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
                       channel_mask.total_channels * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size

                Si = spatial_mask.kernel_size * \
                     (channel_mask.channel_inte_state * spatial_mask.spatial_inte_state).sum()

                t = spatial_mask.total_positions * spatial_mask.flops_per_position
                tempT += t
                tempS += Ssi
                tempC += Sci

                cost += Si
                total += t

            layer_perc_spatial = tempS / tempT
            layer_perc_channel = tempC / tempT
            recoder.add('cha_layer_perc_' + str(i), 1 - layer_perc_channel.item())
            recoder.add('spa_layer_perc_' + str(i), 1 - layer_perc_spatial.item())
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
            loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
            loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound
            loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
            loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound


        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block_channel /= len(meta['masks'])
        loss_block_spatial /= len(meta['masks'])
        loss_network = (perc - 1 + self.sparsity_target) ** 2

        recoder.add('upper_bound', upper_bound_channel)
        recoder.add('lower_bound', lower_bound_channel)
        recoder.add('cost_perc', perc.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block_spatial + loss_block_channel

    def forward_mix7(self, meta):
        sparsity_target = 1 - math.sqrt(self.sparsity_target )
        # 通过控制progress的下界来调整最终约束的强度，当progress的最终值为0则比表明约束的范围是0——1，
        # 如果progress最终等于1，则约束的范围是theta
        # min(max(p, 0), 1) * my_pi  ~ [0,my_pi]
        # cos(min(max(p, 0), 1) * my_pi) ~ [1, cos(my_pi)]
        # my_pi =  math.pi * 0.1845
        p = meta['epoch'] / (0.33 * self.num_epochs)
        # progress = math.cos(min(max(p, 0), 1) * my_pi) ** 2
        progress1 = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        # progress = 1
        # 1-sqrt(2)/2
        upper_bound_channel = (1 - progress1 * (1 -  sparsity_target))
        lower_bound_channel = progress1 * sparsity_target

        upper_bound_spatial = BaseConfig.upper if upper_bound_channel > BaseConfig.upper else upper_bound_channel
        lower_bound_spatial = BaseConfig.lower if lower_bound_channel < BaseConfig.lower else lower_bound_channel

        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), \
                      torch.tensor(.0).to(device=meta['device'])

        count = 0
        for i, mask in enumerate(meta['masks']):
            spatial_masks = mask['spatial']
            channel_masks = mask['channel']

            tempT = torch.tensor(.0).to(device=meta['device'])
            tempS = torch.tensor(.0).to(device=meta['device'])

            for j in range(len(spatial_masks)):
                spatial_mask = spatial_masks[j]
                channel_mask = channel_masks[j]
                # 正常 -> spatial_mask.total_positions
                Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                       spatial_mask.total_positions * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size
                # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
                Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
                       channel_mask.total_channels * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size

                Si = spatial_mask.kernel_size * \
                     (channel_mask.channel_inte_state * spatial_mask.spatial_inte_state).sum()

                t = spatial_mask.total_positions * spatial_mask.flops_per_position
                tempT += t
                tempS += Ssi

                layer_perc_channel = Sci/t
                recoder.add('cha_layer_perc_' + str(count),layer_perc_channel.item())
                assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
                loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
                loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound

                cost += Si
                total += t
                count +=1

            layer_perc_spatial = tempS / tempT
            recoder.add('spa_layer_perc_' + str(i),layer_perc_spatial.item())
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
            loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound


        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block_channel /= count
        loss_block_spatial /= len(meta['masks'])
        # for循环中算的其实是保留率，因此在这里直接使用self.sparsity即可
        loss_network = (perc - self.sparsity_target) ** 2

        recoder.add('upper_bound_cha', upper_bound_channel)
        recoder.add('lower_bound_cha', lower_bound_channel)
        recoder.add('upper_bound_spa', upper_bound_spatial)
        recoder.add('lower_bound_spa', lower_bound_spatial)
        recoder.add('cost_perc',1 - perc.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block_spatial + loss_block_channel
    
    def forward_mix8(self, meta):
        sparsity_target = 1 - math.sqrt(self.sparsity_target )
        # 通过控制progress的下界来调整最终约束的强度，当progress的最终值为0则比表明约束的范围是0——1，
        # 如果progress最终等于1，则约束的范围是theta
        # min(max(p, 0), 1) * my_pi  ~ [0,my_pi]
        # cos(min(max(p, 0), 1) * my_pi) ~ [1, cos(my_pi)]
        # my_pi =  math.pi * 0.1845
        p = meta['epoch'] / (0.33 * self.num_epochs)
        # progress = math.cos(min(max(p, 0), 1) * my_pi) ** 2
        progress1 = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        # progress = 1
        # 1-sqrt(2)/2
        upper_bound_channel = (1 - progress1 * (1 -  sparsity_target))
        lower_bound_channel = progress1 * sparsity_target

        upper_bound_spatial = BaseConfig.upper if upper_bound_channel > BaseConfig.upper else upper_bound_channel
        lower_bound_spatial = BaseConfig.lower if lower_bound_channel < BaseConfig.lower else lower_bound_channel

        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), \
                      torch.tensor(.0).to(device=meta['device'])

        # count = 0
        for i, mask in enumerate(meta['masks']):
            spatial_masks = mask['spatial']
            channel_masks = mask['channel']

            tempT = torch.tensor(.0).to(device=meta['device'])
            tempS = torch.tensor(.0).to(device=meta['device'])
            tempC = torch.tensor(.0).to(device=meta['device'])

            for j in range(len(spatial_masks)):
                spatial_mask = spatial_masks[j]
                channel_mask = channel_masks[j]
                # 正常 -> spatial_mask.total_positions
                Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                       spatial_mask.total_positions * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size
                # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
                Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
                       channel_mask.total_channels * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size

                Si = spatial_mask.kernel_size * \
                     (channel_mask.channel_inte_state * spatial_mask.spatial_inte_state).sum()

                t = spatial_mask.total_positions * spatial_mask.flops_per_position
                tempT += t
                tempS += Ssi
                tempC += Sci

                

                cost += Si
                total += t
                # count +=1

            layer_perc_spatial = tempS / tempT
            recoder.add('spa_layer_perc_' + str(i),layer_perc_spatial.item())
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
            loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound


            layer_perc_channel = tempC / tempT
            recoder.add('cha_layer_perc_' + str(i),layer_perc_channel.item())
            assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
            loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
            loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound


        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block_channel /= len(meta['masks'])
        loss_block_spatial /= len(meta['masks'])
        # for循环中算的其实是保留率，因此在这里直接使用self.sparsity即可
        loss_network = (perc - self.sparsity_target) ** 2

        recoder.add('upper_bound_cha', upper_bound_channel)
        recoder.add('lower_bound_cha', lower_bound_channel)
        recoder.add('upper_bound_spa', upper_bound_spatial)
        recoder.add('lower_bound_spa', lower_bound_spatial)
        recoder.add('cost_perc',1 - perc.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block_spatial + loss_block_channel

    # 直接按固定值来控制各层的比例
    def forward_mix9(self, meta):
        sparsity_target = 1 - math.sqrt(self.sparsity_target )
        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress1 = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), \
                      torch.tensor(.0).to(device=meta['device'])

        count = 0
        for i, mask in enumerate(meta['masks']):
            spatial_masks = mask['spatial']
            channel_masks = mask['channel']

            tempT = torch.tensor(.0).to(device=meta['device'])
            tempS = torch.tensor(.0).to(device=meta['device'])

            for j in range(len(spatial_masks)):
                spatial_mask = spatial_masks[j]
                channel_mask = channel_masks[j]
                # 正常 -> spatial_mask.total_positions
                Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                       spatial_mask.total_positions * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size
                # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
                Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
                       channel_mask.total_channels * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size

                Si = spatial_mask.kernel_size * \
                     (channel_mask.channel_inte_state * spatial_mask.spatial_inte_state).sum()


                layer_perc_channel = Sci / Si
                recoder.add('cha_layer_perc_' + str(i),layer_perc_channel.item())
                assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
                loss_block_channel += (layer_perc_channel - sparsity_target)**2

                t = spatial_mask.total_positions * spatial_mask.flops_per_position
                tempT += t
                tempS += Ssi
                cost += Si
                total += t
                count +=1


            layer_perc_spatial = tempS / tempT
            recoder.add('spa_layer_perc_' + str(i),layer_perc_spatial.item())
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            loss_block_spatial += (layer_perc_spatial - sparsity_target)**2

        perc = cost / total
        assert perc >= 0 and perc <= 1, perc
        loss_block_channel /= count
        loss_block_spatial /= len(meta['masks'])
        # for循环中算的其实是保留率，因此在这里直接使用self.sparsity即可
        loss_network = (perc - self.sparsity_target) ** 2

        recoder.add('cost_perc',1 - perc.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block_spatial + loss_block_channel
    
    # 不对整体的施加约束，单独对空间和通道添加总约束（两者各负责一半）
    def forward_mix10(self, meta):
        sparsity_target = 1 - math.sqrt(self.sparsity_target )
        p = meta['epoch'] / (0.33 * self.num_epochs)
        progress1 = math.cos(min(max(p, 0), 1) * (math.pi / 2)) ** 2
        upper_bound_channel = (1 - progress1 * (1 -  sparsity_target))
        lower_bound_channel = progress1 * sparsity_target

        upper_bound_spatial = upper_bound_channel
        lower_bound_spatial = lower_bound_channel

        # upper_bound_spatial = BaseConfig.upper if upper_bound_channel > BaseConfig.upper else upper_bound_channel
        # lower_bound_spatial = BaseConfig.lower if lower_bound_channel < BaseConfig.lower else lower_bound_channel

        loss_block_channel = torch.tensor(.0).to(device=meta['device'])
        loss_block_spatial = torch.tensor(.0).to(device=meta['device'])
        cost_channel = torch.tensor(.0).to(device=meta['device'])
        cost_spatial = torch.tensor(.0).to(device=meta['device'])
        total = torch.tensor(.0).to(device=meta['device'])

        count = 0

        for i, mask in enumerate(meta['masks']):
            spatial_masks = mask['spatial']
            channel_masks = mask['channel']

            tempT = torch.tensor(.0).to(device=meta['device'])
            tempS = torch.tensor(.0).to(device=meta['device'])

            for j in range(len(spatial_masks)):
                spatial_mask = spatial_masks[j]
                channel_mask = channel_masks[j]
                # 正常 -> spatial_mask.total_positions
                Sci = ((channel_mask.total_channels - channel_mask.active_channels) * \
                       spatial_mask.total_positions * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size
                # 对激活的channel在整个batch上求平均 -> channel_mask.active_channels
                Ssi = ((spatial_mask.total_positions - spatial_mask.active_positions) * \
                       channel_mask.total_channels * \
                       spatial_mask.kernel_size)/spatial_mask.batch_size

                t = spatial_mask.total_positions * spatial_mask.flops_per_position
                tempT += t
                tempS += Ssi

                layer_perc_channel = Sci/t
                recoder.add('cha_layer_perc_' + str(count),layer_perc_channel.item())
                assert layer_perc_channel >= 0 and layer_perc_channel <= 1, layer_perc_channel
                loss_block_channel += max(0, layer_perc_channel - upper_bound_channel) ** 2  # upper bound
                loss_block_channel += max(0, lower_bound_channel - layer_perc_channel) ** 2  # lower bound

                cost_channel += Sci
                cost_spatial += Ssi
                total += t
                count +=1

            layer_perc_spatial = tempS / tempT
            recoder.add('spa_layer_perc_' + str(i),layer_perc_spatial.item())
            assert layer_perc_spatial >= 0 and layer_perc_spatial <= 1, layer_perc_spatial
            loss_block_spatial += max(0, layer_perc_spatial - upper_bound_spatial) ** 2  # upper bound
            loss_block_spatial += max(0, lower_bound_spatial - layer_perc_spatial) ** 2  # lower bound


        perc_channel = cost_channel / total
        perc_spatial = cost_spatial / total
        assert perc_channel >= 0 and perc_channel <= 1, perc_channel
        assert perc_spatial >= 0 and perc_spatial <= 1, perc_spatial
        loss_block_channel /= count
        loss_block_spatial /= len(meta['masks'])
        # for循环中算的其实是保留率，因此在这里直接使用self.sparsity即可
        loss_network = (perc_channel - sparsity_target) ** 2
        loss_network += (perc_spatial - sparsity_target) ** 2

        recoder.add('upper_bound_cha', upper_bound_channel)
        recoder.add('lower_bound_cha', lower_bound_channel)
        recoder.add('upper_bound_spa', upper_bound_spatial)
        recoder.add('lower_bound_spa', lower_bound_spatial)
        recoder.add('cost_perc_channel',perc_channel.item())
        recoder.add('cost_perc_spatial',perc_spatial.item())
        recoder.add('loss_sp_channel', loss_block_channel.item())
        recoder.add('loss_sp_spatial', loss_block_spatial.item())
        recoder.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block_spatial + loss_block_channel



    def forward(self, meta):
        if self.type == "spatial":
            return self.forward_spatial(meta)
        elif self.type == "channel":
            return self.forward_channel(meta)
        elif self.type == "mix1":
            return self.forward_mix(meta)
        elif self.type == "mix7":
            return self.forward_mix7(meta)
        elif self.type == "mix9":
            return self.forward_mix9(meta)
        elif self.type == "mix10":
            return self.forward_mix10(meta)
        else:
            raise NotImplementedError
