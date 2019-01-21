import torch.nn as nn
import torch

class MaskedMSELoss(nn.modules.loss._Loss):
    def __init__(self, is_gpu_avaiable, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MaskedMSELoss, self).__init__(size_average, reduce, reduction)
        self.is_gpu_available = is_gpu_avaiable

    def forward(self, pred, label, mask, is_sum=False):
        total_sample_count = len(mask)
        # if self.is_gpu_available:
        #     total_sample_count = total_sample_count.cuda()
        rmse_sum = 0
        for index, batch_pred in enumerate(pred):
            rmse_sum += torch.sqrt(nn.functional.mse_loss(pred[:mask[index]], label[:mask[index]]))
        if is_sum:
            return rmse_sum
        return rmse_sum / total_sample_count