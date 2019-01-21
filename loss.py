import torch.nn as nn
import torch

class MaskedMSELoss(nn.modules.loss._Loss):
    def __init__(self, is_gpu_avaiable, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MaskedMSELoss, self).__init__(size_average, reduce, reduction)
        self.is_gpu_available = is_gpu_avaiable

    def forward(self, pred, label, mask, is_sum=False):
        total_sample_count = len(mask)
        mse_sum = 0
        for index, batch_pred in enumerate(pred):
            mse_sum += nn.functional.mse_loss(batch_pred[:mask[index]], label[index][:mask[index]])
        if is_sum:
            return mse_sum
        return mse_sum / total_sample_count

class MaskedRMSELoss(nn.modules.loss._Loss):
    def __init__(self, is_gpu_avaiable, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MaskedRMSELoss, self).__init__(size_average, reduce, reduction)
        self.is_gpu_available = is_gpu_avaiable

    def forward(self, pred, label, mask):
        total_sample_count = len(mask)
        rmse_sum = 0
        for index, batch_pred in enumerate(pred):
            rmse_sum += torch.sqrt(nn.functional.mse_loss(batch_pred[:mask[index]], label[index][:mask[index]]))
        return rmse_sum / total_sample_count

class MaskedNormalizedMSELoss(nn.modules.loss._Loss):
    def __init__(self, is_gpu_avaiable, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MaskedNormalizedMSELoss, self).__init__(size_average, reduce, reduction)
        self.is_gpu_available = is_gpu_avaiable

    def forward(self, pred, label, mask, is_sum=False):
        total_sample_count = len(mask)
        mse_sum = 0
        for index, batch_pred in enumerate(pred):
            mse_sum += torch.mean(
                nn.functional.mse_loss(batch_pred[:mask[index]], label[index][:mask[index]], reduce=False)/
                (torch.pow(label[index][:mask[index]], 2) + 1e-2))
        if is_sum:
            return mse_sum
        return mse_sum / total_sample_count

class MaskedNormalizedRMSELoss(nn.modules.loss._Loss):
    def __init__(self, is_gpu_avaiable, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(MaskedNormalizedRMSELoss, self).__init__(size_average, reduce, reduction)
        self.is_gpu_available = is_gpu_avaiable

    def forward(self, pred, label, mask, is_sum=False):
        total_sample_count = len(mask)
        # if self.is_gpu_available:
        #     total_sample_count = total_sample_count.cuda()
        rmse_sum = 0
        for index, batch_pred in enumerate(pred):
            if is_sum:
                rmse_sum += torch.sum(torch.sqrt(
                    nn.functional.mse_loss(batch_pred[:mask[index]], label[index][:mask[index]], reduce=False)/
                    (torch.pow(label[index][:mask[index]], 2) + 1e-2)))
            else:
                rmse_sum += torch.sqrt(torch.mean(
                    nn.functional.mse_loss(batch_pred[:mask[index]], label[index][:mask[index]], reduce=False)/
                    (torch.pow(label[index][:mask[index]], 2) + 1e-2)))
        if is_sum:
            return rmse_sum
        return rmse_sum / total_sample_count

class MaskedMAEAndMAPELoss(nn.modules.loss._Loss):
    def __init__(self, is_gpu_avaiable):
        super(MaskedMAEAndMAPELoss, self).__init__()
        self.is_gpu_available = is_gpu_avaiable

    def forward(self, pred, label, mask):
        total_sample_count = len(mask)
        total_sample_count_mape = len(mask)
        mae_sum = 0
        mape_sum = 0
        for index, batch_pred in enumerate(pred):
            abs_dif = torch.abs(batch_pred[:mask[index]] - label[index][:mask[index]])
            mae_sum += torch.mean(abs_dif)
            less_than_one_mask = label[index][:mask[index]] > 1
            data_number = torch.sum(less_than_one_mask).item()
            if data_number == 0:
                total_sample_count_mape -= 1
            else:
                mape_sum += torch.sum((abs_dif / (label[index][:mask[index]] + 1e-3)) * less_than_one_mask.float()) \
                            / data_number
        return mae_sum, mape_sum, total_sample_count_mape