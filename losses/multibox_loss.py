import torch.nn as nn
def hard_negative_mining(loss, labels, neg_pos_ratio):

    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=0, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    # print("loss",loss.shape)
    # print(num_pos,neg_pos_ratio,num_neg)
    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=0, descending=True)
    _, orders = indexes.sort(dim=0)
    neg_mask = orders < num_neg
    return pos_mask|neg_mask
class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations):

        num_classes = 3
        corner_form_priors=center_form_to_corner_form(priors).to(device)
        gt_locations,labels=assign_priors(gt_locations, labels, corner_form_priors,0.7)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=1)[:, 0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)
            num_selected = mask.sum().item()
        if num_selected ==0:
                # print("yessss")
                classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels, size_average=False)
                # pos_mask = labels > 0
                predicted_locations = predicted_locations.reshape(-1, 4)
                gt_locations = gt_locations.reshape(-1, 4)
                smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
                num_pos = gt_locations.size(0)
                # print("classification_loss",classification_loss)
                return smooth_l1_loss/num_pos, classification_loss/num_pos
        else:
                confidence = confidence[mask, :]
                classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
                pos_mask = labels > 0
                predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
                gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
                smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
                num_pos = gt_locations.size(0)
                return smooth_l1_loss/num_pos, classification_loss/num_pos