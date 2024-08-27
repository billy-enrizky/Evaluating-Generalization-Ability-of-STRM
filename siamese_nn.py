import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *
from model import CNN_STRM
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SiameseNetwork(nn.Module):
    def __init__(self, backbone):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone

    def forward(self, support_images, context_labels, target_images, target_labels=torch.empty(0)):
        # Get the feature dictionaries for support and target images
        support_output = self.backbone(support_images, context_labels, target_images)
        if target_labels.numel() == 0:
            target_output = self.backbone(target_images, context_labels, target_images)
        else:
            target_output = self.backbone(target_images, target_labels, target_images)
        # Extract the features needed for distance computation from the output dictionaries
        support_features = support_output['logits_post_pat']  # Adjust based on what features you want to use
        target_features = target_output['logits_post_pat']  # Adjust accordingly

        # Compute the similarity between the two outputs using L2 distance
        distance = F.pairwise_distance(support_features, target_features, p=2)
        
        return distance

# Example of how to initialize and use the SiameseNetwork
if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128
            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.trans_dropout = 0.1
            self.seq_len = 8 
            self.img_size = 84
            self.method = "resnet50"
            self.num_gpus = 1
            self.temp_set = [2, 3]

    args = ArgsObject()
    torch.manual_seed(0)

    # Initialize the CNN_STRM model
    cnn_strm_backbone = CNN_STRM(args).to(device)
    siamese_network = SiameseNetwork(cnn_strm_backbone).to(device)

    # Generate example data
    support_imgs = torch.rand(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)

    # Example forward pass through the Siamese Network
    distance = siamese_network(support_imgs, support_labels, target_imgs)
    print(f"Distance between support and target images: {distance}")

