import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *
from model import CNN_STRM

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SiameseNetwork(nn.Module):
    def __init__(self, backbone):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone

    def forward(self, input1, input2):
        # Pass both inputs through the same CNN backbone
        output1 = self.backbone(input1)
        output2 = self.backbone(input2)
        
        # Ensure we extract the correct features from CNN_STRM's output
        if isinstance(output1, dict) and 'features' in output1:
            output1 = output1['features']
        if isinstance(output2, dict) and 'features' in output2:
            output2 = output2['features']

        # Compute the similarity between the two outputs using L2 distance
        distance = F.pairwise_distance(output1, output2, p=2)
        
        return distance

# Example of how to initialize and use the SiameseNetwork

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
        self.temp_set = [2,3]

args = ArgsObject()
torch.manual_seed(0)

# Initialize the CNN_STRM model
model = CNN_STRM(args).to(device)

# Generate example data
support_imgs = torch.rand(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size).to(device)
target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size).to(device)
support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)

# Initialize and test the Siamese Network using CNN_STRM as the backbone
cnn_strm_backbone = CNN_STRM(args).to(device)
siamese_network = SiameseNetwork(cnn_strm_backbone).to(device)

# Example forward pass through the Siamese Network
distance = siamese_network(support_imgs, target_imgs)
print(f"Distance between support and target images: {distance}")
