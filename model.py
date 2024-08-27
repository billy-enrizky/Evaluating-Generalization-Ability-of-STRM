import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
import numpy as np
from itertools import combinations 

from torch.autograd import Variable

import torchvision.models as models

NUM_SAMPLES=1

np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """
    Implements the Positional Encoding (PE) function to inject information about the relative or absolute position of 
    tokens in a sequence, which helps the model to understand the position of the words in the input sequence.

    Args:
        d_model (int): The dimension of the model. This is the size of the embedding vector.
        dropout (float): The dropout rate to be applied after adding the positional encodings.
        max_len (int, optional): The maximum length of the input sequences. Default is 5000.
        pe_scale_factor (float, optional): A scaling factor for the positional encodings. Default is 0.1.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied to the positional encodings.
        pe_scale_factor (float): Scaling factor for the positional encodings.
        pe (torch.Tensor): The positional encodings tensor of shape (1, max_len, d_model).

    Methods:
        forward(x):
            Adds positional encodings to the input tensor and applies dropout.

    Example:
        >>> pe = PositionalEncoding(d_model=512, dropout=0.1)
        >>> input_tensor = torch.randn(10, 32, 512)  # (sequence_length, batch_size, d_model)
        >>> output_tensor = pe(input_tensor)

    Raises:
        ValueError: If the dimensions of the input tensor do not match the expected dimensions.
    """

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor

        # Compute the positional encodings once in log space.
        # pe is of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # position is of shape (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # div_term is of shape (d_model // 2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The tensor with positional encodings added, of the same shape as the input tensor.

        Raises:
            ValueError: If the dimensions of the input tensor do not match the expected dimensions.
        """
        if x.dim() != 3 or x.size(2) != self.pe.size(2):
            raise ValueError('Expected input tensor of shape (batch_size, sequence_length, d_model) '
                             'but got {}'.format(x.shape))
        
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class DistanceLoss(nn.Module):
    """
    Compute the Query-class similarity on the patch-enriched features.

    Args:
        args (Namespace): The arguments containing hyperparameters and configurations.
        temporal_set_size (int, optional): The size of the temporal set. Default is 3.

    Attributes:
        args (Namespace): The arguments containing hyperparameters and configurations.
        temporal_set_size (int): The size of the temporal set.
        dropout (nn.Dropout): Dropout layer applied to the input features.
        tuples (list of torch.Tensor): List of all ordered tuples corresponding to the temporal set size.
        tuples_len (int): The length of the tuples list.
        clsW (nn.Linear): Linear layer for classifying the support and query embeddings.
        relu (nn.ReLU): ReLU activation function.

    Methods:
        forward(support_set, support_labels, queries):
            Computes the distance loss between support set and queries.

        _extract_class_indices(labels, which_class):
            Extracts the indices of elements which have the specified label.

    Example:
        >>> args = Namespace(seq_len=8, trans_linear_in_dim=2048, way=5)
        >>> distance_loss = DistanceLoss(args)
        >>> support_set = torch.randn(25, 8, 2048)
        >>> support_labels = torch.randint(0, 5, (25,))
        >>> queries = torch.randn(20, 8, 2048)
        >>> output = distance_loss(support_set, support_labels, queries)

    Raises:
        ValueError: If `degrees` is a single number and is not positive, or if it is a sequence and its length is not 2.
    """

    def __init__(self, args, temporal_set_size=3):
        super(DistanceLoss, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.dropout = nn.Dropout(p=0.1)

        # Generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cuda':
            self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        else:
            self.tuples = [torch.tensor(comb) for comb in frame_combinations]
        self.tuples_len = len(self.tuples)  # 28 for temporal set size 2

        # Define the classification linear layer and ReLU activation function
        self.clsW = nn.Linear(self.args.trans_linear_in_dim * self.temporal_set_size, self.args.trans_linear_in_dim // 2)
        self.relu = torch.nn.ReLU()

    def forward(self, support_set, support_labels, queries):
        """
        Args:
            support_set (torch.Tensor): Support set of shape (25, 8, 2048).
            support_labels (torch.Tensor): Labels for the support set of shape (25,).
            queries (torch.Tensor): Query set of shape (20, 8, 2048).

        Returns:
            dict: Dictionary containing the logits tensor of shape (20, 5).
        """
        n_queries = queries.shape[0]  # 20
        n_support = support_set.shape[0]  # 25

        # Add a dropout before creating tuples
        support_set = self.dropout(support_set)  # 25 x 8 x 2048
        queries = self.dropout(queries)  # 20 x 8 x 2048

        # Construct new queries and support set made of tuples of images after positional encoding
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2).to(device)  # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2)  # 20 x 28 x 4096
        support_labels = support_labels.to(device)
        unique_labels = torch.unique(support_labels)  # 5

        query_embed = self.clsW(queries.view(-1, self.args.trans_linear_in_dim * self.temporal_set_size))  # 560 x 1024

        # Add relu after clsW
        query_embed = self.relu(query_embed)  # 560 x 1024        

        # Initialize tensor to hold distances between every support tuple and every target tuple. Shape: (20, 5)
        dist_all = torch.zeros(n_queries, self.args.way)  # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            class_k = torch.index_select(support_set, 0, self._extract_class_indices(support_labels, c))  # 5 x 28 x 4096

            # Reshaping the selected keys
            class_k = class_k.view(-1, self.args.trans_linear_in_dim * self.temporal_set_size)  # 140 x 4096

            # Get the support set projection from the current class
            support_embed = self.clsW(class_k.to(queries.device))  # 140 x 1024

            # Add relu after clsW
            support_embed = self.relu(support_embed)  # 140 x 1024

            # Calculate p-norm distance between the query embedding and the support set embedding
            distmat = torch.cdist(query_embed, support_embed)  # 560 x 140

            # Get the minimum distance for each of the 560 queries
            min_dist = distmat.min(dim=1)[0].reshape(n_queries, self.tuples_len)  # 20 x 28

            # Average across the 28 tuples
            query_dist = min_dist.mean(dim=1)  # 20

            # Make it negative as this has to be reduced
            distance = -1.0 * query_dist
            c_idx = c.long()
            dist_all[:, c_idx] = distance  # Insert into the required location

        return_dict = {'logits': dist_all}
        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.

        Args:
            labels (torch.Tensor): Labels of the context set.
            which_class (int): Label for which indices are extracted.

        Returns:
            torch.Tensor: Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class TemporalCrossTransformer(nn.Module):
    """
    A module to compute the temporal cross-transformer for the few-shot learning task.
    """
    def __init__(self, args, temporal_set_size=3):
        """
        Initializes the TemporalCrossTransformer.

        Args:
            args: ArgumentParser object containing various settings.
            temporal_set_size (int): The size of the temporal set, default is 3.
        """
        super(TemporalCrossTransformer, self).__init__()
        
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # Generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == 'cuda':
            self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        else:
            self.tuples = [torch.tensor(comb) for comb in frame_combinations]
        self.tuples_len = len(self.tuples) # 28 for temporal_set_size=3

    def forward(self, support_set, support_labels, queries):
        """
        Forward pass of the TemporalCrossTransformer.

        Args:
            support_set (torch.Tensor): Support set of shape (n_support, seq_len, feature_dim).
            support_labels (torch.Tensor): Labels for the support set of shape (n_support).
            queries (torch.Tensor): Query set of shape (n_queries, seq_len, feature_dim).

        Returns:
            dict: Dictionary containing logits of shape (n_queries, n_classes).
        """
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # Apply positional encoding to support set and queries
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # Construct new queries and support set made of tuples of images after positional encoding
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # Apply linear maps and layer normalization
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        mh_support_set_ks = self.norm_k(support_set_ks).to(device)
        mh_queries_ks = self.norm_k(queries_ks).to(device)
        support_labels = support_labels.to(device)
        mh_support_set_vs = support_set_vs.to(device)
        mh_queries_vs = queries_vs.to(device)
        
        unique_labels = torch.unique(support_labels)

        # Initialize tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way)

        for label_idx, c in enumerate(unique_labels):
            # Select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            # Compute class scores
            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(self.args.trans_linear_out_dim)
            class_scores = class_scores.permute(0, 2, 1, 3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0, 2, 1, 3)

            # Get query specific class prototype
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1).to(device)

            # Calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2, -1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance

        return_dict = {'logits': all_distances_tensor}
        
        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.

        Args:
            labels (torch.Tensor): Labels of the context set.
            which_class (int): Label for which indices are extracted.

        Returns:
            torch.Tensor: Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # Binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # Indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # Reshape to be a 1D vector


class Token_Perceptron(torch.nn.Module):
    '''
        2-layer Token MLP
    '''
    def __init__(self, in_dim):
        super(Token_Perceptron, self).__init__()
        # in_dim 8
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):

        # Applying the linear layer on the input
        output = self.inp_fc(x) # B x 2048 x 8

        # Apply the relu non-linearity
        output = self.relu(output) # B x 2048 x 8

        # Apply the 2nd linear layer
        output = self.out_fc(output)
        
        return output

class Bottleneck_Perceptron_2_layer(torch.nn.Module):
    '''
        2-layer Bottleneck MLP
    '''
    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_2_layer, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.out_fc(output)
        
        return output 

class Bottleneck_Perceptron_3_layer_res(torch.nn.Module):
    '''
        3-layer Bottleneck MLP followed by a residual layer
    '''
    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_3_layer_res, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim//2)
        self.hid_fc = nn.Linear(in_dim//2, in_dim//2)
        self.out_fc = nn.Linear(in_dim//2, in_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.relu(self.hid_fc(output)) 
        output = self.out_fc(output)
        
        return output + x # Residual output

class Self_Attn_Bot(nn.Module):
    """ Self attention Layer
        Attention-based frame enrichment
    """
    def __init__(self,in_dim, seq_len):
        super(Self_Attn_Bot,self).__init__()
        self.chanel_in = in_dim # 2048
        
        # Using Linear projections for Key, Query and Value vectors
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)

        self.softmax  = nn.Softmax(dim=-1) #
        self.gamma = nn.Parameter(torch.zeros(1))
        self.Bot_MLP = Bottleneck_Perceptron_3_layer_res(in_dim)
        max_len = int(seq_len * 1.5)
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):

        """
            inputs :
                x : input feature maps( B X C X W )[B x 16 x 2048]
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width)
        """

        # Add a position embedding to the 16 patches
        x = self.pe(x) # B x 16 x 2048

        m_batchsize,C,width = x.size() # m = 200/160, C = 2048, width = 16

        # Save residual for later use
        residual = x # B x 16 x 2048

        # Perform query projection
        proj_query  = self.query_proj(x) # B x 16 x 2048

        # Perform Key projection
        proj_key = self.key_proj(x).permute(0, 2, 1) # B x 2048  x 16

        energy = torch.bmm(proj_query,proj_key) # transpose check B x 16 x 16
        attention = self.softmax(energy) #  B x 16 x 16

        # Get the entire value in 2048 dimension 
        proj_value = self.value_conv(x).permute(0, 2, 1) # B x 2048 x 16

        # Element-wise multiplication of projected value and attention: shape is x B x C x N: 1 x 2048 x 8
        out = torch.bmm(proj_value,attention.permute(0,2,1)) # B x 2048 x 16

        # Reshaping before passing through MLP
        out = out.permute(0, 2, 1) # B x 16 x 2048

        # Passing via gamma attention
        out = self.gamma*out + residual # B x 16 x 2048

        # Pass it via a 3-layer Bottleneck MLP with Residual Layer defined within MLP
        out = self.Bot_MLP(out)  # B x 16 x 2048

        return out

class MLP_Mix_Enrich(nn.Module):
    """ 
        Pure Token-Bottleneck MLP-based enriching features mechanism
    """
    def __init__(self,in_dim, seq_len):
        super(MLP_Mix_Enrich,self).__init__()
        # in_dim = 2048
        self.Tok_MLP = Token_Perceptron(seq_len) # seq_len = 8 frames
        self.Bot_MLP = Bottleneck_Perceptron_2_layer(in_dim)

        max_len = int(seq_len * 1.5) # seq_len = 8
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W ) # B(25/20) x 8 x 2048
            returns :
                out : self MLP-enriched value + input feature 
        """

        # Add a position embedding to the 8 frames
        x = self.pe(x) # B x 8 x 2048

        # Store the residual for use later
        residual1 = x # B x 8 x 2048

        # Pass it via a 2-layer Token MLP followed by Residual Layer
        # Permuted before passing into the MLP: B x 2048 x 8 
        out = self.Tok_MLP(x.permute(0, 2, 1)).permute(0, 2, 1) + residual1 # B x 8 x 2048

        # Storing a residual 
        residual2 = out # B x 8 x 2048
        
        # Pass it via 2-layer Bottleneck MLP defined on Channel(2048) features
        out = self.Bot_MLP(out) + residual2 # B x 8 x 2048

        return out

class CNN_STRM(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args):
        super(CNN_STRM, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.num_patches = 16

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

        # New-distance metric for post patch-level enriched features
        self.new_dist_loss_post_pat = [DistanceLoss(args, s) for s in args.temp_set]

        # Linear-based patch-level attention over the 16 patches
        self.attn_pat = Self_Attn_Bot(self.args.trans_linear_in_dim, self.num_patches)

        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(self.args.trans_linear_in_dim, self.args.seq_len)

    def forward(self, context_images, context_labels, target_images):

        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        context_features = self.resnet(context_images) # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images) # 160 x 2048 x 7 x 7

        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features) # 200 x 2048 x 4 x 4
        target_features = self.adap_max(target_features) # 160 x 2048 x 4 x 4

        # Reshape before averaging across all the patches
        context_features = context_features.reshape(-1, self.args.trans_linear_in_dim, self.num_patches) # 200 x 2048 x 16
        target_features = target_features.reshape(-1, self.args.trans_linear_in_dim, self.num_patches) # 160 x 2048 x 16       

        # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 2048
        target_features = target_features.permute(0, 2, 1) # 160 x 16 x 2048

        # Performing self-attention across the 16 patches
        context_features = self.attn_pat(context_features) # 200 x 16 x 2048 
        target_features = self.attn_pat(target_features) # 160 x 16 x 2048

        # Average across the patches 
        context_features = torch.mean(context_features, dim = 1) # 200 x 2048
        target_features = torch.mean(target_features, dim = 1) # 160 x 2048

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 20 x 8 x 2048

        # Compute logits using the new loss before applying frame-level attention
        all_logits_post_pat = [n(context_features, context_labels, target_features)['logits'] for n in self.new_dist_loss_post_pat]
        all_logits_post_pat = torch.stack(all_logits_post_pat, dim=-1) # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]

        # Combing the patch and frame-level logits
        sample_logits_post_pat = all_logits_post_pat
        sample_logits_post_pat = torch.mean(sample_logits_post_pat, dim=[-1]) # 20 x 5

        # Perform self-attention across the 8 frames
        context_features_fr = self.fr_enrich(context_features) # 25 x 8 x 2048
        target_features_fr = self.fr_enrich(target_features) # 20 x 8 x 2048

        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        # Frame-level logits
        all_logits_fr = [t(context_features_fr, context_labels, target_features_fr)['logits'] for t in self.transformers]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1) # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1]) # 20 x 5

        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]), 
                    'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)
            self.new_dist_loss_post_pat = [n.cuda(0) for n in self.new_dist_loss_post_pat]

            self.attn_pat.cuda(0)
            self.attn_pat = torch.nn.DataParallel(self.attn_pat, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.fr_enrich.cuda(0)
            self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=[i for i in range(0, self.args.num_gpus)])

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
            self.temp_set = [2,3]
    args = ArgsObject()
    torch.manual_seed(0)
    model = CNN_STRM(args).to(device)
    support_imgs = torch.rand(args.way * args.shot * args.seq_len,3, args.img_size, args.img_size)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len ,3, args.img_size, args.img_size)
    support_labels = torch.tensor([0,1,2,3,4])

    out = model(support_imgs, support_labels, target_imgs)

    print("STRM returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(out['logits'].shape))
