import torch
import gpytorch

import torch.nn as nn

from torch.nn import functional as F

# https://discuss.pytorch.org/t/causal-2d-convolution/131720
class CausalConv2d(nn.Module):
    def __init__(self, args, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3)):
        super(CausalConv2d, self).__init__()
        self.pad = nn.ZeroPad2d((0, 0, kernel_size[0] - 1, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        x = self.conv(x)
        x = F.relu(x)
        return x

class TSFeatureExtractor(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.MTL_method = args.MTL_method
        self.num_heads = args.num_heads
        self.num_tran_layer = args.num_tran_layer
        self.emb_dim = args.emb_dim
        self.temporal_conv_length_list = args.temporal_conv_length_list
        self.num_variants = 15 # args.num_variants
        self.dkl_layers = args.dkl_layers
        self.ignore_cnn = args.ignore_cnn
        self.layernorm = args.layernorm
        self.ignore_transformer = args.ignore_transformer

        # Feature Extractor
        self.linear0 = torch.nn.Linear(1, self.emb_dim)
        self.linear1 = torch.nn.Linear(1, self.emb_dim)
        self.linear2 = torch.nn.Linear(1, self.emb_dim)
        self.linear3 = torch.nn.Linear(1, self.emb_dim)
        self.linear4 = torch.nn.Linear(1, self.emb_dim)
        self.linear5 = torch.nn.Linear(1, self.emb_dim)
        self.variants = torch.nn.Linear(self.num_variants, self.emb_dim)
        self.linear6 = torch.nn.Linear(8 * self.emb_dim, 4 * self.emb_dim)
        self.linear7 = nn.Sequential(*[nn.Identity()] + [nn.Linear(4 * self.emb_dim, 4 * self.emb_dim), nn.ReLU()] * self.dkl_layers)
        self.linear8 = torch.nn.Linear(4 * self.emb_dim, self.emb_dim)
        # Linear layers for processing concatenated features

        self.holiday_lookup = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.emb_dim)
        self.variants_lookup = torch.nn.Embedding(num_embeddings=self.num_variants, embedding_dim=self.emb_dim)
        
        # Channel Transformer Encoder ------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=self.num_heads, dim_feedforward=8, activation="gelu", dropout=0)
        self.channel_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_tran_layer)
        # ----------------------------------------------------------------
        
        self.channel_convs = [CausalConv2d(args, in_channels=1, out_channels=1, kernel_size=(conv_length, 1)).to(device)
                                         for conv_length in self.temporal_conv_length_list]
        self.channel_convs = nn.ModuleList(self.channel_convs)

        # MTL - uncertainty
        self.logsigma = nn.Parameter(torch.FloatTensor([-1 * args.logsigma] * 3)) # 3 = num_task
        
        # plot
        self.plot_attn = args.plot_attn
        
        self.is_train = None
        
    def forward(self, x):
        self.is_train = (x.shape[1] == 860)
        # C, 1, L, num_feature
        inoculation_feature = self.linear0(x[..., 0:1])
        temperature_feature = self.linear1(x[..., 1:2])
        humidity_feature = self.linear2(x[...,2:3])
        precipitation_feature = self.linear3(x[...,3:4])
        stringency_feature = self.linear4(x[...,4:5])
        time_feature = self.linear5(x[...,-1:])
        holiday_feature = torch.mean(self.holiday_lookup(x[..., 5:6].to(torch.int)), axis=2)
        variants_feature = torch.mean(self.variants_lookup(x[..., 6: -1].to(torch.int)), axis=2)
        
        # Concatenate every feature map
        cat_x = torch.cat([inoculation_feature, temperature_feature, humidity_feature, precipitation_feature, \
                        stringency_feature, holiday_feature, variants_feature, time_feature], axis=2)
        
        x = F.relu(self.linear6(cat_x))
        x = self.linear7(x)
        x = F.relu(self.linear8(x)) # C, 1, L, K
        # x.shape [C, L, K]
        
        if not self.ignore_transformer:
            # temporal conv
            x_ = x.unsqueeze(1) # x.shape [C, 1, L, K]
            channel_feature_list = [x_.permute(2, 1, 0, 3)] # channel_feature_list [L, 1, C, K] * num_conv
            if not self.ignore_cnn:
                # (L, 1, C, K) * num_conv
                channel_feature_list = [channel_conv(x_).permute(2, 1, 0, 3) for channel_conv in self.channel_convs]
            
            # Using whole encoder layer --------------------
            channel_feature_list = [self.channel_transformer_encoder(channel_feature.squeeze()) for channel_feature in channel_feature_list]
            attention_list = [self.channel_transformer_encoder.layers[0].self_attn(channel_feature.squeeze().permute(1,0,2), 
                                                                                   channel_feature.squeeze().permute(1,0,2), 
                                                                                   channel_feature.squeeze().permute(1,0,2)) for channel_feature in channel_feature_list]
            #-----------------------------------------------
            
            # non-zero mean --------------------------------
            # ft.shape = [num_conv, L, C, K]
            ft = torch.stack(channel_feature_list)
            # channel_feature.shape = [L, C, K]
            channel_feature = ft.sum(dim=0)
            # debugging step : If C=1 so that channel_feature.shape = [L, K]
            if len(channel_feature.shape) == 2:
                channel_feature = channel_feature.unsqueeze(1)
            # ----------------------------------------------

            x = channel_feature.permute(1,0,2) # (C, L, K)

        if self.plot_attn or self.is_train:
            if self.MTL_method == 'uncert':
                return x, self.logsigma, attention_list
            else:
                return x, attention_list
        else:
            if self.MTL_method == 'uncert':
                return x, self.logsigma
            else:
                return x

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, args, kernel='SM+P'):
        self.train_y = train_y
        self.args = args
        super(ExactGPModel, self).__init__(train_x, self.train_y, likelihood)
        
        self.dkl = args.dkl
        self.emb_dim = args.emb_dim
        self.model_type = args.model_type
        self.num_variants = 15 # args.num_variants
        self.dkl_layers = args.dkl_layers
        self.ard_num_dims = self.emb_dim if self.dkl else 35
        self.model_type = args.model_type
        self.mix_concat = args.mix_concat
        self.kernel_name = kernel
        print(f"GP defined with {self.kernel_name} kernel.")
        # Feature Extractor Linear Layers
        self.linear0 = nn.Linear(1, self.emb_dim)
        self.linear1 = nn.Linear(1, self.emb_dim)
        self.linear2 = nn.Linear(1, self.emb_dim)
        self.linear3 = nn.Linear(1, self.emb_dim)
        self.linear4 = nn.Linear(1, self.emb_dim)
        self.linear5 = nn.Linear(1, self.emb_dim)
        self.variants = nn.Linear(self.num_variants, self.emb_dim)

        # Initialize LayerNorm layers for linear transformations
        self.norm = nn.LayerNorm(2 * self.emb_dim)

        # Embeddings for categorical features
        self.holiday_lookup = nn.Embedding(num_embeddings=2, embedding_dim=self.emb_dim)
        # self.variants = nn.Linear(self.num_variants, self.emb_dim)
        self.variants_lookup = nn.Embedding(num_embeddings=self.num_variants, embedding_dim=self.emb_dim)

        # Linear layers for processing concatenated features
        self.linear6 = nn.Linear(7 * self.emb_dim, 4 * self.emb_dim)
        self.linear7 = nn.Sequential(*[nn.Identity()] + [nn.Linear(4 * self.emb_dim, 4 * self.emb_dim), nn.ReLU()] * self.dkl_layers)
        self.linear8 = nn.Linear(4 * self.emb_dim, self.emb_dim)
        self.linear9 = nn.Linear(2 * self.emb_dim, self.emb_dim)

        # GP components
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.MultitaskMean(self.mean_module, num_tasks=2)

        # Covariance kernel selection based on args
        if self.kernel_name == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() )

        elif self.kernel_name == 'SpectralMixture':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims))

        elif self.kernel_name == 'SML':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.LinearKernel())
            
        elif self.kernel_name == 'SM+L':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)+gpytorch.kernels.LinearKernel())
            
        elif self.kernel_name == 'SMP':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims))

        elif self.kernel_name == 'SM+P':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)+gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims))

        elif self.kernel_name == 'SMPL':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims)*gpytorch.kernels.LinearKernel())

        elif self.kernel_name == 'SMP+L':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims) + gpytorch.kernels.LinearKernel())

        elif self.kernel_name == 'SM+P+L':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims) + gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims) + gpytorch.kernels.LinearKernel())

        elif self.kernel_name == 'SML+P':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.LinearKernel() + gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims))

        self.covar_module = gpytorch.kernels.MultitaskKernel(self.covar_module, num_tasks=2, rank=args.rank)
        self.scale_bound = False
    
    def forward(self, x):
        if self.dkl:
            inoculation_feature = self.linear0(x[:,0].unsqueeze(1))
            temperature_feature = self.linear1(x[:,1].unsqueeze(1))
            humidity_feature = self.linear2(x[:,2].unsqueeze(1))
            precipitation_feature = self.linear3(x[:,3].unsqueeze(1))
            stringency_feature = self.linear4(x[:,4].unsqueeze(1))
            time_feature = self.linear5(x[:,-1].unsqueeze(1))
            
            holiday_feature = self.holiday_lookup(x[:, 5].to(torch.int))
            variants_feature = self.variants_lookup(x[:, 6].to(torch.int))
            
            # Concatenate every feature maps
            feature_list = [inoculation_feature, temperature_feature, humidity_feature, precipitation_feature, stringency_feature, holiday_feature, variants_feature]#, time_feature,]
            x = torch.cat(feature_list, axis=1)
            x = F.relu(self.linear6(x))
            x = self.linear7(x)
            x = self.linear8(x)
                        
            if not self.args.ignore_transformer:
                # concat private dkl, shared dkl 
                if x.shape[0] == self.shared_emb.shape[0] * 2:
                    self.shared_emb = torch.cat([self.shared_emb, self.shared_emb], axis=0)
                x = torch.cat([x, self.shared_emb], axis=1)
                x = self.norm(x)
                x = self.linear9(x)
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        pred = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        return pred