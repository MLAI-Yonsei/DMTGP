import torch
import gpytorch

import torch.nn as nn

from torch.nn import functional as F
from gpytorch.mlls import SumMarginalLogLikelihood

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, args):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.dkl = False
        self.emb_dim = args.emb_dim
        self.model_type = args.model_type
        self.num_variants = args.num_variants

        self.ard_num_dims = 2 * self.emb_dim if self.dkl else 35
        self.model_type = args.model_type

        # define mean, covar modules
        self.mean_module = gpytorch.means.ConstantMean()
        if self.model_type in ['MTGP', 'ours', 'full']:
            self.mean_module = gpytorch.means.MultitaskMean(
                self.mean_module, num_tasks=2
            )

        if args.kernel_name == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() )

        elif args.kernel_name == 'SpectralMixture':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims))

        elif args.kernel_name == 'SML':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.LinearKernel())
            
        elif args.kernel_name == 'SM+L':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)+gpytorch.kernels.LinearKernel())
            
        elif args.kernel_name == 'SMP':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims))

        elif args.kernel_name == 'SM+P':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)+gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims))

        elif args.kernel_name == 'SMPL':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims)*gpytorch.kernels.LinearKernel())

        elif args.kernel_name == 'SMP+L':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims) + gpytorch.kernels.LinearKernel())

        elif args.kernel_name == 'SM+P+L':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims) + gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims) + gpytorch.kernels.LinearKernel())

        elif args.kernel_name == 'SML+P':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=self.ard_num_dims)*gpytorch.kernels.LinearKernel() + gpytorch.kernels.PeriodicKernel(ard_num_dims=self.ard_num_dims))

        if args.model_type in ['MTGP', 'ours', 'full']:
            self.covar_module = gpytorch.kernels.MultitaskKernel(self.covar_module, num_tasks=2, rank=args.rank)
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if self.model_type in ['MTGP', 'ours', 'full']:
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) # error here!
        else:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)