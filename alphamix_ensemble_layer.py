from numpy import dtype
import torch
import torch.nn as nn
import torch.functional as F


class BatchEnsemble (nn.Linear):
    def __init__(self, 
                input_dim,
                units,
                ensemble_size=4,
                use_bias=True,
                ):
        super().__init__(input_dim,units,True)
        self.input_dim = input_dim
        self.units = units
        self.ensemble_size = ensemble_size
        self.use_ensemble_bias = use_bias
        alpha_shape = [self.ensemble_size, self.input_dim]
        gamma_shape = [self.ensemble_size, self.units]
        bias_shape = [self.ensemble_size, self.units]
        self.alpha = nn.Parameter(torch.ones(alpha_shape, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones(gamma_shape, dtype=torch.float32))
        self.ensemble_bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))


        # weight = torch.Tensor(self.input_size)
        # self.weight = nn.Parameter(weight)
        # torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        input_dim = self.alpha.shape[-1]
        # print(batch_size)
        # print(self.ensemble_size)
        examples_per_model = batch_size // self.ensemble_size

        inputs= inputs.view(self.ensemble_size, examples_per_model, input_dim)
        alpha = torch.unsqueeze(self.alpha, 1)
        gamma = torch.unsqueeze(self.gamma, 1)
        perturb_x = inputs * alpha
        outputs = super().forward(perturb_x) * gamma

        bias = torch.unsqueeze(self.ensemble_bias, 1)
        outputs += bias
        return outputs.reshape((batch_size,self.units))

# a = BatchEnsemble(10,7)
# print(a.weight)

# myNetwork = nn.Sequential(
#     a
# )


# input = torch.ones([8,10])
# print(myNetwork(input).shape)