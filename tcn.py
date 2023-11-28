"""
    Author(s):
    Marcello Zanghieri <marcello.zanghieri2@unibo.it>
    
    Copyright (C) 2023 University of Bologna
    
    Licensed under the GNU Lesser General Public License (LGPL), Version 2.1
    (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        https://www.gnu.org/licenses/lgpl-2.1.txt
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from __future__ import annotations

import torch
from torch import nn
import torchinfo


NUM_CHANNELS = 9
NUM_SAMPLES = 2048


class TCN(nn.Module):

    def __init__(
        self,
        num_channels: int=NUM_CHANNELS,
        bias: bool=False,
        bn: bool=True,
    ):
        super(TCN, self).__init__()
        
        self.num_channels = num_channels
        self.bias = bias
        self.bn = bn
        
        self.b0_tcn = nn.Conv1d(
            num_channels, 4, kernel_size=3, padding=1,
            stride=2, bias=bias)
        if bn:
            self.b0_bn = nn.BatchNorm1d(4)
        self.b0_relu = nn.ReLU()

        self.b1_tcn = nn.Conv1d(
            4, 4, 3, padding=1, stride=2, bias=bias)
        if bn:
            self.b1_bn = nn.BatchNorm1d(4)
        self.b1_relu = nn.ReLU()

        self.b2_tcn = nn.Conv1d(
            4, 2, 3, padding=1, stride=2, bias=bias)
        if bn:
            self.b2_bn = nn.BatchNorm1d(2)
        self.b2_relu = nn.ReLU()

        self.b3_tcn = nn.Conv1d(
            2, 2, 3, padding=1, stride=2, bias=bias)
        if bn:
            self.b3_bn = nn.BatchNorm1d(2)
        self.b3_relu = nn.ReLU()

        self.b4_tcn = nn.Conv1d(
            2, 1, 3, padding=1, stride=2, bias=bias)
        if bn:
            self.b4_bn = nn.BatchNorm1d(1)
        self.b4_relu = nn.ReLU()

        self.b5_tcn = nn.Conv1d(
            1, 1, 3, padding=1, stride=2, bias=bias)
        if bn:
            self.b5_bn = nn.BatchNorm1d(1)
        self.b5_relu = nn.ReLU()

        self.fc0 = nn.Linear(32, 8, bias=bias)
        if bn:
            self.fc0_bn = nn.BatchNorm1d(8)
        self.fc0_relu = nn.ReLU()

        self.fc1 = nn.Linear(8, 8, bias=bias)
        if bn:
            self.fc1_bn = nn.BatchNorm1d(8)
        self.fc1_relu = nn.ReLU()

        self.fc2 = nn.Linear(8, 1, bias=bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        
        if self.bn:
            
            x = self.b0_relu(self.b0_bn(self.b0_tcn(x)))
            x = self.b1_relu(self.b1_bn(self.b1_tcn(x)))
            x = self.b2_relu(self.b2_bn(self.b2_tcn(x)))
            x = self.b3_relu(self.b3_bn(self.b3_tcn(x)))
            x = self.b4_relu(self.b4_bn(self.b4_tcn(x)))
            x = self.b5_relu(self.b5_bn(self.b5_tcn(x)))

            x = x.flatten(1)

            x = self.fc0_relu(self.fc0_bn(self.fc0(x)))
            x = self.fc1_relu(self.fc1_bn(self.fc1(x)))
            y = self.fc2(x)
        
        else:
            
            x = self.b0_relu(self.b0_tcn(x))
            x = self.b1_relu(self.b1_tcn(x))
            x = self.b2_relu(self.b2_tcn(x))
            x = self.b3_relu(self.b3_tcn(x))
            x = self.b4_relu(self.b4_tcn(x))
            x = self.b5_relu(self.b5_tcn(x))

            x = x.flatten(1)

            x = self.fc0_relu(self.fc0(x))
            x = self.fc1_relu(self.fc1(x))
            y = self.fc2(x)

        return y


def summarize(
    model: nn.Module,
    verbose: 0 | 1 | 2 = 0,
) -> torchinfo.ModelStatistics:

    # set all parameters for the function torch.summary

    input_size = (NUM_CHANNELS, NUM_SAMPLES)
    batch_dim = 0  # index of the batch dimension
    col_names = [
        'input_size',
        'output_size',
        'num_params',
        'params_percent',
        'kernel_size',
        'mult_adds',
        'trainable',
    ]
    device = 'cpu'
    mode = 'eval'
    row_settings = [
        'ascii_only',
        'depth',
        'var_names',
    ]

    # call the function torch.summary

    model_stats = torchinfo.summary(
        model=model,
        input_size=input_size,
        batch_dim=batch_dim,
        col_names=col_names,
        device=device,
        mode=mode,
        row_settings=row_settings,
        verbose=verbose,
    )

    return model_stats


def main() -> None:
    pass


if __name__ == '__main__':
    main()
