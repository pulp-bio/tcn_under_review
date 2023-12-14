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

from tcn import TCN, summarize

# As explained in the manuscript's Subsection III-E, "Temporal Convolutional
# Network: Structure, Training, and Deployment," Batch-norm (BN) folding is
# used after training to merge each BN with its previous layer, slightly
# reducing the number of parameters and operations.

# The network's structure and size that we are interested in is the final one
# deployed and executed on the MCU. So, we call our ``TCN``'s constructor with
# the flag ``bn = False``, which omits BN's, and the flag ``bias = True``,
# which accounts for the biases that each folded BN confers to its previous
# linear layer (either 1d-convolutional or dense).

bias = True
bn = False
tcn = TCN(bias=bias, bn=bn)
tcn.eval()

verbose = 1
tcn_model_stats = summarize(tcn, verbose=1)

OUT_FILE_NAME = 'tcn_table.txt'
with open(OUT_FILE_NAME, 'w') as out_file:
    print(f"{tcn_model_stats}", file=out_file)
