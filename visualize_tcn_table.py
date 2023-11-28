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


bias = True
bn = False
tcn = TCN(bias=bias, bn=bn)
tcn.eval()

verbose = 1
tcn_model_stats = summarize(tcn, verbose=1)

OUT_FILE_NAME = 'tcn_table.txt'
with open(OUT_FILE_NAME, 'w') as out_file:
    print(f"{tcn_model_stats}", file=out_file)
