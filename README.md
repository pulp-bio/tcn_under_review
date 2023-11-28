# Tiny TCN by Zanghieri *et al*. (UNDER REVIEW)

This repository implements the Temporal Convolutional Network (TCN) [[1]](#1) [[2]](#2) presented in a manuscript by M. Zanghieri *et al*. [[3]](#3) currently under review at the IEEE *Access* journal.
We provide this as a supplementary documentation for the peer-review now in progress.
For a **technical report** about an earlier stage of the same research project, please refer to F. Conti *et al*. [[4]](#4).



## Usage

The requirements (see ``requirements.txt``) are the Python packages PyTorch (v. 1.9.0) and torchinfo (v. 1.8.0), which can be quickly installed via the shell with the command
```
> python -m pip install -r requirements.txt``
```
The TCN is implemented in the module ``tcn.py``.
1. Run ``visualize_tcn_table.ipynb`` (or equivalently ``visualize_tcn_table.py``) to generate the TCN, its ``torchinfo.ModelStatistics``, and the printed table.
2. See the printed table in the standard output or in the file ``tcn_table.txt``.



## Authors

The manuscript documented by this repository involves several authors.
The piece of work reported here is developed at the Energy-Efficient Embedded Systems (EEES) Lab of University of Bologna (Italy) by:
- [Marcello Zanghieri](https://scholar.google.com/citations?user=WnIqQj4AAAAJ&hl=en) (Conceptualization, Software, Analysis)
- [Prof. Francesco Conti](https://scholar.google.it/citations?user=A70PCXoAAAAJ&hl=en) (Supervision, Funding Acquisition)
- [Prof. Luca Benini](https://scholar.google.com/citations?user=8riq3sYAAAAJ&hl=en) (Supervision, Funding Acquisition)

Prof. Luca Benini is also with the ETH Zürich (Switzerland).



## Citation

This work is still under review.



## References

<a id="1">[1]</a>
C. Lea *et al*., "Temporal convolutional networks for action segmentation and detection," *CoRR*, vol. [abs/1611.05267](https://doi.org/10.48550/arXiv.1611.05267), 2016.

<a id="2">[2]</a>
S. Bai *et al*., "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," *CoRR*, vol. [abs/1803.01271 (https://doi.org/10.48550/arXiv.1803.01271), 2018.

<a id="3">[3]</a>
M. Zanghieri *et al*., MANUSCRIPT UNDER REVIEW, [*IEEE Access*](https://ieeeaccess.ieee.org/).

<a id="4">[4]</a>
F. Conti *et al*., "AI-powered collision avoidance safety system for industrial woodworking machinery," in *AI4DI – Applications*. River Publishers, 2021. DOI: [10.1201/9781003337232-17](https://www.doi.org/10.1201/9781003337232-17).



## License

All files are released under the LGPL-2.1 license (`LGPL-2.1`) (see `LICENSE`).
