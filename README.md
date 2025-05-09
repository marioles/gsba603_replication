# Reproduction of Kinnan et al. (2024)

This package reproduces and performs robustness checks for some of the main results of the paper "Propagation and
Insurance in Village Networks" by Kinnan, Samphantharak, Townsend & Vera-Cossio (2024).

# Features

* Reproduction of figures 1 and 2, and table 1 from the original paper.
* Robustness checks for the direct effects results, including bootstrapping, placebo test and parameter sensitivity.

# Configuration

This project requires a `.env` file in the root directory with the following environment variables:

* `BASE_PATH`: base directory with all the data
* `CLEAN_PATH`: directory within `BASE_PATH` that contains the pre-processed data
* `DYADS_FILE_PATH`: name of file containing processed network data
* `FILE_PATH`: name of file containing processed direct effects data
* `EXPORT_PATH`: directory where outputs will be stored

# Usage

By executing the `main()` function of each module, the resulting figures or tables will be stored in the export path.

```python
from gsba603_replication import figure_1, figure_1_robustness, figure_2, table_1

# Direct effects results
figure_1.main()
table_1.main()

# Direct effects robustness
figure_1_robustness.main()

# Indirect effects results
figure_2.main()

```

# References
- Kinnan, C., Samphantharak, K., Townsend, R., & Vera-Cossio, D. (2024). Propagation and insurance in village networks. American Economic Review, 114(1), 252-284.