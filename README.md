Implementation of [Revisiting Deep Learning Models for Tabular Data
](https://arxiv.org/abs/2106.11959v2) in JAX based on [lucidrains/tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch).

### Usage
```py
from fttjax import FTTransformer
from jax import random

model = FTTransformer(
    categories = (10, 5, 6, 5, 8),
    num_continuous = 10,
    dim = 32,
    dim_out = 1,
    depth = 6,
    heads = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.1
)
x_categ =
x_numer =
rng = random.PRNGKey(0)
p_rng, d_rng = random.split(rng)
pred = model.init({'params': p_rng, 'dropout': d_rng}, x_categ, x_numer)
```

### Citation
```bibtex
@article{Gorishniy2021RevisitingDL,
    title   = {Revisiting Deep Learning Models for Tabular Data},
    author  = {Yu. V. Gorishniy and Ivan Rubachev and Valentin Khrulkov and Artem Babenko},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2106.11959}
}
```