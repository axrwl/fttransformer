import jax.numpy as jnp
from flax import linen as nn
from jax.nn import gelu
from einops import rearrange, repeat

class GEGLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        x, gates = jnp.split(x, 2, -1)
        return x * gelu(gates)
    
class FeedForward(nn.Module):
    dim: int
    mult: int = 4
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(features = self.dim * self.mult * 2)(x)
        x = GEGLU()(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        x = nn.Dense(features = self.dim)(x)
        return x
    
class Attention(nn.Module):
    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head ** -0.5
        self.norm = nn.LayerNorm(self.dim)
        self.to_qkv = nn.Dense(inner_dim * 3, use_bias = False)
        self.to_out = nn.Dense(self.dim, use_bias = False)
        self.dropout_l = nn.Dropout(self.dropout)

    def __call__(self, x):
        h = self.heads
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = jnp.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = nn.softmax(sim, axis = -1)
        dropped_attn = self.dropout_l(attn, deterministic = False)
        out = jnp.einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn
    
class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int
    attn_dropout: float
    ff_dropout: float

    def setup(self):
        self.layers = [( Attention(  
                            dim         = self.dim, 
                            heads       = self.heads, 
                            dim_head    = self.dim_head, 
                            dropout     = self.attn_dropout ), 
                        FeedForward(
                            dim     = self.dim, 
                            dropout = self.ff_dropout )
        ) for _ in range(self.depth)]

    def __call__(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x
        return x, jnp.stack(post_softmax_attns)
        
class NumericalEmbedder(nn.Module):
    dim: int
    num_numerical_types: int

    def setup(self):
        self.weights = self.param('weights', nn.initializers.normal(), (self.num_numerical_types, self.dim))
        self.biases = self.param('biases', nn.initializers.normal(), (self.num_numerical_types, self.dim))

    def __call__(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases
    
class FTTransformer(nn.Module):
    categories: list
    num_continuous: int
    dim: int
    depth: int
    heads: int
    dim_head: int = 16
    dim_out: int = 1
    num_special_tokens: int = 2
    attn_dropout: float = 0.
    ff_dropout: float = 0.

    def setup(self):
        self.num_categories = len(self.categories)
        self.num_unique_categories = sum(self.categories)
        total_tokens = self.num_unique_categories + self.num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = jnp.pad(jnp.array(self.categories), (1, 0), constant_values = self.num_special_tokens)
            self.categories_offset = categories_offset.cumsum(-1)[:-1]
            self.categorical_embeds = nn.Embed(total_tokens, self.dim)

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(self.dim, self.num_continuous)

        self.cls_token = self.param('cls_token', nn.initializers.normal(), (1, 1, self.dim))

        self.transformer = Transformer(
            dim             = self.dim, 
            depth           = self.depth, 
            heads           = self.heads, 
            dim_head        = self.dim_head, 
            attn_dropout    = self.attn_dropout, 
            ff_dropout      = self.ff_dropout
        )

        self.to_logits = nn.Sequential([
            nn.LayerNorm(self.dim),
            nn.relu,
            nn.Dense(self.dim_out)
        ])

    def __call__(self, x_categ, x_numer, return_attn = False):
        xs = []
        if self.num_unique_categories > 0:
            x_categ += self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        x = jnp.concatenate(xs, 1)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = jnp.concatenate((cls_tokens, x), 1)
        x, attns = self.transformer(x, return_attn = True)
        x = x[:, 0]
        logits = self.to_logits(x)

        if not return_attn:
            return logits
        return logits, attns