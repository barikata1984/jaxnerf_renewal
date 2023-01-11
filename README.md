# jaxnerf_renewal
## Update
20230110: `train.py` and `utils.py` have been renewed.

## Objective
Replace the obsolete functionarities in the original [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).

## How changed
The codes below examplify how lines are changed from the original.

### `jax.host_id()` to `jax.process_index()` in `train.py` 
```python
︙
def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.host_id())
︙
```
to
```python
︙
def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.process_index())
︙
```

### `jax.host_count()` to `jax.process_count()` in `nerf/utils.py` 
```python
︙
  else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # host_count.
    rays_per_host = chunk_rays[0].shape[0] // jax.host_count()
︙
```
to
```python
︙
  else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # process_count.
    rays_per_host = chunk_rays[0].shape[0] // jax.process_count()
︙
```

### `flax.optim` to `optax` in `train.py`
```python
︙
  rng, key = random.split(rng)
  model, variables = models.get_model(key, dataset.peek(), FLAGS)
  optimizer = flax.optim.Adam(FLAGS.lr_init).create(variables)
︙
```
to
```python
︙
  rng, key = random.split(rng)
  model, variables = models.get_model(key, dataset.peek(), FLAGS)
  schedule = utils.create_learning_rate_decay_schedule(
    lr_init=FLAGS.lr_init,
    lr_final=FLAGS.lr_final,
    max_steps=FLAGS.max_steps,
    lr_delay_steps=FLAGS.lr_delay_steps,
    lr_delay_mult=FLAGS.lr_delay_mult)
  
  tx = optax.adam(learning_rate=schedule)
︙
```

### `jaxnerf.nerf.utils.TrainState` to `flax.train_state.TrainState` in `train.py`
```python
︙
  state = utils.TrainState(optimizer=optimizer)
︙
```
to
```python
︙
  state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
︙
```