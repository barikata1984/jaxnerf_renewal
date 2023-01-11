# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for Nerf."""

import functools
import gc
import time
from absl import app
from absl import flags
import flax
import optax # added
from flax.metrics import tensorboard
from flax.training import checkpoints, train_state
import jax
from jax import config
from jax import random
import jax.numpy as jnp
import numpy as np

from jaxnerf.nerf import datasets
from jaxnerf.nerf import models
from jaxnerf.nerf import utils

FLAGS = flags.FLAGS

utils.define_flags()
config.parse_flags_with_absl()


def train_step(rng, state, batch):
  """One optimization step.

  Args:
    rng: jnp.ndarray, random number generator.
    state: utils.TrainState, state of the model/optimizer.
    batch: dict, a mini-batch of data for training.

  Returns:
    new_state: utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
    rng: jnp.ndarray, updated random number generator.
  """
  rng, key_0, key_1 = random.split(rng, 3)

  def loss_fn(params):
    rays = batch["rays"]
    ret = state.apply_fn({"params": params}, key_0, key_1, rays, FLAGS.randomized)
    if len(ret) not in (1, 2):
      raise ValueError(
          "ret should contain either 1 set of output (coarse only), or 2 sets"
          "of output (coarse as ret[0] and fine as ret[1]).")
    # The main prediction is always at the end of the ret list.
    rgb, unused_disp, unused_acc = ret[-1]
    loss = ((rgb - batch["pixels"][Ellipsis, :3])**2).mean()
    psnr = utils.compute_psnr(loss)
    if len(ret) > 1:
      # If there are both coarse and fine predictions, we compute the loss for
      # the coarse prediction (ret[0]) as well.
      rgb_c, unused_disp_c, unused_acc_c = ret[0]
      loss_c = ((rgb_c - batch["pixels"][Ellipsis, :3])**2).mean()
      psnr_c = utils.compute_psnr(loss_c)
    else:
      loss_c = 0.
      psnr_c = 0.

    def tree_sum_fn(fn):
      return jax.tree_util.tree_reduce(
          lambda x, y: x + fn(y), params, initializer=0)

    weight_l2 = (
        tree_sum_fn(lambda z: jnp.sum(z**2)) /
        tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))

    stats = utils.Stats(
        loss=loss, psnr=psnr, loss_c=loss_c, psnr_c=psnr_c, weight_l2=weight_l2)
    return loss + loss_c + FLAGS.weight_decay_mult * weight_l2, stats

  (_, stats), grads = (
      jax.value_and_grad(loss_fn, has_aux=True)(state.params))
  grads = jax.lax.pmean(grads, axis_name="batch")
  statss = jax.lax.pmean(stats, axis_name="batch")

  # Clip the gradient by value.
  if FLAGS.grad_max_val > 0:
    clip_fn = lambda z: jnp.clip(z, -FLAGS.grad_max_val, FLAGS.grad_max_val)
    grads = jax.tree_util.tree_map(clip_fn, grads)

  # Clip the (possibly value-clipped) gradient by norm.
  if FLAGS.grad_max_norm > 0:
    grad_norm = jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2), grads, initializer=0))
    mult = jnp.minimum(1, FLAGS.grad_max_norm / (1e-7 + grad_norm))
    grads = jax.tree_util.tree_map(lambda z: mult * z, grads)
  
  new_state = state.apply_gradients(grads=grads)
  
  return new_state, stats, rng


def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by process_index() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.process_index())

  if FLAGS.config is not None:
    utils.update_flags(FLAGS)
  if FLAGS.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")
  if FLAGS.train_dir is None:
    raise ValueError("train_dir must be set. None set now.")
  if FLAGS.data_dir is None:
    raise ValueError("data_dir must be set. None set now.")
  dataset = datasets.get_dataset("train", FLAGS)
  test_dataset = datasets.get_dataset("test", FLAGS)

  rng, key = random.split(rng)
  model, variables = models.get_model(key, dataset.peek(), FLAGS) # model has a function called apply, which is apply_fn in train_state
                                                                  # variables is a dict that has params as its member
  schedule = utils.create_learning_rate_decay_schedule(
    lr_init=FLAGS.lr_init,
    lr_final=FLAGS.lr_final,
    max_steps=FLAGS.max_steps,
    lr_delay_steps=FLAGS.lr_delay_steps,
    lr_delay_mult=FLAGS.lr_delay_mult)
  
  tx = optax.adam(learning_rate=schedule) # tx means gradient transnformation
  state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx) # this instantiation works
  del tx, variables#, optimizer, params

  train_pstep = jax.pmap(
      train_step, 
      axis_name="batch",
      in_axes=(0, 0, 0),
      donate_argnums=(2,)) # here

  def render_fn(params, key_0, key_1, rays):
    return jax.lax.all_gather(
        state.apply_fn({"params": params}, key_0, key_1, rays, FLAGS.randomized),
        axis_name="batch")

  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0),  # Only distribute the data input.
      donate_argnums=(3,),
      axis_name="batch",
  )

  # Compiling to the CPU because it's faster and more accurate.
  ssim_fn = jax.jit(
      functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")

  if not utils.isdir(FLAGS.train_dir):
    utils.makedirs(FLAGS.train_dir)
  state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
  # Resume training a the step of the last checkpoint.
  init_step = state.step + 1
  state = flax.jax_utils.replicate(state)

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)

  # Prefetch_buffer_size = 3 x batch_size
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  n_local_devices = jax.local_device_count()
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = random.split(rng, n_local_devices)  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  stats_trace = []
  reset_timer = True
  for step, batch in zip(range(init_step, FLAGS.max_steps + 1), pdataset):
    if reset_timer:
      t_loop_start = time.time()
      reset_timer = False
    lr = schedule(step)
    state, stats, keys = train_pstep(keys, state, batch)
    if jax.process_index() == 0:
      stats_trace.append(stats)
    if step % FLAGS.gc_every == 0:
      gc.collect()

    # Log training summaries. This is put behind a process_index check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.process_index() == 0:
      if step % FLAGS.print_every == 0:
        summary_writer.scalar("train_loss", stats.loss[0], step)
        summary_writer.scalar("train_psnr", stats.psnr[0], step)
        summary_writer.scalar("train_loss_coarse", stats.loss_c[0], step)
        summary_writer.scalar("train_psnr_coarse", stats.psnr_c[0], step)
        summary_writer.scalar("weight_l2", stats.weight_l2[0], step)
        avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
        avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
        stats_trace = []
        summary_writer.scalar("train_avg_loss", avg_loss, step)
        summary_writer.scalar("train_avg_psnr", avg_psnr, step)
        summary_writer.scalar("learning_rate", lr, step)
        steps_per_sec = FLAGS.print_every / (time.time() - t_loop_start)
        reset_timer = True
        rays_per_sec = FLAGS.batch_size * steps_per_sec
        summary_writer.scalar("train_steps_per_sec", steps_per_sec, step)
        summary_writer.scalar("train_rays_per_sec", rays_per_sec, step)
        precision = int(np.ceil(np.log10(FLAGS.max_steps))) + 1
        print(("{:" + "{:d}".format(precision) + "d}").format(step) +
              f"/{FLAGS.max_steps:d}: " + f"i_loss={stats.loss[0]:0.4f}, " +
              f"avg_loss={avg_loss:0.4f}, " +
              f"weight_l2={stats.weight_l2[0]:0.2e}, " + f"lr={lr:0.2e}, " +
              f"{rays_per_sec:0.0f} rays/sec")
      if step % FLAGS.save_every == 0:
        state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            FLAGS.train_dir, state_to_save, int(step), keep=100)

    # Test-set evaluation.
    if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      t_eval_start = time.time()
      eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                   state)).params
      test_case = next(test_dataset)
      pred_color, pred_disp, pred_acc = utils.render_image(
          functools.partial(render_pfn, eval_variables),
          test_case["rays"],
          keys[0],
          FLAGS.dataset == "llff",
          chunk=FLAGS.chunk)

      # Log eval summaries on host 0.
      if jax.process_index() == 0:
        psnr = utils.compute_psnr(
            ((pred_color - test_case["pixels"])**2).mean())
        ssim = ssim_fn(pred_color, test_case["pixels"])
        eval_time = time.time() - t_eval_start
        num_rays = jnp.prod(jnp.array(test_case["rays"].directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar("test_rays_per_sec", rays_per_sec, step)
        print(f"Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec")
        summary_writer.scalar("test_psnr", psnr, step)
        summary_writer.scalar("test_ssim", ssim, step)
        summary_writer.image("test_pred_color", pred_color, step)
        summary_writer.image("test_pred_disp", pred_disp, step)
        summary_writer.image("test_pred_acc", pred_acc, step)
        summary_writer.image("test_target", test_case["pixels"], step)

  if FLAGS.max_steps % FLAGS.save_every != 0:
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    checkpoints.save_checkpoint(
        FLAGS.train_dir, state, int(FLAGS.max_steps), keep=100)


if __name__ == "__main__":
  app.run(main)
