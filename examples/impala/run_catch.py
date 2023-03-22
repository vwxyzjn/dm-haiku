# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Single-process IMPALA wiring."""

import functools
import threading
from typing import List

import os
import uuid
os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

from absl import app
from bsuite.environments import catch
import actor as actor_lib
import agent as agent_lib
import haiku_nets
import learner as learner_lib
import util
import jax
import optax

from atari_env import AtariEnv
from tensorboardX import SummaryWriter

ACTION_REPEAT = 1
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
MAX_ENV_FRAMES = 50000000
NUM_ACTORS = 32
UNROLL_LENGTH = 20

FRAMES_PER_ITER = ACTION_REPEAT * BATCH_SIZE * UNROLL_LENGTH


def run_actor(actor: actor_lib.Actor, stop_signal: List[bool]):
  """Runs an actor to produce num_trajectories trajectories."""
  while not stop_signal[0]:
    frame_count, params = actor.pull_params()
    actor.unroll_and_push(frame_count, params)


def main(_):
  env_id = "Breakout-v5"
  seed = 1
  run_name = f"dm_haiku_impala_{env_id}_{seed}_{uuid.uuid4()}"

  import wandb

  wandb.init(
      project="cleanrl",
      sync_tensorboard=True,
      name=run_name,
      monitor_gym=True,
      save_code=True,
  )
  writer = SummaryWriter(f"runs/{run_name}")

  # A thunk that builds a new environment.
  # Substitute your environment here!
  build_env = AtariEnv

  # Construct the agent. We need a sample environment for its spec.
  env_for_spec = build_env()
  num_actions = env_for_spec.action_spec().num_values
  agent = agent_lib.Agent(num_actions, env_for_spec.observation_spec(),
                          functools.partial(haiku_nets.AtariNet, use_resnet=False, use_lstm=False))

  # Construct the optimizer.
  max_updates = MAX_ENV_FRAMES / FRAMES_PER_ITER
  opt = optax.rmsprop(5e-3, decay=0.99, eps=1e-7)

  # Construct the learner.
  learner = learner_lib.Learner(
      agent,
      jax.random.PRNGKey(428),
      opt,
      BATCH_SIZE,
      DISCOUNT_FACTOR,
      FRAMES_PER_ITER,
      max_abs_reward=1.,
      logger=util.AbslLogger(),  # Provide your own logger here.
      writer=writer,
  )

  # Construct the actors on different threads.
  # stop_signal in a list so the reference is shared.
  actor_threads = []
  stop_signal = [False]
  for i in range(NUM_ACTORS):
    actor = actor_lib.Actor(
        agent,
        build_env(),
        UNROLL_LENGTH,
        learner,
        rng_seed=i,
        logger=util.AbslLogger(),  # Provide your own logger here.
        writer=writer,
    )
    args = (actor, stop_signal)
    actor_threads.append(threading.Thread(target=run_actor, args=args))

  # Start the actors and learner.
  for t in actor_threads:
    t.start()
  learner.run(int(max_updates))

  # Stop.
  stop_signal[0] = True
  for t in actor_threads:
    t.join()


if __name__ == '__main__':
  app.run(main)
