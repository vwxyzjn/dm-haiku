from bsuite.environments import base

import dm_env
from dm_env import specs
import numpy as np
import envpool

ATARI_MAX_FRAMES = int(
    108000 / 4
)  # 108000 is the max number of frames in an Atari game, divided by 4 to account for frame skipping

def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=False,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 6
            repeat_action_probability=0.25,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12
            noop_max=1,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) p. 12 (no-op is deprecated in favor of sticky action, right?)
            full_action_space=True,  # Machado et al. 2017 (Revisitng ALE: Eval protocols) Tab. 5
            max_episode_steps=ATARI_MAX_FRAMES,  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
            seed=seed,
        )
        return envs

    return thunk


class AtariEnv(base.Environment):
  """A Catch environment built on the dm_env.Environment class.

  The agent must move a paddle to intercept falling balls. Falling balls only
  move downwards on the column they are in.

  The observation is an array shape (rows, columns), with binary values:
  zero if a space is empty; 1 if it contains the paddle or a ball.

  The actions are discrete, and by default there are three available:
  stay, move left, and move right.

  The episode terminates when the ball reaches the bottom of the screen.
  """

  def __init__(self, env_id="Breakout-v5", seed=1):
    """Initializes a new Catch environment.

    Args:
      rows: number of rows.
      columns: number of columns.
      seed: random seed for the RNG.
    """
    _total_regret = 0
    self.env = make_env(env_id, seed, 1)()

  def _reset(self) -> dm_env.TimeStep:
    obs = self.env.reset()[0]
    obs = np.transpose(obs, (1, 2, 0))

    return dm_env.restart(obs)

  def _step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    if self._reset_next_step:
      return self.reset()

    obs, reward, done, info = self.env.step(np.array([action]))
    obs = obs[0]
    obs = np.transpose(obs, (1, 2, 0))
    reward = reward[0]
    done = done[0]

    # Check for termination.
    if done:
      self._reset_next_step = True
      return dm_env.termination(reward=reward, observation=obs)

    return dm_env.transition(reward=reward, observation=obs)

  def observation_spec(self) -> specs.BoundedArray:
    """Returns the observation spec."""
    return specs.BoundedArray(shape=np.array([84, 84, 4]), dtype=self.env.observation_space.dtype,
                              name="board", minimum=0, maximum=255)

  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return specs.DiscreteArray(
        dtype=np.int, num_values=self.env.action_space.n, name="action")

#   def _observation(self) -> np.ndarray:
#     self._board.fill(0.)
#     self._board[self._ball_y, self._ball_x] = 1.
#     self._board[self._paddle_y, self._paddle_x] = 1.

#     return self._board.copy()

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)
