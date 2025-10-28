import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack,DummyVecEnv, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import GrayscaleObservation

import numpy as np
NUM_EPISODES = 10


env_id = "CarRacing-v3"
env_kwargs = {"continuous": False, "render_mode":"human", "max_episode_steps": 1000}
wrapper_kwargs = {"keep_dim": True}


eval_env = make_vec_env(
    env_id=env_id,
    n_envs=1,
    vec_env_cls=DummyVecEnv,
    wrapper_class=GrayscaleObservation,
    wrapper_kwargs=wrapper_kwargs,
    env_kwargs=env_kwargs,
)

eval_env = VecFrameStack(eval_env, n_stack=5)
eval_env = VecTransposeImage(eval_env)
model = PPO.load("../logs/best_model_908.60_s.zip", device="cuda")

try:
    for episode in range(NUM_EPISODES):
        obs = eval_env.reset()
        terminated = False
        total_reward = 0.0
        print(obs.shape)
        print(f"\n--- Inizio Episodio {episode + 1}/{NUM_EPISODES} ---")

        while not terminated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated= eval_env.step(action)
            print(f" Troncato: {terminated} , Ricompensa: {reward}")
            total_reward += reward
            eval_env.render()

        print(f"--- Fine Episodio {episode + 1} --- Ricompensa Totale: {total_reward}")

except KeyboardInterrupt:
    print("\nValutazione interrotta dall'utente.")

finally:
    eval_env.close()
    print("\nAmbiente chiuso. Valutazione terminata.")
