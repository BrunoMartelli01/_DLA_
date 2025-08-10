
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.callbacks  import StopTrainingOnNoModelImprovement
import numpy as np

#idea ho provato a fare un reward positivo quando l'auto accelera e negativo quando frena
#cercare di trovare la linea di percorso migliore
class SpeedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SpeedRewardWrapper, self).__init__(env)
        self.past_action = [0, 0, 0, 0, 0]

    def reset(self, **kwargs):
        # Esegui il reset originale nell'ambiente
        obs, info = self.env.reset(**kwargs)
        self.past_action = [0, 0, 0, 0, 0]
        return obs, info

    def step(self, action):
        # Esegui il passo originale nell'ambiente
        obs, reward, terminated, truncated, info = self.env.step(action)



        if action == 3:
            reward += 0.05
            self.past_action.pop(0)
            self.past_action.append(1)


        if action == 4:
            reward -= 0.01
            self.past_action.pop(0)
            self.past_action.append(0)

        if np.mean(self.past_action) == 0:
            reward -= 0.1

        return obs, reward, terminated, truncated, info



#volevo evitare che l'auto uscisse dal percorso
class PunishmentWrapper(gym.Wrapper):
    def __init__(self, env, failure_threshold=20, punishment_multiplier=0.01):
        super(PunishmentWrapper, self).__init__(env)
        self.failure_threshold = failure_threshold
        self.punishment_multiplier = punishment_multiplier
        self.consecutive_negative_rewards = 0

    def reset(self, **kwargs):
        self.consecutive_negative_rewards = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward < 0:
            self.consecutive_negative_rewards += 1
            punishment = -self.punishment_multiplier * self.consecutive_negative_rewards
            reward += punishment
        else:

            self.consecutive_negative_rewards = 0
        if self.consecutive_negative_rewards >= self.failure_threshold:
            terminated = True
            reward -= 100



        return obs, reward, terminated, truncated, info



if __name__ == '__main__':
    env_id = "CarRacing-v3"
    n_envs = 8
    env_kwargs = {"continuous": False, "max_episode_steps": 1000}
    wrapper_kwargs = {"keep_dim": True}


    def make_env():
        env = gym.make(env_id, **env_kwargs)
        # env = SpeedRewardWrapper(env)
        # env = PunishmentWrapper(env)

        return env

    vec_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=GrayscaleObservation,
        wrapper_kwargs=wrapper_kwargs
    )
    vec_env = VecFrameStack(vec_env, n_stack=5)
    vec_env = VecTransposeImage(vec_env)


    print("Ambiente vettorizzato creato con successo!")
    print("Classe dell'ambiente:", type(vec_env))
    print("Numero di ambienti paralleli:", vec_env.num_envs)
    print("Spazio di osservazione:", vec_env.observation_space.shape)
    print("Spazio di azione:", vec_env.action_space)



    eval_env = make_vec_env(
        env_id ,
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=GrayscaleObservation,
        wrapper_kwargs=wrapper_kwargs,
        env_kwargs = env_kwargs,
    )

    eval_env = VecFrameStack(eval_env, n_stack=5)
    eval_env = VecTransposeImage(eval_env)


    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path="./logs/",
                                 log_path="./logs/",
                                 eval_freq=max(25_000//n_envs, 1),
                                 n_eval_episodes=5,
                                 deterministic=True,
                                 render=False,
                                 callback_after_eval=StopTrainingOnNoModelImprovement(40)
                                 )


    model = PPO("CnnPolicy", vec_env, device="cuda", verbose=1)
    model.learn(total_timesteps=1_000_000_000,
                callback=eval_callback,
                progress_bar=True
                )



    print("Training completato!")
    eval_env.close()
    vec_env.close()



