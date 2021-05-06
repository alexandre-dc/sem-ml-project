from gym.envs.registration import register

register(
    id='sem-v0',
    entry_point='gym_sem.envs:SemEnv',
)