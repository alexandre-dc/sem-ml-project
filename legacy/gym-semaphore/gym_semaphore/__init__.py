from gym.envs.registration import register

register(
    id='semaphore-v0',
    entry_point='gym_semaphore.envs:SemaphoreEnv',
)