import gym
import numpy as np

env = gym.make('CartPole-v0')
seed = 1234 # seed for reproducibility

iters = 200 # number of policy updates
T = 70 # number of steps per trajectory
H = 100 # number of trajectories
sigma = 0.05 # std for control noise

# Initialize theta
theta = np.random.randn(env.action_space.n, np.prod(env.observation_space.shape))

for iters in range(iters):
    # Create empty variables for later value update
    Phi = np.zeros([H*T, np.prod(env.observation_space.shape)])
    A = np.zeros([H*T, env.action_space.n])
    Q = np.eye(H*T)
    
    for i in range(H):
        env.seed(seed=seed) # Give same initial state with env.reset()
        state = np.expand_dims(env.reset(), axis=1) # initial state
        Phi[i*T, :] = state.squeeze() # Append to Phi

        q_value_list = np.zeros([T]) # Accumulator for Q
        for j in range(T):
            action_noise = np.random.normal(0, sigma, size=[env.action_space.n, 1])

            action_scores = theta.dot(state) + action_noise
            A[i*T + j, :] = action_scores.squeeze() # Append to A

            action_prob = np.exp(action_scores)/np.sum(np.exp(action_scores))

            observation, reward, done, info = env.step(np.random.choice(env.action_space.n, 
                                                                        size=1, 
                                                                        p=action_prob.squeeze()).squeeze())
            q_value_list[j] = reward # Append q value list

        # Append Q value matrix
        for j in range(len(q_value_list)):
            Q[i*T + j, i*T + j] = np.sum(q_value_list[j:])
    
    # Policy update
    old_theta = theta
    theta = np.linalg.pinv((Phi.T).dot(Q).dot(Phi)).dot(Phi.T).dot(Q).dot(A).T
    
    #print(np.sum(np.abs(theta - old_theta)))

# New test
env.seed(seed=seed) # Give same initial state with env.reset()
state = np.expand_dims(env.reset(), axis=1)
for i in range(100):
    env.render(close=True)
    
    action_noise = np.random.normal(0, sigma, size=[env.action_space.n, 1])
    action_scores = theta.dot(state) + action_noise
    action_prob = np.exp(action_scores)/np.sum(np.exp(action_scores))
    observation, reward, done, info = env.step(np.random.choice(env.action_space.n, size=1, p=action_prob.squeeze()).squeeze())
    
    print('done {}: '.format(i), done)