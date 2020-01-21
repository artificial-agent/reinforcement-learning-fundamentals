# This file is meant to act as an exerimentation testbed for the k-armed bandits problem.
# Refer to Sutton and Barto's Reinforcement Learning 2nd ed, Chapter 2 for the background.
# Experiment by passing different options for parameters to the Bandit class.
# The default is to repeat a experiment 2000 times, for 1000 time steps each, to obtain
# an average of the expected reward / optimal action percentage at each time step.
# Sample figures have been provided to showcase what is possible with this testbed.

# Library Imports
import os # saving files
import numpy as np 
import matplotlib # Visualization
import matplotlib.pyplot as plt
from tqdm import trange # Progress Bar

class Bandit:
  # @k_arm: # of arms
  # @initial_epsilon: probability for exploration in epsilon-greedy algorithm
  # @initial: initial estimation for each action
  # @step_size: constant step size for updating estimations
  # @sample_averages: if True, use sample averages to update estimations instead of constant step size
  # @UCB_param: if not None, use UCB algorithm to select action
  # @gradient: if True, use gradient based bandit algorithm
  # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
  # @true_reward: mean of the distribution
  
  default_options = {
    'k_arm': 10,
    'action_method': {
      'name': 'epsilon_greedy',
      'parameters': {
        'initial_epsilon': 0.1
      }
    },
    'update_est_method': {
      'name': 'sample_averages'
    },
    'initial_estimation': 0,
    'true_reward': 0
  }

  # Constructor
  def __init__(self, options = default_options):
    # Bandit Parameters
    self.k = options['k_arm']
    self.action_method = options['action_method']
    self.update_est_method = options['update_est_method']
    self.initial_estimation = options['initial_estimation']
    self.true_reward = options['true_reward']
    self.cumulative_reward = options['cumulative_reward']
    #Initial values
    self.q_true = np.random.randn(self.k) + self.true_reward # Initial reward scalar for each action
    self.q_estimation = np.zeros(self.k) + self.initial_estimation # Inital estimates for each action
    self.best_action = np.argmax(self.q_true) # Returns index of action that has the highest value

    # Bookkeeping
    self.time = 0
    self.average_reward = 0
    self.indices = np.arange(self.k) # Return k evenely spaced values
    self.action_count = np.zeros(self.k) #  Number of times each action has been selected

  # Get an action for this bandit
  # Returns a number between 1 and k where k is the number of arms
  def act(self):
    # Action selection methods

    # Epsilon Greedy
    if(self.action_method['name'] == 'epsilon_greedy'):
      
      # Get epsilon value - sometimes our epsilon may be the result of a function rather than constant
      epsilon = self.__getEpsilonValue(self.action_method['parameters'], self.time)

      # Should I explore? Explore if random value between 0 - 1 is less than epsilon
      if np.random.rand() < epsilon:
        return np.random.choice(self.indices)
      # Greedy
      else:
        q_best = np.max(self.q_estimation) # chose greedily
        # Search through the array of estimations and return the index of the estimation that is the same as our q_best
        index = np.where(self.q_estimation == q_best)[0]
        # If there are multiple estimations with the same value, chose a random choice
        return np.random.choice(index)

    # UCB
    elif(self.action_method['name'] == 'UCB'):
      UCB_estimation = self.q_estimation + self.action_method['parameters']['UCB_param'] * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
      q_best = np.max(UCB_estimation)
      return np.random.choice(np.where(UCB_estimation == q_best)[0])

    # Gradient
    elif(self.action_method['name'] == 'gradient'):
      exp_est = np.exp(self.q_estimation)
      self.action_prob = exp_est / np.sum(exp_est)
      return np.random.choice(self.indices, p = self.action_prob)
  
  # Used for epsilon value selection in epsilon greedy action selection
  # Sometimes you may want the epsilon value to decrease over time
  def __getEpsilonValue(self, action_method_params, time):
    try:
      if(action_method_params['epsilon_method'] == 'geometriclly_decreasing'):
        return np.exp(-action_method_params['p_epsilon'] * time)
      elif(action_method_params['epsilon_method'] == 'exponentially_decreasing'):
        return action_method_params['p_epsilon'] ** time
      
      # constant epsilon
      else:
        return action_method_params['initial_epsilon']
    except:
      print(''' 
        Missing value for the following variable
       'options['action_method']['parameters']['epsilon_method']'. Make sure to add it to your
       options when initalizing a bandit. ex) {...,'epsilon_method'; 'constant',...}
      ''')
      raise

  # Take an action, get reward back, update estimation
  def step(self, action):
    # Take an action and recieve some reward
    # amount of reward is our action's true reward scalar + single sample from a random normal distribution
    reward =  self.q_true[action] + np.random.randn()

    # update time, action count, and average reward
    self.time += 1
    self.action_count[action] += 1
    self.average_reward += (reward - self.average_reward) / self.time

    # Update estimation (Learning Step)
    self.__update_estimation(action, reward)

    return reward

  # update the value_estimation (Q_t) for this action
  def __update_estimation(self, action, reward):
    if(self.update_est_method['name'] == 'sample_averages'):
      self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]

    elif(self.update_est_method['name'] == 'constant_step_size'):
      self.q_estimation[action] += self.update_est_method['parameters']['step_size'] * (reward - self.q_estimation[action])

    elif(self.update_est_method['name'] == 'gradient'):
      one_hot = np.zeros(self.k)
      one_hot[action] = 1

      if self.update_est_method['parameters']['gradient_baseline'] == True:
        baseline = self.average_reward

      self.q_estimation += self.update_est_method['parameters']['step_size'] * (reward - baseline) * (one_hot - self.action_prob)

  # Generate random true reward, estimation, action, reset time
  def reset(self):
    self.q_true = np.random.randn(self.k) + self.true_reward # inital reward scalar for each action
    self.q_estimation = np.zeros(self.k) + self.initial_estimation # initial estimation for each action

    # reset time
    self.time = 0
    # reset number of times each action was chosen - represented by an array of size k with all zeros
    self.action_count = np.zeros(self.k)

    # determine best initial action based on highest reward scalar
    self.best_action = np.argmax(self.q_true) #returns index of best highest value


# K Armed Bandits Simulation
def simulate(runs, time, bandits):
  rewards = np.zeros((len(bandits), runs, time)) #initialize 3D array
  best_action_counts = np.zeros(rewards.shape) # initiale 3D array same size as rewards

  for i, bandit in enumerate(bandits):
    for r in trange(runs): #equiv to tqdm(range(runs)) - gives us the completion bar
      reward = 0
      bandit.reset()
      for t in range(time):
        action = bandit.act() # returns a value between 0-K
        if(bandit.cumulative_reward):
          reward += bandit.step(action)
        else:
          reward = bandit.step(action)
        rewards[i, r, t] = reward
        if action == bandit.best_action:
          best_action_counts[i, r, t] = 1

  # calcualte percentage of times the best action was taken on this run
  mean_best_action_counts = best_action_counts.mean(axis=1) # condense the num runs (matrix) into a single vector for each epsilon (how often does 1 (optimal action) appear at each time step)
  # calculate mean reward for this run
  mean_rewards = rewards.mean(axis=1) # condense num runs (matrix) into a single run (vector) for each epsilon. average reward at each time step across all 'n' runs s.t. n = num runs 
  return mean_best_action_counts, mean_rewards

############################ EXPERIMENTS #############################################################
## Define your own experiments here
def experiment_1(runs = 2000, time = 1000):
  parameters = [
    # Constant
    {
      'initial_epsilon': .1,
      'epsilon_method': 'constant',
    },
    # Geometricly Decreasing
    {
      'initial_epsilon': .1,
      'epsilon_method': 'geometriclly_decreasing',
      'p_epsilon': .1
    },
    # Exponentially Decreasing
    {
      'initial_epsilon': .9,
      'epsilon_method': 'exponentially_decreasing',
      'p_epsilon': .9
    }
  ]

  # One bandit for each epsilon
  bandits = [ Bandit({
    'k_arm': 10,
    'action_method': {
      'name': 'epsilon_greedy',
      'parameters': param_val
    },
    'update_est_method': {
      'name': 'sample_averages',
      'parameters': {
        'step_size': 0.1
      }
    },
    'initial_estimation': 0,
    'true_reward': 0,
    'cumulative_reward': False
  }) for param_val in parameters ]

  epsilons = ['constant - 0.1', 'geometriclly_decreasing - 0.1', 'exponentially_decreasing - 0.9']

  # District 12, mainframe laser control, begin simulation.
  best_action_counts, rewards = simulate(runs, time, bandits)

  # Setup Mat Plot Lib
  plt.figure(figsize=(10, 20))

  # Subplot 1
  plt.subplot(2, 1, 1)

  i = 0
  for eps, rewards in zip(epsilons, rewards):
    plt.plot(rewards, label=f'{eps}')
    i += 1
  plt.xlabel('steps')
  plt.ylabel('average reward')
  plt.legend()

  # Subplot 2 
  plt.subplot(2, 1, 2)
  i = 0
  for eps, counts in zip(epsilons, best_action_counts):
    plt.plot(counts, label=f'{eps}')
    i += 1
  plt.xlabel('steps')
  plt.ylabel('% optimal action')
  plt.legend()
  plt.savefig(f'figure_2_2 - optimal vals - 1000 runs.png')
  plt.close()

############################ START OF PROGRAM ########################################################
if __name__ == "__main__":
  experiment_1() # THIS IS JUST AN EXAMPLE
############################ END OF PROGRAM ########################################################


############################ EXAMPLE BANDITS #######################################################
## Epsilon Greedy Bandit
E_greedy_options = {
  'k_arm': 10,
  'action_method': {
    'name': 'epsilon_greedy',
    'parameters': {
      'initial_epsilon': 0.1,
      'epsilon_method': 'constant',
      'p_epsilon': 0.1
    }
  },
  'update_est_method': {
    'name': 'sample_averages'
  },
  'initial_estimation': 0,
  'true_reward': 0,
}

## UCB Bandit
UCB_options = {
  'k_arm': 10,
  'action_method': {
    'name': 'UCB',
    'parameters': {
      'UCB_param': 0.1
    }
  },
  'update_est_method': {
    'name': 'constant_step_size'
  },
  'initial_estimation': 0,
  'true_reward': 0
}

## Gradient Bandit
gradient_options = {
  'k_arm': 10,
  'action_method': {
    'name': 'gradient',
  },
  'update_est_method': {
    'name': 'gradient',
    'parameters': {
      'gradient_baseline': True,
      'step_size': 0.1
    },
  },
  'initial_estimation': 0,
  'true_reward': 0
}