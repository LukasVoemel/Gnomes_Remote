import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque



def norm_helper(value, exp_max, exp_min):
    """ 
    This helper function returns a value that
    is approximately between [-1,1] given a value,
    the expected max and the expected min. 
    """
    return 2 * (value / (exp_max - exp_min)) - 1

def reward(home):
    """
    Reward function
    :input: PlayerHome object
    :output: float
    """
    print(home.obs_dict)
    #reward = home.obs_dict['my_demand'] / 3.5

    current_indoor_temp = home.obs_dict["t_in"]
    #current_tank_temp = home.obs_dict["t_wh"]
    current_out_temp = home.obs_dict["t_out"]
    predict_out_temp6 = home.obs_dict["t_out_6hr"]
    predict_out_temp12 = home.obs_dict["t_out_12hr"]
    current_outdoor_hum = home.obs_dict["ghi"]
    predict_out_hum6 = home.obs_dict["ghi_6hr"]
    predict_out_hum12 = home.obs_dict["ghi_12hr"]
    time_of_day = home.obs_dict["time_of_day"]
    day_of_week = home.obs_dict["day_of_week"]
    #occupancy = home.obs_dict["occupancy"]
    # current_net_consumption = home.obs_dict["my_demand"]
    # reward = 0

    # # Penalize high electricity consumption during occupancy
    # # if occupancy:
    # #     reward -= current_net_consumption / 5.0

    # # Penalize high temperature difference between indoor and outdoor
    # reward -= abs(current_indoor_temp - current_out_temp) / 2.0

    # # Penalize high tank temperature
    # # if current_tank_temp > 55:
    # #     reward -= (current_tank_temp - 55) / 2.0

    # # Penalize high predicted outdoor temperature and humidity
    # if predict_out_temp6 > 30:
    #     reward -= (predict_out_temp6 - 30) / 2.0
    # if predict_out_temp12 > 30:
    #     reward -= (predict_out_temp12 - 30) / 2.0
    # if predict_out_hum6 > 80:
    #     reward -= (predict_out_hum6 - 80) / 2.0
    # if predict_out_hum12 > 80:
    #     reward -= (predict_out_hum12 - 80) / 2.0

    # # Encourage charging electric vehicle during off-peak hours
    # if time_of_day >= 22 or time_of_day < 6:
    #     reward += current_net_consumption / 10.0

    # # Encourage maintaining indoor temperature within comfortable range
    # if current_indoor_temp >= 18 and current_indoor_temp <= 22:
    #     reward += 1.0

    # # Encourage maintaining tank temperature within optimal range
    # # if current_tank_temp >= 50 and current_tank_temp <= 55:
    # #     reward += 1.0

    # # Encourage energy efficiency during non-occupancy
    # # if not occupancy and current_net_consumption < 0:
    # #     reward += abs(current_net_consumption) / 5.0
    # return reward

def normalization(home):
    """
    A function that returns the normalized list of observations
    :input: PlayerHome object
    :output: list of floats (any length)
    """
    normalized_obs = []
    for name, value in sorted(home.obs_dict.items()):
        if "t_out" in name:
            print("tast1")
            #norm_value = norm_helper(value, np.max(home.home.all_oat), np.min(home.home.all_oat))
        elif "t_in" in name:
            norm_value = norm_helper(value, 22, 18)
        elif "time_of_day" in name:
            norm_value = norm_helper(value, 24, 0)
        elif "ghi" in name:
            print("test")
            #norm_value = norm_helper(value, np.max(home.home.all_ghi), 0)
        else:
            norm_value = value
        normalized_obs += [norm_value]
    return normalized_obs

# 1. Initialize the neural network (called the Q-network) with random weights.
# 2. Observe the current state of the environment.
# 3. Use the Q-network to select an action to take in the current state. This is done by feeding the state into the Q-network and selecting the action with the highest Q-value.
# 4. Take the selected action and observe the resulting reward and next state.
# 5. Store the experience (state, action, reward, next state) in a replay buffer.
# 6. Sample a batch of experiences from the replay buffer and use them to train the Q-network. The Q-network is trained to minimize the difference between the predicted Q-values and the actual Q-values, which are calculated using the Bellman equation.
# 7. Repeat steps 2-6 for a fixed number of timesteps or until convergence.
# In this implementation, the DQN agent uses a dueling architecture, which separates the Q-network into two streams: one for estimating the state value function and one for estimating the advantage function. This allows the agent to better generalize across actions and states.

# Overall, the goal of the DQN agent is to learn an optimal policy for the given environment by maximizing the expected cumulative reward over time.

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size #size of the stae (this would be all of the given keys)
        self.action_size = action_size #size of the aciton (predict 3 continout actions HVAC, water heater on/off, vehicle charge demand array each between 0 and 1)
        #for temporal corelation 
        self.memory = deque(maxlen=2000)  #double ended queue data structure used to store expereince replay memory 
        #gamma is a discount factor used to balance the importance of immediate and future rewards. It is a scalar value between 0 and 1, and represents the relative importance of future rewards to the agent. A high value of gamma (closer to 1) indicates that the agent values future rewards more strongly, whereas a low value of gamma (closer to 0) indicates that the agent prioritizes immediate rewards. The gamma value is multiplied by the future reward estimate during the calculation of the agent's expected return.
        self.gamma = 0.95    # discount rate
        # epsilon is the probability of selecting a random action over the action recommended by the current policy. It is a hyperparameter used to balance exploration and exploitation. When epsilon is high, the agent is more likely to choose random actions, which can help the agent explore new parts of the environment. When epsilon is low, the agent is more likely to choose the action with the highest Q-value, which can help the agent exploit its current knowledge. Epsilon usually starts high at the beginning of training and then decays over time as the agent learns.
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model() # creates a neural network 

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # linear stack of layers
        model = Sequential()
        # adds a fully connected dense layer with 24 usitns to the model, the size of the state, with the activation function REctifein Linear ReLU
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        #adds another layer same as above, good for non-linearity
        model.add(Dense(24, activation='relu'))
        #this layer has the same numebr of neurons as the action space, uses lienar activation funciotn, allowing output to have any value in the aciton range
        model.add(Dense(self.action_size, activation='linear'))
        #compiels the model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    #stores the previous experenicne tuples that the agent has encounteed while exploring the env
    #used for expereince replay: which is an imporntat technique which helps stability and efficiency of the learning process, randomly sampleing a batch of experince tubles and using it to trian neural network
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    #funciton that selects an action given the current state, using an epsilon-greedy stragey to balance exploration and exploitation 

    #if rand number is between 0 or 1 is less the the exploration prob it selects random action between low and high value of each action

    #if the numebr is greater the afents predicts the expected Q value of each action using the neural netowek 
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(low=-1, high=1, size=self.action_size)
        act_values = self.model.predict(state)
        return act_values[0]

    def replay(self, batch_size):
        #get random sample of expereinecs from agents memory buffer
        minibatch = random.sample(self.memory, batch_size)
        #this loops though the experience in the minibatch and etaxts all the info 
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                #If the experience did not result in the agent reaching a terminal state, update the target value using the Q-learning update rule
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            #get the preicted Q value for the current state 
            target_f = self.model.predict(state)
            #update the Q -value of the selecres action with new target value
            target_f[0][action] = target
            #train moden on the staet andupdate Q value 
            self.model.fit(state, target_f, epochs=1, verbose=0)
            # Decrease the exploration rate (epsilon) if it is greater than the minimum epsilon threshold.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




#Pedict 
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------

state_size = 10
action_size = 3
dqn_model = DQNAgent(state_size, action_size)
def predict(home):

    #return [0,0,0]
    """
    Simple rule-based prediction function as a template
    Parameters
    ----------
    home : dragg_comp.player.PlayerHome
        Your home
    Returns
    -------
    list
        List of actions corresponding to hvac, wh, and electric vehicle
    """

    
    state = np.array([home.obs_dict[key] for key in home.obs_dict.keys()])
    state = np.reshape(state, [1, 17])

    print(state)

    # Get the predicted action from the DQN agent
    action = dqn_model.act(state)

    # Map the action values to the appropriate range for each component
    # For the HVAC, map to the range [home.hvac_min, home.hvac_max]
    hvac_action = np.interp(action[0], [-1, 1], [65,78])

    # For the water heater, map to the range [-1, 1]
    wh_action = action[1]

    # For the electric vehicle, map to the range [-5, 5] kW
    ev_action = np.interp(action[2], [-1, 1], [-5, 5])

    # Return the list of actions
    return [hvac_action, wh_action, ev_action]
  
   