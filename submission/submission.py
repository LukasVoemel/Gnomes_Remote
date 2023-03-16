

#------------------------------------PREDICT----------------------------------------------------
def predict(home):
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

    """
    Notes: 
    ------
        at each time stamp t, agent is at certian state s_t, and chooses action a_t to perform 
        run selected action and returns reward to the agent 

        Goal: the goal of the agent is to max the total rewards to get from env. 
        This funtion is called the expected discounted return function 

        agent needs to find optional policy which is probabiliy distriubtion of a given state over actions


        Q-learning algorithm 
            goal is to learn iterativley the optinal Q-value function using Bellman OPtimality Equation
            Store all Q-values in table that we update at each timestep using the Q-learning iteration: 
    """


    """
        This is where the actual policy is 
        returns the predicted actions for next time step 
        action is the actual action choosen by the reinforcement learnign agent at current time stamp 

        policy: state -> action



        traing time is heavily influenced on reward
            if you give reward for almost you can train faster 

        terminal conditions: 
            determine when to reset the enviorment 
    """

    hvac_action = 0  # no action
    wh_action = 0  # no action
    ev_action = 0  # no action
    action = [hvac_action, wh_action, ev_action]

    return action


#-------------------------------------------------------REWARD FUNCTION-----------------------------------

def reward_function(state, action):


    """
        three entries long 
        [-1,1]

        HVAC: [-1,1] = [lowest possible value, highest possible value]
        Water heater: [-1,1] = [water heater off, water on]
            intermediate values will correspond to being on part of the time
            (0) avg power consumption of 50% over 15 min interval
        EV: charge is interpolated between max possible charge and max possible dichagre 

        we take in predict(home) as an array of [-1, 1]
        that is the next action to be predicted

        calualcte reward from current state to next state from the given action
    """

    #get the new prediction based on return (next state)
    

    #current state now 
    hvac_currentState = state[0] #hvac
    water_currentState = state[1] #water
    ev_currentState = state[2] #ev

    #the action to be taken 
    hvac_action = action[0] #hvac
    water_action = action[1] #water
    ev_action = action[2] #ev

    #the state after the action
    #these values are coming in normalized and this must be accounted for
    hvac_newState, water_newState, ev_newState = predict(home)


    #set setpoints for the optimal goal
    #not yet adjusted to the normalized values
    min_hvacTemp = 60
    max_hvacTemp = 80


    hvac_reward = 0
    #try to adjust reward based on how much in the middle the hvac is
    if min_hvacTemp <= hvac_newState <= max_hvacTemp:
        hvac_reward = (hvac_newState - hvac_currentState) / (max_hvacTemp- min_hvacTemp)


    min_waterTemp = 60 #X
    max_waterTemp = 120 #Y
    
    water_reward = 0
    if min_waterTemp <= water_newState <= max_waterTemp:
        water_reward = (water_newState - water_currentState)


    #missing implemetnation of ev


    
    reward = hvac_reward + water_reward

    return reward



predict(home)








