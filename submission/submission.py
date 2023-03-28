


def reward(home):
    """
    Reward function
    :input: PlayerHome object
    :output: float
    """
    print(home.obs_dict)
    reward = home.obs_dict['my_demand'] / 3.5
    return reward

def normalization(home):
    """
    A function that returns the normalized list of observations
    :input: PlayerHome object
    :output: list of floats (any length)
    """
    normalized_obs = []
    for name, value in sorted(home.obs_dict.items()):
        if "t_out" in name:
            norm_value = norm_helper(value, np.max(home.home.all_oat), np.min(home.home.all_oat))
        elif "t_in" in name:
            norm_value = norm_helper(value, 22, 18)
        elif "time_of_day" in name:
            norm_value = norm_helper(value, 24, 0)
        elif "ghi" in name:
            norm_value = norm_helper(value, np.max(home.home.all_ghi), 0)
        else:
            norm_value = value
        normalized_obs += [norm_value]
    return normalized_obs
   

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

    
    hvac_action = 0  # no action
    wh_action = 0  # no action
    ev_action = 0  # no action
    action = [hvac_action, wh_action, ev_action]

     # adjust hvac setpoints
    if home.obs_dict["time_of_day"] >= 6 and home.obs_dict["time_of_day"] < 8:
        hvac_action = 20  # set HVAC setpoint to 20°C during morning peak
    elif home.obs_dict["time_of_day"] >= 16 and home.obs_dict["time_of_day"] < 18:
        hvac_action = 25  # set HVAC setpoint to 25°C during afternoon peak

    # cycle water heater based on current temperature
    if home.obs_dict["t_wh"] < 50:
        wh_action = 1  # turn on water heater if temperature is too low
    elif home.obs_dict["t_wh"] > 60:
        wh_action = -1  # turn off water heater if temperature is too high

    # charge electric vehicle during off-peak hours
    if home.obs_dict["time_of_day"] >= 22 or home.obs_dict["time_of_day"] < 6:
        ev_action = 1  # charge electric vehicle during off-peak hours

    action = [hvac_action, wh_action, ev_action]



    #action should be iterable 
    return action