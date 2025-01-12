def get_scenario_map():
    scenario_map = {
        'accelerating_at_crosswalk': [15.0, -3.0],
        'accelerating_at_stop_sign': [15.0, -3.0],
        'accelerating_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'accelerating_at_traffic_light': [15.0, -3.0],
        'accelerating_at_traffic_light_with_lead': [15.0, -3.0],
        'accelerating_at_traffic_light_without_lead': [15.0, -3.0],
        'behind_bike': [15.0, -3.0],
        'behind_long_vehicle': [15.0, -3.0],
        'behind_pedestrian_on_driveable': [15.0, -3.0],
        'behind_pedestrian_on_pickup_dropoff': [15.0, -3.0],
        'changing_lane': [15.0, -3.0],
        'changing_lane_to_left': [15.0, -3.0],
        'changing_lane_to_right': [15.0, -3.0],
        'changing_lane_with_lead': [15.0, -3.0],
        'changing_lane_with_trail': [15.0, -3.0],
        'crossed_by_bike': [15.0, -3.0],
        'crossed_by_vehicle': [15.0, -3.0],
        'following_lane_with_lead': [15.0, -3.0],
        'following_lane_with_slow_lead': [15.0, -3.0],
        'following_lane_without_lead': [15.0, -3.0],
        'high_lateral_acceleration': [15.0, -3.0],
        'high_magnitude_jerk': [15.0, -3.0],
        'high_magnitude_speed': [15.0, -3.0],
        'low_magnitude_speed': [15.0, -3.0],
        'medium_magnitude_speed': [15.0, -3.0],
        'near_barrier_on_driveable': [15.0, -3.0],
        'near_construction_zone_sign': [15.0, -3.0],
        'near_high_speed_vehicle': [15.0, -3.0],
        'near_long_vehicle': [15.0, -3.0],
        'near_multiple_bikes': [15.0, -3.0],
        'near_multiple_pedestrians': [15.0, -3.0],
        'near_multiple_vehicles': [15.0, -3.0],
        'near_pedestrian_at_pickup_dropoff': [15.0, -3.0],
        'near_pedestrian_on_crosswalk': [15.0, -3.0],
        'near_pedestrian_on_crosswalk_with_ego': [15.0, -3.0],
        'near_trafficcone_on_driveable': [15.0, -3.0],
        'on_all_way_stop_intersection': [15.0, -3.0],
        'on_carpark': [15.0, -3.0],
        'on_intersection': [15.0, -3.0],
        'on_pickup_dropoff': [15.0, -3.0],
        'on_stopline_crosswalk': [15.0, -3.0],
        'on_stopline_stop_sign': [15.0, -3.0],
        'on_stopline_traffic_light': [15.0, -3.0],
        'on_traffic_light_intersection': [15.0, -3.0],
        'starting_high_speed_turn': [15.0, -3.0],
        'starting_left_turn': [15.0, -3.0],
        'starting_low_speed_turn': [15.0, -3.0],
        'starting_protected_cross_turn': [15.0, -3.0],
        'starting_protected_noncross_turn': [15.0, -3.0],
        'starting_right_turn': [15.0, -3.0],
        'starting_straight_stop_sign_intersection_traversal': [15.0, -3.0],
        'starting_straight_traffic_light_intersection_traversal': [15.0, -3.0],
        'starting_u_turn': [15.0, -3.0],
        'starting_unprotected_cross_turn': [15.0, -3.0],
        'starting_unprotected_noncross_turn': [15.0, -3.0],
        'stationary': [15.0, -3.0],
        'stationary_at_crosswalk': [15.0, -3.0],
        'stationary_at_traffic_light_with_lead': [15.0, -3.0],
        'stationary_at_traffic_light_without_lead': [15.0, -3.0],
        'stationary_in_traffic': [15.0, -3.0],
        'stopping_at_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_with_lead': [15.0, -3.0],
        'stopping_at_stop_sign_without_lead': [15.0, -3.0],
        'stopping_at_traffic_light_with_lead': [15.0, -3.0],
        'stopping_at_traffic_light_without_lead': [15.0, -3.0],
        'stopping_with_lead': [15.0, -3.0],
        'traversing_crosswalk': [15.0, -3.0],
        'traversing_intersection': [15.0, -3.0],
        'traversing_narrow_lane': [15.0, -3.0],
        'traversing_pickup_dropoff': [15.0, -3.0],
        'traversing_traffic_light_intersection': [15.0, -3.0],
        'waiting_for_pedestrian_to_cross': [15.0, -3.0]
    }

    return scenario_map


def get_filter_parameters(num_scenarios_per_type=20, limit_total_scenarios=None, shuffle=True):
    # nuplan challenge
    scenario_types = [
        'starting_left_turn',
        'starting_right_turn',
        'starting_straight_traffic_light_intersection_traversal',
        'stopping_with_lead',
        'high_lateral_acceleration',
        'high_magnitude_speed',
        'low_magnitude_speed',
        'traversing_pickup_dropoff',
        'waiting_for_pedestrian_to_cross',
        'behind_long_vehicle',
        'stationary_in_traffic',
        'near_multiple_vehicles',
        'changing_lane',
        'following_lane_with_lead',
    ]

    scenario_tokens = None              # List of scenario tokens to include
    log_names = None                     # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None          # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = False           # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = True         # Whether to remove scenarios where the mission goal is invalid
    shuffle                             # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance
