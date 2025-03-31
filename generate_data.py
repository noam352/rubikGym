from cube import Cube
import numpy as np
import pandas as pd
import torch

def generate_cube_data(max_value=20, steps=5000000):
    """
    Generates transition data with three columns: start, stop, and value.
    
    Process:
      1. Initialize and shuffle the cube.
      2. Store the initial state.
      3. For each action (step count from 1 to max_value):
           - Take one action.
           - Get the new state.
           - Record a row for every previously seen state:
               (start_state, new_state, step_count)
         (e.g., after 1st action, record 1 row with value 1;
                after 2nd action, record 2 rows with value 2; etc.)
    Only transitions with a value up to max_value are recorded.
    
    Returns:
      A pandas DataFrame with columns: 'start', 'stop', and 'value'.
    """
    # Create and shuffle the cube
    cube = Cube(n=3)
    cube.shuffle()
    
    # List to hold transitions and a list to store states
    transitions = []
    # Store a copy of the initial flattened state
    states = [cube.get_state().copy()]
    last_three = []
    for i in range(steps):
        not_action = -1
        if len(last_three) == 3 and len(set(last_three)) == 1:
            not_action = last_three[0]
        while True:
            action = cube.action_space.sample()
            if action != not_action:
                break
        last_three.append(action)
        if len(last_three) > 3:
            last_three.pop(0)
        cube.step(action)
        new_state = cube.get_state().copy()
        states.append(new_state)
        step = min(i + 1, max_value)
        for s in range(step):
            transitions.append((states[-s-2], states[-1], s+1))
    
    # Convert the list of transitions to a DataFrame
    df = pd.DataFrame(transitions, columns=['start', 'stop', 'value'])
    return df


df = generate_cube_data()
# df.to_csv('cube_data.csv', index=False)
# Stack all 'start' and 'stop' arrays into single numpy arrays
starts_np = np.stack(df['start'].to_numpy())  # shape: (num_transitions, state_size)
stops_np = np.stack(df['stop'].to_numpy())      # same shape as starts_np
values_np = df['value'].to_numpy()              # shape: (num_transitions,)

# Convert the numpy arrays to torch tensors
starts_tensor = torch.from_numpy(starts_np)
stops_tensor = torch.from_numpy(stops_np)
values_tensor = torch.from_numpy(values_np)

# Option 1: Save as a dictionary of tensors
torch.save({
    'start': starts_tensor,
    'stop': stops_tensor,
    'value': values_tensor
}, 'cube_data_5m_20Steps_noTriples.pt')
