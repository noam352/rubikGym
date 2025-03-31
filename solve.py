

import numpy as np
import torch
from train_model import MLP
from cube import Cube

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 54 * 2
model = MLP(input_dim=input_dim).to(device)

# Load the saved model state (model.pth)
# model.load_state_dict(torch.load('model.pth', map_location=device))
# model.load_state_dict(torch.load('model_128h_20Steps_noTriples.pth', map_location=device))
# model.load_state_dict(torch.load('model_1mil_256h_20Steps_noTriples.pth', map_location=device))
# model.load_state_dict(torch.load('model_5layers_1mil_256h_20Steps_noTriples.pth', map_location=device))
model.load_state_dict(torch.load('model_epoch_10.pth', map_location=device))
model.eval()
cube = Cube(n=3)
finished_state = cube.get_state()
cube.shuffle()
state = cube.get_state()
s = 0
min_value = 1000
while not np.array_equal(state, finished_state):
    costs = []
    for action in range(6*3):
        cube.step(action)
        new_state = cube.get_state()
        inp = np.concatenate([new_state, finished_state])
        inputs_tensor = torch.from_numpy(inp).float()
        inputs_tensor = inputs_tensor.unsqueeze(0)
        with torch.no_grad():
            value = model(inputs_tensor)
        min_value = min(min_value, value)
        costs.append(value)
        if action % 2 == 0:
            anti_action = action+1
        else:
            anti_action = action-1
        cube.step(anti_action)
        
    epsilon = 0.1
    random = np.random.rand()
    if random < epsilon:
        action = np.random.randint(6*3)
    else:
        action = np.argmin(costs)
    if min(costs) <2:
        cube.render_3d()
    cube.step(action)
    s+=1
    state = cube.get_state()