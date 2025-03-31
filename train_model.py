import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def get_data(filename):
    # -----------------------------
    # 1. Load the Pre-Saved Data
    # -----------------------------
    # data = torch.load('cube_data.pt')
    # data = torch.load('cube_data_20Steps_noTriples.pt')
    # filename = ""
    data = torch.load(filename)
    # Assume data is a dictionary with keys 'start', 'stop', 'value'
    # start and stop are tensors with shape (N, state_dim)
    # value is a tensor with shape (N,)
    starts_tensor = data['start']  # shape: (N, state_dim)
    stops_tensor  = data['stop']   # shape: (N, state_dim)
    values_tensor = data['value']  # shape: (N,)

    # Convert to float (and add a dimension for values so they are (N, 1))
    starts_tensor = starts_tensor.float()
    stops_tensor = stops_tensor.float()
    values_tensor = values_tensor.float().unsqueeze(1)

    # -----------------------------
    # 2. Create Input/Label Pairs
    # -----------------------------
    # Concatenate start and stop states along the feature dimension
    inputs_tensor = torch.cat([starts_tensor, stops_tensor], dim=1)
    # The label remains the value tensor

    # Create a TensorDataset
    dataset = TensorDataset(inputs_tensor, values_tensor)

    # -----------------------------
    # 3. Train-Test Split and DataLoaders
    # -----------------------------
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, inputs_tensor

# -----------------------------
# 4. Define the MLP Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

def main():
    input_file = ""
    output_file = ""
    train_loader, test_loader, inputs_tensor = get_data(input_file)

    # The input dimension is the combined dimension of start and stop states
    input_dim = inputs_tensor.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=input_dim).to(device)

    # -----------------------------
    # 5. Loss, Optimizer, and Training
    # -----------------------------
    criterion = nn.MSELoss()  # For regression; for classification, consider nn.CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_labels in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item() * batch_inputs.size(0)
        test_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        # Save the model state dictionary
        
        torch.save(model.state_dict(), output_file)

# if __name__ == '__main__':
    # main()

