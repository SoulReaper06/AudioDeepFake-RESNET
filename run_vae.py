from models import VanillaVAE
import torch.optim as optim


in_channels = 3  # Assuming input images have 3 channels (e.g., RGB)
latent_dim = 20  # Set the desired dimensionality of the latent space
vae_model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dim)

optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

#Load datasets

def train(train_loader, num_epochs, optimizer, vae_model):
    # Assuming you have a dataset (e.g., DataLoader) named `train_loader`
    for epoch in range(num_epochs):
        for data in train_loader:
            # Get the input data
            inputs, _ = data  # Assuming each data point is a tuple (input, label)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = vae_model(inputs)

            # Calculate the loss
            loss_dict = vae_model.loss_function(*outputs, M_N=1)  # Adjust M_N as needed

            # Backward pass
            loss_dict['loss'].backward()

            # Update weights
            optimizer.step()

            # Print or log the loss if desired
            print(f'Epoch {epoch + 1}, Loss: {loss_dict["loss"].item()}')

def sample(num_samples, vae_model):
    # To generate samples from the trained model
    #num_samples = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generated_samples = vae_model.sample(num_samples=num_samples, current_device=device)
    return generated_samples
    
def generate(test_loader):
    reconstructed_images_list = []
    # To generate reconstructions
    # Assuming you have a DataLoader named `test_loader` containing test data
    for data in test_loader:
        inputs, _ = data  # Assuming each data point is a tuple (input, label)
        reconstructed_images = vae_model.generate(inputs)
        reconstructed_images_list.append(reconstructed_images)
    
    return reconstructed_images_list

