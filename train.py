import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model.cvae_model import CVAE, loss_function

def train_cvae():
    # =====================================================
    # 1. Prepare Data (Example Only)
    # =====================================================
    # Assume EEG data shape: (7640, 10, 3000)
    # and label shape: (7640, 5)
    # np.random.seed(42)
    eeg_data = np.random.randn(7640, 10, 3000).astype(np.float32)
    labels_data = np.random.randint(0, 2, size=(7640, 5)).astype(np.float32)  # Random 0/1 for demonstration

    # Convert to PyTorch tensors
    eeg_data = torch.from_numpy(eeg_data)
    labels_data = torch.from_numpy(labels_data)

    dataset = TensorDataset(eeg_data, labels_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    # =====================================================
    # 2. Initialize Model
    # =====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(
        input_channels=10,
        input_length=3000,
        num_classes=5,
        label_emb_dim=16,
        latent_dim=64,
        hidden_dims_encoder=[32, 64, 128],
        hidden_dims_decoder=[128, 64, 10]  # Adjusted to output 10 channels
    ).to(device)

    # =====================================================
    # 3. Optimizer Settings
    # =====================================================
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 20
    kl_anneal_start = 0.0
    kl_anneal_end = 1.0
    kl_anneal_epochs = 10  # Number of epochs to linearly ramp up the KL weight

    # =====================================================
    # 4. Training Loop
    # =====================================================
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

        # Compute current KL weight (KL annealing)
        if epoch <= kl_anneal_epochs:
            kld_weight = kl_anneal_start + (kl_anneal_end - kl_anneal_start) * (epoch / float(kl_anneal_epochs))
        else:
            kld_weight = kl_anneal_end

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x, y)
            loss, recon_loss, kld = loss_function(recon, x, mu, logvar, kld_weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kld = total_kld / len(dataloader)

        print(f"Epoch [{epoch}/{epochs}] | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f} | kld_weight: {kld_weight:.4f}")

    # =====================================================
    # 5. Save the model after training
    # =====================================================
    torch.save(model.state_dict(), "cvae_eeg.pth")
    print("Training completed and model saved!")

if __name__ == "__main__":
    train_cvae()
