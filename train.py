import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from model.cvae_model import CVAE, loss_function

def train_cvae():
    # =====================================================
    # 1. Prepare Data
    # =====================================================
    data = np.load('data/ISRUC_S3.npz')
    eeg_data = data['Fold_data'].astype(np.float32)   # shape (10, 764, 10, 3000)
    labels_data = data['Fold_label'].astype(np.float32)  # shape (10, 764, 5)

    # reshape
    eeg_data = eeg_data.reshape(-1, 10, 3000)     # (7640, 10, 3000)
    labels_data = labels_data.reshape(-1, 5)      # (7640, 5)

    # (Optional) z-score or min-max
    # mean_val, std_val = eeg_data.mean(), eeg_data.std()
    # eeg_data = (eeg_data - mean_val)/(std_val + 1e-8)

    eeg_data = torch.from_numpy(eeg_data)
    labels_data = torch.from_numpy(labels_data)

    dataset = TensorDataset(eeg_data, labels_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    # =====================================================
    # 2. Init Model
    # =====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(
        input_channels=10,
        input_length=3000,
        num_classes=5,
        label_emb_dim=16,
        latent_dim=64,
        hidden_dims_encoder=[32, 64, 128],
        hidden_dims_decoder=[128, 64, 10]
    ).to(device)

    # =====================================================
    # 3. Optimizer & Hyperparams
    # =====================================================
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    epochs = 150
    kl_anneal_start = 0.0
    kl_anneal_end = 1.0
    kl_anneal_epochs = 10
    alpha = 1.0         # classifier loss weight
    recon_weight = 5.0  # scale reconstruction MSE

    os.makedirs("checkpoints", exist_ok=True)

    # =====================================================
    # 4. Training Loop
    # =====================================================
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        total_cls = 0.0

        # linearly ramp up the KL weight
        if epoch <= kl_anneal_epochs:
            kld_weight = kl_anneal_start + (kl_anneal_end - kl_anneal_start)*(epoch/kl_anneal_epochs)
        else:
            kld_weight = kl_anneal_end

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            recon, mu, logvar, logits = model(x, y)
            loss, recon_loss, kld, cls_loss = loss_function(
                recon, x, mu, logvar, logits, y,
                kld_weight=kld_weight,
                alpha=alpha,
                recon_weight=recon_weight
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
            total_cls += cls_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kld = total_kld / len(dataloader)
        avg_cls = total_cls / len(dataloader)

        print(f"Epoch [{epoch}/{epochs}] "
              f"| Loss: {avg_loss:.4f} "
              f"| Recon: {avg_recon:.4f} "
              f"| KLD: {avg_kld:.4f} "
              f"| CLS: {avg_cls:.4f} "
              f"| kld_weight: {kld_weight:.4f} ")

    # =====================================================
    # 5. Save Model
    # =====================================================
    model_save_name = "checkpoints/cvae_eeg_" + str(epochs) + ".pth"
    torch.save(model.state_dict(), model_save_name)
    print("Training completed and model saved at " + model_save_name)

if __name__ == "__main__":
    train_cvae()
