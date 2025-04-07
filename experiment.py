import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from model.cvae_model import CVAE, loss_function
from scipy.stats import pearsonr

def load_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load trained model ===
    model = CVAE(
        input_channels=10,
        input_length=3000,
        num_classes=5,
        label_emb_dim=16,
        latent_dim=64,
        hidden_dims_encoder=[32, 64, 128],
        hidden_dims_decoder=[128, 64, 10]
    ).to(device)

    checkpoint_path = "checkpoints/cvae_eeg.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # === Load real data for analysis
    data = np.load('data/ISRUC_S3.npz')
    eeg_data = data['Fold_data'].astype(np.float32)  # shape (10, 764, 10, 3000)
    labels_data = data['Fold_label'].astype(np.float32) # shape (10, 764, 5)

    eeg_data = eeg_data.reshape(-1, 10, 3000)
    labels_data = labels_data.reshape(-1, 5)

    # (Optional) if you used z-score in training, do same here
    # mean_val, std_val = eeg_data.mean(), eeg_data.std()
    # eeg_data = (eeg_data - mean_val)/(std_val + 1e-8)

    eeg_tensor = torch.from_numpy(eeg_data)
    label_tensor = torch.from_numpy(labels_data)
    dataset = TensorDataset(eeg_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    return model, dataloader, device

def visualize_latent_space(model, dataloader, device):
    print("Generating t-SNE latent space plot...")
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            mu, logvar = model.encode(x, y)
            all_mu.append(mu.cpu())
            all_labels.append(y.cpu())

    all_mu = torch.cat(all_mu).numpy()
    all_labels = torch.cat(all_labels).numpy()
    label_ids = all_labels.argmax(axis=1)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    mu_2d = tsne.fit_transform(all_mu)

    plt.figure(figsize=(8, 6))
    for i in range(5):
        idx = (label_ids == i)
        plt.scatter(mu_2d[idx, 0], mu_2d[idx, 1], label=f"Stage {i}", alpha=0.6)
    plt.legend()
    plt.title("t-SNE of Latent Space")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/latent_tsne.png")
    plt.show()
    print("Saved t-SNE plot to results/latent_tsne.png ✅")

def generate_samples(model, device, latent_dim=64):
    print("Generating EEG samples by condition...")
    model.eval()
    os.makedirs("results", exist_ok=True)

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        for i in range(5):
            label = torch.zeros(1, 5).to(device)
            label[0, i] = 1  # one-hot for stage i

            gen = model.decode(z, label)  # shape: (1, 10, 3000)

            plt.figure(figsize=(12, 3))
            plt.plot(gen[0, 0].cpu().numpy())  # just channel 0
            plt.title(f"Generated EEG Sample (Stage {i})")
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.savefig(f"results/generated_stage_{i}.png")
            plt.show()
            print(f"Saved EEG plot for Stage {i} → results/generated_stage_{i}.png ✅")


def evaluate_reconstruction_quality(model, dataloader, device, num_examples_to_plot=3):
    """
    Evaluate model's reconstruction performance:
      - Compute average MSE & Pearson correlation across dataset
      - Plot original vs reconstructed for a few random examples
    """
    print("Evaluating reconstruction quality...")
    model.eval()

    total_mse = 0.0
    total_corr = 0.0
    count = 0

    # We'll store a few examples for visualization:
    examples_data = []
    examples_recon = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            recon, mu, logvar, logits = model(x, y)

            x_np = x.cpu().numpy()
            recon_np = recon.cpu().numpy()

            batch_size = x_np.shape[0]
            for i in range(batch_size):
                # We'll just measure channel 0 for simplicity
                original_0 = x_np[i, 0]
                recon_0 = recon_np[i, 0]

                # 1) MSE
                mse_i = np.mean((original_0 - recon_0)**2)
                total_mse += mse_i

                # 2) Pearson correlation
                corr_i = 0.0
                if (np.std(original_0) > 1e-12) and (np.std(recon_0) > 1e-12):
                    from scipy.stats import pearsonr
                    corr_i, _ = pearsonr(original_0, recon_0)
                total_corr += corr_i

            count += batch_size

            if len(examples_data) < num_examples_to_plot:
                examples_data.append(x_np[0])
                examples_recon.append(recon_np[0])

    avg_mse = total_mse / count
    avg_corr = total_corr / count

    print(f"Average MSE (channel 0): {avg_mse:.6f}")
    print(f"Average Pearson Correlation (channel 0): {avg_corr:.6f}")

    os.makedirs("results", exist_ok=True)
    for i in range(len(examples_data)):
        original = examples_data[i]
        reconst = examples_recon[i]

        fig = plt.figure(figsize=(12, 3))
        plt.plot(original[0], label="Original (ch0)")
        plt.plot(reconst[0], label="Reconstructed (ch0)")
        plt.legend()
        plt.title(f"Reconstruction Example #{i}")
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.savefig(f"results/reconstruction_example_{i}.png")
        plt.show()
        print(f"Saved reconstruction comparison to results/reconstruction_example_{i}.png ✅")

if __name__ == "__main__":
    model, dataloader, device = load_model_and_data()

    # 1) Visualize latent
    visualize_latent_space(model, dataloader, device)

    # 2) Generate some samples
    generate_samples(model, device, latent_dim=64)

    # 3) Evaluate Reconstruction
    evaluate_reconstruction_quality(model, dataloader, device, num_examples_to_plot=3)
