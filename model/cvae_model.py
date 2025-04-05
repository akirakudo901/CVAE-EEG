import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    """
    Conditional Variational AutoEncoder for EEG Data
    Data shape: (batch_size, 10, 3000)
    Label shape: (batch_size, 5) (one-hot or multi-hot for 5 sleep stages)
    """
    def __init__(
        self,
        input_channels=10,
        input_length=3000,
        num_classes=5,
        label_emb_dim=16,
        latent_dim=64,
        hidden_dims_encoder=[32, 64, 128],
        hidden_dims_decoder=[128, 64, 10]  # Adjusted: final layer outputs 10 channels directly
    ):
        super(CVAE, self).__init__()

        # ----------------------------------------------------
        # Label embedding
        # ----------------------------------------------------
        self.label_emb = nn.Linear(num_classes, label_emb_dim)

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------
        # Fuse label embedding with the input by broadcasting along the time dimension
        self.encoder_convs = nn.ModuleList()
        in_channels = input_channels + label_emb_dim

        for h_dim in hidden_dims_encoder:
            self.encoder_convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = h_dim

        # Calculate downsampled length after repeated convolutions
        self.last_encoder_dim = hidden_dims_encoder[-1]
        self.downsampled_length = input_length
        for _ in hidden_dims_encoder:
            # Formula: L_out = (L_in + 2*padding - (kernel_size-1) -1) // stride + 1
            self.downsampled_length = (self.downsampled_length + 2 - 4) // 2 + 1

        encoder_out_dim = self.last_encoder_dim * self.downsampled_length
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------
        # Map (z + label_emb) to flattened decoder input
        self.decoder_input = nn.Linear(latent_dim + label_emb_dim, encoder_out_dim)

        # Build transposed convolution blocks
        self.decoder_convs = nn.ModuleList()
        # Prepare the list of in_channels for each transposed conv layer
        hidden_dims_decoder_in = hidden_dims_decoder[:-1]
        hidden_dims_decoder_in.insert(0, hidden_dims_encoder[-1])  # First input channel from encoder's output

        for i in range(len(hidden_dims_decoder)):
            self.decoder_convs.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=hidden_dims_decoder_in[i],
                        out_channels=hidden_dims_decoder[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0
                    ),
                    nn.BatchNorm1d(hidden_dims_decoder[i]),
                    nn.LeakyReLU(0.2),
                )
            )
        # Note: 3 layers of upsampling:
        # 375 -> 750 -> 1500 -> 3000, and final output channels = 10.

    def encode(self, x, labels):
        """
        x shape: (batch_size, 10, 3000)
        labels shape: (batch_size, 5)
        """
        # Get label embedding and expand along time dimension
        label_emb = self.label_emb(labels)  # (batch_size, label_emb_dim)
        batch_size = x.size(0)
        label_emb_expanded = label_emb.unsqueeze(-1).repeat(1, 1, x.size(-1))
        x_cond = torch.cat((x, label_emb_expanded), dim=1)  # (batch_size, 10+label_emb_dim, 3000)

        # Pass through convolutional encoder layers
        for conv in self.encoder_convs:
            x_cond = conv(x_cond)

        x_cond = x_cond.view(x_cond.size(0), -1)
        mu = self.fc_mu(x_cond)
        logvar = self.fc_logvar(x_cond)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        """
        z shape: (batch_size, latent_dim)
        labels shape: (batch_size, 5)
        """
        label_emb = self.label_emb(labels)  # (batch_size, label_emb_dim)
        z_cond = torch.cat((z, label_emb), dim=1)  # (batch_size, latent_dim + label_emb_dim)
        x = self.decoder_input(z_cond)  # (batch_size, encoder_out_dim)
        batch_size = x.size(0)
        # Reshape to (batch_size, last_encoder_dim, downsampled_length)
        x = x.view(batch_size, self.last_encoder_dim, self.downsampled_length)

        # Pass through transposed convolution layers
        for conv in self.decoder_convs:
            x = conv(x)
        return x

    def forward(self, x, labels):
        """
        Forward pass: Encode -> Reparameterize -> Decode -> Reconstruction
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar, kld_weight=1.0):
    """
    CVAE Loss = Reconstruction Loss + KL Divergence
    recon_x, x shape: (batch_size, 10, 3000)
    mu, logvar shape: (batch_size, latent_dim)
    kld_weight: weight for the KL term (e.g., for KL annealing)
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kld_weight * kld
    return loss, recon_loss, kld
