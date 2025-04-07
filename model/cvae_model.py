import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------
# FiLM block
# -------------------------------------------
class FiLM(nn.Module):
    """
    FiLM block: given label_emb, produce gamma, beta to modulate x
    x shape: (B, C, L)
    label_emb: (B, label_emb_dim)
    """
    def __init__(self, channel_dim, label_emb_dim):
        super().__init__()
        self.film_gen = nn.Linear(label_emb_dim, channel_dim * 2)

    def forward(self, x, label_emb):
        """
        x: (B, C, L)
        label_emb: (B, label_emb_dim)
        output: (B, C, L)
        """
        B, C, L = x.shape
        film_params = self.film_gen(label_emb)  # (B, 2*C)
        film_params = film_params.view(B, 2, C) # (B, 2, C)
        gamma = film_params[:, 0, :]
        beta = film_params[:, 1, :]

        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)    # (B, C, 1)
        return x * gamma + beta


# -------------------------------------------
# Residual Block (Conv1d stride=1)
# -------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class CVAE(nn.Module):
    """
    Conditional Variational AutoEncoder with:
      - FiLM in each decoder layer
      - Residual block in the decoder
      - Final refine conv (no tanh)
      - Latent Classifier
    """

    def __init__(
        self,
        input_channels=10,
        input_length=3000,
        num_classes=5,
        label_emb_dim=16,
        latent_dim=64,
        hidden_dims_encoder=[32, 64, 128],
        # We do 3 upsampling steps => [128, 64, 10]
        hidden_dims_decoder=[128, 64, 10],
    ):
        super(CVAE, self).__init__()

        # ----------------------------------------------------
        # 1) Label Embedding
        # ----------------------------------------------------
        self.label_emb = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, label_emb_dim)
        )

        # ----------------------------------------------------
        # 2) Encoder
        # ----------------------------------------------------
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

        # compute downsampled length
        self.last_encoder_dim = hidden_dims_encoder[-1]
        self.downsampled_length = input_length
        for _ in hidden_dims_encoder:
            self.downsampled_length = (self.downsampled_length + 2 - 4)//2 + 1
        # e.g. 3000->1500->750->375 with 3 layers

        encoder_out_dim = self.last_encoder_dim * self.downsampled_length
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)

        # ----------------------------------------------------
        # 3) Decoder
        # ----------------------------------------------------
        self.decoder_input = nn.Linear(latent_dim + label_emb_dim, encoder_out_dim)

        self.resblock = ResidualBlock(self.last_encoder_dim)

        self.decoder_convs = nn.ModuleList()
        self.decoder_films = nn.ModuleList()

        hidden_dims_decoder_in = hidden_dims_decoder[:-1]
        hidden_dims_decoder_in.insert(0, hidden_dims_encoder[-1])  # match encoder last dim

        for i in range(len(hidden_dims_decoder)):
            out_ch = hidden_dims_decoder[i]
            self.decoder_convs.append(
                nn.ConvTranspose1d(
                    in_channels=hidden_dims_decoder_in[i],
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            self.decoder_films.append(FiLM(out_ch, label_emb_dim))

        # final refine conv => same channels as hidden_dims_decoder[-1] => output must match input_channels
        # we can do a small conv1d to map from "10" to "10" or keep it if you'd like to do more channels
        self.final_refine = nn.Conv1d(hidden_dims_decoder[-1], input_channels, kernel_size=3, padding=1)
        # no tanh => allow output range ~ [-∞, +∞]

        # ----------------------------------------------------
        # 4) Latent Classifier
        # ----------------------------------------------------
        self.latent_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def encode(self, x, labels):
        """
        x: (B, 10, 3000)
        labels: (B, 5)
        """
        label_emb = self.label_emb(labels)  # (B, label_emb_dim)
        B = x.size(0)
        label_emb_expanded = label_emb.unsqueeze(-1).repeat(1, 1, x.size(-1))
        x_cond = torch.cat((x, label_emb_expanded), dim=1)  # (B, 10+label_emb_dim, 3000)

        for conv in self.encoder_convs:
            x_cond = conv(x_cond)

        x_cond = x_cond.view(B, -1)
        mu = self.fc_mu(x_cond)
        logvar = self.fc_logvar(x_cond)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        """
        z: (B, latent_dim)
        labels: (B, 5)
        """
        label_emb_out = self.label_emb(labels)  # (B, label_emb_dim)
        z_cond = torch.cat((z, label_emb_out), dim=1)
        x = self.decoder_input(z_cond)  # shape: (B, encoder_out_dim)

        B = x.size(0)
        x = x.view(B, self.last_encoder_dim, self.downsampled_length)

        # residual block first
        x = self.resblock(x)

        # upsampling + FiLM
        for conv, film in zip(self.decoder_convs, self.decoder_films):
            x = conv(x)
            x = film(x, label_emb_out)
            x = F.leaky_relu(x, 0.2)

        # final refine => (B, input_channels, length=3000)
        x = self.final_refine(x)
        # no tanh => let the net produce arbitrary amplitude
        return x

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        logits = self.latent_classifier(mu)
        return recon, mu, logvar, logits


def loss_function(
    recon_x, x,
    mu, logvar,
    logits, labels,
    kld_weight=1.0,
    alpha=1.0,
    recon_weight=1.0
):
    """
    Enhanced CVAE Loss
    - recon_weight: scale reconstruction MSE to encourage better reconstruction
    """
    # MSE
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # KL
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # classification
    label_ids = labels.argmax(dim=1)
    cls_loss = F.cross_entropy(logits, label_ids)

    loss = recon_weight * recon_loss + kld_weight * kld + alpha * cls_loss
    return loss, recon_loss, kld, cls_loss