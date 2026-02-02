import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
import numpy as np
import torch.nn.functional as F


class ViT(nn.Module):
    def __init__(self, patch_size=16, img_size=224, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        init_values = None

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
    
    def forward(self, x, ids_keep=None):
        x = self.patch_embed(x)
        B, N, D = x.shape
        x = x + self.pos_embed[:, 1:, :]

        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)


def vit_base():
    model = ViT(embed_dim=768, depth=12, num_heads=12)
    return model

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#
# SSL ALGORITHMS
#
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    def __init__(self, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        patch_size = 16
        embed_dim = 768
        in_chans = 3

        # --------------------------------------------------------------------------
        num_patches = 196
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]
        return x


class MAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.head = decoder

    def patchify(self, imgs):
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def random_masking(self, N, mask_ratio, device):
        L = 196  # always use 196 patches per image
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        N = imgs.shape[0]
        ids_keep, mask, ids_restore = self.random_masking(N, mask_ratio, imgs.device)

        encodings = self.encoder(imgs, ids_keep=ids_keep)
        pred = self.head(encodings, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss


class MoCoV3(nn.Module):
    def __init__(self, encoder, T):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(768, int(4*768)),
            nn.GELU(),
            nn.Linear(int(4*768), 768)
        )
        self.T = T

    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]

        labels = torch.arange(N, dtype=torch.long, device=q.device)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, imgs_1, imgs_2):
        q = self.head(self.encoder(imgs_1)[:, 0, :])

        with torch.no_grad():
            k = self.encoder(imgs_2)[:, 0, :]

        return self.contrastive_loss(q, k)


class MMCR(nn.Module):
    def __init__(self, encoder, do_local):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(768, int(4*768)),
            nn.GELU(),
            nn.Linear(int(4*768), 768)
        )
        self.do_local = do_local

    def mmcr_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z_local = torch.stack([z1, z2], dim=-1)
        batch_size = z_local.shape[0]

        centroids = torch.mean(z_local, dim=-1)
        global_nuc = torch.linalg.svdvals(centroids).sum() / batch_size

        if self.do_local:
            local_nuc = torch.linalg.svdvals(z_local).sum()
            loss = local_nuc / batch_size - global_nuc
        else:
            loss = -global_nuc
        
        return loss

    def forward(self, imgs_1, imgs_2):
        cls_1 = self.head(self.encoder(imgs_1)[:, 0, :])

        with torch.no_grad():
            cls_2 = self.head(self.encoder(imgs_2)[:, 0, :])

        return self.mmcr_loss(cls_1, cls_2)


def fetch_model_for_inter_training(cfg, save_dir):
    assert cfg.algo in ["mmcr", "mae", "mocov3"]

    encoder = vit_base()      

    if cfg.algo == "mae":
        decoder = MAEDecoder()
        model = MAE(encoder, decoder)    

        decoder_weights = torch.load(f"{save_dir}/mae_decoder.pth", weights_only=True, map_location="cpu")
        model.head.load_state_dict(decoder_weights, strict=True)

    elif cfg.algo == "mmcr":
        model = MMCR(encoder, cfg.do_local)

    elif cfg.algo == "mocov3":
        model = MoCoV3(encoder, cfg.T)

    encoder_weights = torch.load(f"{save_dir}/mae_encoder.pth", weights_only=True, map_location="cpu") 
    model.encoder.load_state_dict(encoder_weights, strict=True)

    return model