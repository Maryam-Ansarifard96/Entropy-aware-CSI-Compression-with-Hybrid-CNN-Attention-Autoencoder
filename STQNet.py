import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F
import scipy.io as sio 
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from collections import OrderedDict
from collections import Counter, namedtuple
import heapq

img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
encoded_dim = 512 #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
img_size = 32
num_heads = 4 # for multi-head attention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
depth = 1 # Number of STB 
qkv_bias=True
window = 8 # window size for LSA
envir = 'outdoor'

class GroupAttention(nn.Module):

    def __init__(self, num_heads=4, qkv_bias=False):
        super(GroupAttention, self).__init__()

        self.num_heads = num_heads
        head_dim = img_size // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(img_size, img_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(img_size, img_size)
        self.ws = window

    def forward(self, x):
        B, C, H, W = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, C, h_group, self.ws, W)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, C, total_groups, -1, 3, self.num_heads, self.ws // self.num_heads)
        qkv = qkv.permute(4, 0, 1, 2, 5, 3, 6)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        return x

class GlobalAttention(nn.Module):

    def __init__(self, num_heads=4, qkv_bias=False):
        super().__init__()

        self.dim = img_size
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim//window, self.dim//window * 2, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        self.sr = nn.Conv2d(2, 2, kernel_size=window, stride=window)
        self.norm = nn.LayerNorm(self.dim//window)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, -1, self.dim//window, self.dim//window).permute(0,1,3,2,4)
        x_ = self.sr(x).reshape(B, C, -1, self.dim//window, self.dim//window)
        x_ = self.norm(x_)
        kv = self.kv(x_).reshape(B, C, -1, 2, self.dim//window, self.dim//window).permute(3,0,1,4,2,5)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)

        return x

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.cc1 = nn.Linear(img_size, img_size)
        self.cc2 = nn.Linear(img_size, img_size)
        self.act = nn.GELU()

    def forward(self, x):

        x = self.cc1(x)
        x = self.act(x)
        x = self.cc2(x)

        return x


class WTL(nn.Module):
    def __init__(self, num_heads, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(img_size, eps=1e-6)
        self.attn1 = GroupAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
        )
        self.attn2 = GlobalAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm4 = nn.LayerNorm(img_size, eps=1e-6)
        self.mlp1 = MLP()
        self.mlp2 = MLP()

    def forward(self, x):

        x = x + self.attn1(self.norm1(x))
        x = x + self.mlp1(self.norm2(x))
        x = x + self.attn2(self.norm3(x))
        x = x + self.mlp2(self.norm4(x))

        return x

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            img_size=img_size,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
    ):
        super().__init__()


        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.conv1 = nn.Conv2d(2,16, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(16,2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(2*img_size*img_size, encoded_dim)

    def forward(self, x):

        n_samples = x.shape[0]
        x = self.conv1(x)
        x = self.conv5(x)
        X = x 

        for block in self.blocks:
            x = block(x)
        x = self.norm3(x)
        x = self.convT(x) 
        x = X + self.conv4(x)
        x = self.norm2(x)
        x = x.reshape(n_samples,2*img_size*img_size)
        x = self.fc(x)
        return x


class Decoder(nn.Module):   
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(encoded_dim, img_channels*img_size*img_size)
        self.act = nn.Sigmoid()
        self.conv5 = nn.Conv2d(2,2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)

        self.dense_layers = nn.Sequential(
            nn.Linear(encoded_dim, img_total)
        )

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)

    def forward(self, x):
        img = self.dense_layers(x)
        img = img.view(-1, img_channels, img_height, img_width)

        out = self.decoder_feature(img)
        x = self.conv5(img)

        for block in self.blocks:
            x = block((x+out))

        x = self.norm2(x)
        x = self.convT(x)
        x = self.conv4(x) 

        for block in self.blocks:
            x = block((x+out))

        x = self.norm3(x)

        x = self.act(x) 

        return x
class EntropyBottleneck(nn.Module):
    def __init__(self, dim):
        super(EntropyBottleneck, self).__init__()
        self.means = nn.Parameter(torch.zeros(dim))
        self.log_scales = nn.Parameter(torch.zeros(dim))  # ensures positive std

    def _standardized_cumulative(self, x):
        # Standard Gaussian CDF
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def forward(self, z_q):
        """
        z_q: quantized latents, shape (B, D)
        Returns: estimated entropy in bits per sample
        """
        scales = torch.exp(self.log_scales).clamp(min=1e-9)
        centered = z_q - self.means  # (B, D)
        upper = (centered + 0.5) / scales
        lower = (centered - 0.5) / scales
        cdf_upper = self._standardized_cumulative(upper)
        cdf_lower = self._standardized_cumulative(lower)

        # Probability mass between quantization bins
        probs = (cdf_upper - cdf_lower).clamp(min=1e-9)

        # Estimated bits (entropy)
        entropy = -torch.sum(torch.log2(probs), dim=1).mean()  # bits per sample
        return entropy
encoder = Encoder()
encoder.to(device)
#print(encoder)

decoder = Decoder()
decoder.to(device)
#print(decoder)
entropy_bottleneck = EntropyBottleneck(dim=encoded_dim).to(device)

print('Data loading begins.....')

if envir == 'indoor':
    mat = sio.loadmat('/cost2100_dataset/DATA_Htrainin.mat') 
    x_train = mat['HT'] 
    mat = sio.loadmat('/cost2100_dataset/DATA_Hvalin.mat')
    x_val = mat['HT'] 
    mat = sio.loadmat('/cost2100_dataset/DATA_Htestin.mat')
    x_test = mat['HT'] 

elif envir == 'outdoor':
    mat = sio.loadmat('/cost2100_dataset/DATA_Htrainout.mat') 
    x_train = mat['HT'] 
    mat = sio.loadmat('/cost2100_dataset/DATA_Hvalout.mat')
    x_val = mat['HT'] 
    mat = sio.loadmat('/cost2100_dataset/DATA_Htestout.mat')
    x_test = mat['HT']

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))

print('Data loading done!')


x_train = x_train.to(device, dtype=torch.float)

x_test = x_test.to(device, dtype=torch.float)

#%% -------- Normalizer ---------
def normalize(x):
    x_min = x.min(dim=1, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0]
    return 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1, x_min, x_max
# -------- Quantization with STE --------
class MuLawQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_min, x_max, mu=255, bits=2):
        # μ-law quantization/ Dequantization/ Denormalize
        x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu, dtype=x.dtype, device=x.device))
        levels = 2 ** bits
        x_scaled = (x_mu + 1) / 2
        indices = torch.clamp((x_scaled * (levels - 1)).round(), 0, levels - 1).to(torch.int)

        # ***** You should apply it after entropy decoding *****
        x_scaled_back = indices.float() / (levels - 1)
        x_mu_back = x_scaled_back * 2 - 1
        x_recon = torch.sign(x_mu_back) * (1.0 / mu) * ((1 + mu) ** torch.abs(x_mu_back) - 1)
        x_dequant = (x_recon + 1) / 2 * (x_max - x_min + 1e-8) + x_min

        # Uniform quantization/ Dequantization/ Denormalize
        # levels = 2 ** bits
        # x_scaled = (x + 1) / 2
        # indices = torch.clamp((x_scaled * (levels - 1)).round(), 0, levels - 1).to(torch.int)
        # x_scaled_back = indices.float() / (levels - 1)
        # x_mu_back = x_scaled_back * 2 - 1
        # x_dequant = (x_mu_back + 1) / 2 * (x_max - x_min + 1e-8) + x_min

        return x_dequant, indices

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        # Straight-through estimator: identity gradient
        return grad_output1, None, None, None, None
    
def mu_law_quantize_ste(x, x_min, x_max, mu=255, bits=2, return_indices=False):
    x_dequant, indices = MuLawQuantizeSTE.apply(x, x_min, x_max, mu, bits)
    if return_indices:
        return x_dequant, indices
    return x_dequant
#%%
def train_autoencoder(uncompressed_images, opt_enc, opt_dec, opt_entropy, lambda_entropy=1e-3): # lambda_entropy=1e-3
    opt_enc.zero_grad()
    opt_dec.zero_grad()
    opt_entropy.zero_grad()

    compressed_data = encoder.forward(uncompressed_images)
    # === Quantization with STE ===
    normalized, x_min, x_max = normalize(compressed_data)
    dequantized_data = mu_law_quantize_ste(normalized, x_min, x_max)

    reconstructed_images = decoder.forward(dequantized_data)

    # === Compute losses ===
    mse_loss_fn = nn.MSELoss()
    mse = mse_loss_fn(uncompressed_images, reconstructed_images)

    entropy_loss = entropy_bottleneck(normalized)

    total_loss = mse + lambda_entropy * entropy_loss
    total_loss.backward()

    opt_enc.step()
    opt_dec.step()
    opt_entropy.step()
    return total_loss.item()

def fit(epochs, lr, start_idx=1):
    
    losses_dec = []
    losses_auto = []

    opt_enc = Adam(encoder.parameters(), lr, betas=(0.5, 0.999))
    opt_dec = Adam(decoder.parameters(), lr, betas=(0.5, 0.999))
    opt_entropy = Adam(entropy_bottleneck.parameters(), lr)

    reps = int(len(x_train) / (batch_size)) # Number of batches. Note that len(x_train) returns the first dimension which is number of samples

    for epoch in range(epochs):
        x_train_idx = torch.randperm(x_train.size()[0])
        for i in range(reps):
            loss_auto= train_autoencoder(x_train[x_train_idx[i*batch_size:(i+1)*batch_size]], opt_enc, opt_dec, opt_entropy)
            if i % 600 == 0:               
                print('epoch',epoch+1,'/',epochs,'batch:',i+1,'/',reps, "loss_auto: {:.12f}".format(loss_auto))
            losses_auto.append(loss_auto)

    return losses_auto

epochs = 1000
lr = 0.001
batch_size = 200
print('training starts.....')
losses_auto = fit(epochs, lr)

plt.figure(figsize=(10,5))
plt.title("autoencoder and aecoder Loss During Training")
plt.plot(losses_auto,label="AE")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
# del losses_auto
# del x_train
#n = 10
#%% Huffman Encode and Decode
class Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):  # Required for heapq
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = Counter(data)
    heap = [Node(freq, sym) for sym, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.freq + right.freq, None, left, right)
        heapq.heappush(heap, merged)

    return heap[0]

def build_codebook(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_codebook(node.left, prefix + "0", codebook)
        build_codebook(node.right, prefix + "1", codebook)
    return codebook

def entropy_encode(tensor_int):
    data = tensor_int.view(-1).tolist()
    tree = build_huffman_tree(data)
    codebook = build_codebook(tree)
    encoded = ''.join(codebook[sym] for sym in data)
    return encoded, tree, tensor_int.shape  # Save tree + shape for decoding

def entropy_decode(encoded_str, tree, shape, device):
    result = []
    node = tree
    for bit in encoded_str:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            result.append(node.symbol)
            node = tree
    tensor = torch.tensor(result, dtype=torch.int32, device=device).view(shape)
    return tensor
#%% ----- Evaluating SNR-NMSE-Bits -----
# def nmse(x, x_hat):
#     power = torch.sum(x ** 2, dim=(1, 2, 3))
#     mse = torch.sum((x - x_hat) ** 2, dim=(1, 2, 3))
#     return torch.mean(mse / power).item()

# def evaluate_nmse_vs_snr(bits_list, snr_db_range, test_data):
#     nmse_results = {bits: [] for bits in bits_list}

#     with torch.no_grad():
#         for snr_db in snr_db_range:
#             snr_linear = 10 ** (snr_db / 10)
#             for bits in bits_list:
#                 total_nmse = 0
#                 batches = 0
#                 for i in range(0, len(test_data), batch_size):
#                     x = test_data[i:i+batch_size]
#                     compressed = encoder(x)
#                     normalized, x_min, x_max = normalize(compressed)
                    
#                     # Add AWGN noise to simulate channel
#                     signal_power = normalized.pow(2).mean()
#                     noise_power = signal_power / snr_linear
#                     noise = torch.randn_like(normalized) * torch.sqrt(noise_power)
#                     noisy_normalized = normalized + noise
                    
#                     # Quantize + Decode
#                     dequantized = mu_law_quantize_ste(noisy_normalized, x_min, x_max, bits=bits)
#                     recon = decoder(dequantized)
                    
#                     total_nmse += nmse(x, recon)
#                     batches += 1
#                 nmse_results[bits].append(total_nmse / batches)
    
#     return nmse_results
# snr_range_db = list(range(0, 21, 2))  # 0 to 20 dB in 2 dB steps
# bit_widths = [2, 4, 6, 8]  # Choose bit-depths to evaluate

# nmse_data = evaluate_nmse_vs_snr(bit_widths, snr_range_db, x_test)

# Plot
# plt.figure(figsize=(10,6))
# for bits in bit_widths:
#     plt.plot(snr_range_db, 10 * np.log10(nmse_data[bits]), label=f"{bits} bits")

# plt.xlabel("SNR (dB)")
# plt.ylabel("NMSE (dB)")
# plt.title("NMSE vs SNR for Different Quantization Bit Depths")
# plt.legend()
# plt.grid(True)
# plt.show()

#%% ----- Testing the Model -----
with torch.no_grad():
    latent = encoder.forward(x_test[0:1000,:,:,:]) # Encode
    latent_norm, x_min, x_max = normalize(latent)       # Normalize
    quantized, quantized_indices = mu_law_quantize_ste(latent_norm, x_min, x_max, return_indices=True) # Quantize
    
    # Entropy encode
    bitstream, tree, shape = entropy_encode(quantized_indices)
    print(f"Compressed size: {len(bitstream)} bits")

    # 1. Entropy decode: recover quantized int values
    decoded_int = entropy_decode(bitstream, tree, shape, device=device)

    # Reverse μ-law dequantization/ denormalization
    mu =255
    levels = 4
    decoded_float = decoded_int.float() / (levels - 1)
    x_mu_back = decoded_float * 2 - 1
    x_recon = torch.sign(x_mu_back) * (1.0 / mu) * ((1 + mu) ** torch.abs(x_mu_back) - 1)
    reconstructed_latent = (x_recon + 1) / 2 * (x_max - x_min + 1e-8) + x_min

    x_hat = decoder.forward(reconstructed_latent)

x_test_in = x_test[0:1000,:,:,:].to(device)
x_hat_in = x_hat.to(device)


x_test_in = x_test_in.cpu().numpy()
x_hat_in = x_hat_in.cpu().numpy()

x_test_real = np.reshape(x_test_in[:, 0, :, :], (len(x_test_in), -1))
x_test_imag = np.reshape(x_test_in[:, 1, :, :], (len(x_test_in), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat_in[:, 0, :, :], (len(x_hat_in), -1))
x_hat_imag = np.reshape(x_hat_in[:, 1, :, :], (len(x_hat_in), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)


print("NMSE is ", 10*math.log10(np.mean(mse/power)))

