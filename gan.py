#   Imports
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

#   Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
latentZDim = 128
batchSize = 128
neurons = 512
outShape = (-1, 28, 28)
numEpochs = 10

#   Data
trainData = MNIST('../Desktop/mnist/', True, T.ToTensor(), download = True)
trainLoader = DataLoader(trainData, batchSize, True, drop_last = True)

#   Helper Funcs
def fcLayer(inNeurons: int, outNeurons: int, leak: float = 0.1):
    '''`inNeurons:` inputs to the layer
    `outNeurons:` outputs to the layer
    `leak:` leaky reu leak value'''
    return nn.Sequential(
        nn.Linear(inNeurons, outNeurons),
        nn.LeakyReLU(leak),
        nn.LayerNorm(outNeurons)
    )

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape) 

#   Defining Discrimantor and Generator
def simpleGAN(latentZDim: int, neurons: int, outShape, sigmoidG: bool = False,
              leak: float = 0.2):
    '''This function creates a simple GAN and returns it a `tuple` of `(G, D)`
    `latentZDim:` number of latent variables to use as `input` to the `Generator`
    `neurons:` how many hidden neurons to use in each hidden layer
    `outShape:` shape of Generator's `output` and Discrimantor's `input`
    `sigmoidG:` `True` if Generator ends with a sigmoid activation function and `False` if
    it should just  return unbounded activations'''
    G = nn.Sequential(
        fcLayer(latentZDim, neurons, leak),
        fcLayer(neurons, neurons, leak),
        fcLayer(neurons, neurons, leak),
        nn.Linear(neurons, abs(np.prod(outShape))),
        View(outShape) # Reshapes output to what D expect
    )
    if sigmoidG:
        G = nn.Sequential(G, nn.Sigmoid())

    D = nn.Sequential(
        nn.Flatten(),
        fcLayer(abs(np.prod(outShape)), neurons, leak),
        fcLayer(neurons, neurons, leak),
        fcLayer(neurons, neurons, leak),
        nn.Linear(neurons, 1) # D has one output for binary classification
    )
    return G, D

G, D = simpleGAN(latentZDim, neurons, outShape, True)

#   Training Loop (Game)
G.to(device)
D.to(device)

lossFunc = nn.BCEWithLogitsLoss()
realLabel = 1
fakeLabel = 0

optimizerD = optim.AdamW(D.parameters(), lr = 0.0001, betas = (0.0, 0.9)) 
optimizerG = optim.AdamW(G.parameters(), lr = 0.0001, betas = (0.0, 0.9))

def trainingLoop():

    GLosses = []
    DLosses = []

    for _ in tqdm(range(numEpochs)):
        for data, _ in tqdm(trainLoader, leave = False):

            dataReal = data.to(device)

            yReal = torch.full((batchSize, 1), realLabel, dtype = torch.float32, 
                               device = device) 
            yFake = torch.full((batchSize, 1), fakeLabel, dtype = torch.float32, 
                               device = device) 

            #   l(D(x_real), y_real)
            D.zero_grad()
            errDReal = lossFunc(D(dataReal), yReal)
            errDReal.backward()
            
            #   Random vector z is a noise
            z = torch.randn(batchSize, latentZDim, device = device)

            #   l(D(G(z)), y_fake)
            fake = G(z)
            errDFake = lossFunc(D(fake.detach()), yFake)
            errDFake.backward()

            errD = errDReal + errDFake

            optimizerD.step()

            #   l(D(G(z)), y_real)
            G.zero_grad()
            errG = lossFunc(D(fake), yReal)
            errG.backward()

            optimizerG.step()
            GLosses.append(errG.item())
            DLosses.append(errD.item())

# trainingLoop()

#   Inspecting the results (test)
with torch.no_grad():
    noise = torch.randn(batchSize, latentZDim, device = device)
    fakeDigits = G(noise)
    scores = torch.sigmoid(D(fakeDigits))
    fakeDigits = fakeDigits.cpu()
    scores = scores.cpu().numpy().flatten()

#   Plots
def plotGenImgs(fakeDigits, scores = None):
    batchSize = fakeDigits.size(0)
    #   Here we assume we're working with B & W imgs
    fakeDigits = fakeDigits.reshape(-1, fakeDigits.size(-1), fakeDigits.size(-1))
    iMax = int(round(np.sqrt(batchSize)))
    jMax = int(round(np.floor(batchSize / float(iMax))))
    _, axarr = plt.subplots(iMax, jMax, figsize = (10, 10))
    for i in range(iMax):
        for j in range(jMax):
            idx = i * jMax + j
            axarr[i, j].imshow(fakeDigits[idx,:].numpy(), cmap = 'gray',
                               vmin = 0, vmax = 1)
            axarr[i, j].set_axis_off()
            if scores is not None:
                axarr[i, j].text(0.0, 0.5, str(round(scores[idx], 2)), dict(size = 20, 
                                color = 'red'))
    plt.show()

# plotGenImgs(fakeDigits, scores)

#   Trying to solve Mode Collapse
gausGrid = (3, 3)
samplesPer = 1000

X = []
for i in range(gausGrid[0]):
    for j in range(gausGrid[1]):
        z = np.random.normal(0, 0.05, size = (samplesPer, 2))
        #   Shifts samples to have specific x & y axis
        z[:, 0] += i/1.0 - (gausGrid[0] - 1) / 2.0
        z[:, 1] += j/1.0 - (gausGrid[1] - 1) / 2.0
        X.append(z)
X = np.vstack(X)

# plt.figure(figsize = (10, 10))
# sns.kdeplot(x = X[:, 0], y = X[:, 1], shade = True, fill = True, thresh = -0.001)
# plt.show()

toyDataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype = torch.float32))
toyLoader = DataLoader(toyDataset, batch_size = batchSize, shuffle = True,
                       drop_last = True)

def trainWGAN(D, G, loader, latentZDim, epochs = 20, Lambda = 10, device = 'cpu'):
    GLosses = []
    DLosses = []
    G.to(device)
    D.to(device)

    optimizerD = optim.AdamW(D.parameters(), lr = 0.0001, betas = (0.0, 0.9))
    optimizerG = optim.AdamW(G.parameters(), lr = 0.0001, betas = (0.0, 0.9))

    for epoch in tqdm(range(epochs)):
        for data in tqdm(loader, leave = False):
            if isinstance(data, tuple) or len(data) == 2:
                data, classLabel = data
                classLabel = classLabel.to(device)
            elif isinstance(data, tuple) or len(data) == 1:
                data = data[0]
            batchSize = data.size(0)
            D.zero_grad()
            G.zero_grad()
            real = data.to(device)

            DSuccess = D(real)
            noise = torch.randn(batchSize, latentZDim, device = device)
            fake = G(noise) # a fake batch
            DFailure = D(fake)

            epsShape = [batchSize] + [1]*(len(data.shape) - 1)
            eps = torch.rand(epsShape, device = device)
            mixed = eps * real + (1 - eps) * fake
            output = D(mixed)

            grad = torch.autograd.grad(outputs = output, inputs = mixed,
                        grad_outputs = torch.ones_like(output), create_graph = True,
                        retain_graph = True, only_inputs = True, allow_unused = True)[0]

            DGradPenalty = ((grad.norm(2, dim = 1) - 1) ** 2)
            errD = (DFailure - DSuccess).mean() + DGradPenalty.mean() * Lambda
            errD.backward()
            optimizerD.step()

            D.zero_grad()
            G.zero_grad()

            noise = torch.randn(batchSize, latentZDim, device = device)
            output = -D(G(noise))
            errG = output.mean()
            errG.backward()
            optimizerG.step()
            
            DLosses.append(errD.item())
            GLosses.append(errG.item())
    return DLosses, GLosses

# G, D = simpleGAN(latentZDim, 512, (-1, 2))
# trainWGAN(D, G, toyLoader, latentZDim, epochs = 20, device = device)
# G, D = G.eval(), D.eval()
#
# with torch.no_grad():
#     noise = torch.randn(X.shape[0], latentZDim, device = device)
#     fakeSamplesW = G(noise).cpu().numpy()
# plt.figure(figsize = (10, 10))
# ax = sns.kdeplot(x = fakeSamplesW[:, 0], y = fakeSamplesW[:, 1], shade = True,
#                  thresh = -0.001)
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# plt.show()

#   Back to MNIST
latentZDim = 128
outShape = (-1, 1, 28, 28)
# G, D = simpleGAN(latentZDim, neurons, outShape, sigmoidG = True)
# DLosses, GLosses = trainWGAN(D, G, trainLoader, latentZDim, epochs = 40, device = device)
# D = D.eval()
# G = G.eval()
#
# with torch.no_grad():
#     noise = torch.randn(batchSize, latentZDim, device = device)
#     fakeDigits = G(noise)
#     scores = D(fakeDigits)
#     fakeDigits = fakeDigits.cpu()
#     scores = scores.cpu().numpy().flatten()
# plotGenImgs(fakeDigits)
# plt.figure(figsize = (10, 5))
# plt.title('Generator and Discrimantor Loss during Training')
# plt.plot(np.convolve(GLosses, np.ones((100,)) / 100, mode = 'valid'), label = 'G')
# plt.plot(np.convolve(GLosses, np.ones((100,)) / 100, mode = 'valid'), label = 'D')
# plt.xlabel('iterations')
# plt.xlabel('Loss')
# plt.legend()
# plt.show()

# ----------------------------------------------------------------------------------------

#   Convolutional GANs
startSize = 28 // 4
latentChannels = 16
latentDConv = latentChannels * (startSize ** 2)
inShape = (-1, latentChannels, startSize, startSize) # (B, C', H', W')

nFilters = 32
kSize = 5
kSizeT = 4
leak = 0.2

#   Helper Functions
def cnnLayer(inChannels, outChannels, filterSize, whSize, leak =  0.2):
    return nn.Sequential(
        nn.Conv2d(inChannels, outChannels, filterSize, padding = filterSize // 2),
        nn.LeakyReLU(leak),
        nn.LayerNorm([outChannels, whSize, whSize])
    )
#   Transposed convolution to expand in size
def tcnnLayer(inChannels, outChannels, whSize, leak =  0.2):
    return nn.Sequential(
        nn.ConvTranspose2d(inChannels, outChannels, kSizeT, padding = 1, 
                           output_padding = 0, stride = 2),
        nn.LeakyReLU(leak),
        nn.LayerNorm([outChannels, whSize, whSize])
    )

G = nn.Sequential(
    View(inShape), # Changing the latent vector z to desired img shape
    cnnLayer(latentChannels, nFilters, kSize, 28 // 4, leak),
    cnnLayer(nFilters, nFilters, kSize, 28 // 4, leak),
    cnnLayer(nFilters, nFilters, kSize, 28 // 4, leak),
    tcnnLayer(nFilters, nFilters // 2, 28 // 2, leak),
    cnnLayer(nFilters // 2, nFilters // 2, kSize, 28 // 2, leak),
    cnnLayer(nFilters // 2, nFilters // 2, kSize, 28 // 2, leak),
    tcnnLayer(nFilters // 2, nFilters // 4, 28, leak),
    cnnLayer(nFilters // 4, nFilters // 4, kSize, 28, leak),
    cnnLayer(nFilters // 4, nFilters // 4, kSize, 28, leak),
    nn.Conv2d(nFilters // 4, 1, kSize, padding = kSize // 2),
    nn.Sigmoid(),
)

D = nn.Sequential(
    cnnLayer(1, nFilters, kSize, 28, leak),
    cnnLayer(nFilters, nFilters, kSize, 28, leak),
    nn.AvgPool2d(2),
    cnnLayer(nFilters, nFilters, kSize, 28 // 2, leak),
    cnnLayer(nFilters, nFilters, kSize, 28 // 2, leak),
    nn.AvgPool2d(2),
    cnnLayer(nFilters, nFilters, 3, 28 // 4, leak),
    cnnLayer(nFilters, nFilters, 3, 28 // 4, leak),
    nn.AdaptiveAvgPool2d(4),
    nn.Flatten(),
    nn.Linear(nFilters*4**2, 256),
    nn.LeakyReLU(leak),
    nn.Linear(256, 1),
)

DLosses, GLosses = trainWGAN(D, G, trainLoader, latentDConv, epochs = 15, device = device)
D = D.eval()
G = G.eval()
with torch.no_grad():
    noise = torch.randn(batchSize, latentDConv, device = device)
    fakeDigits = G(noise)
    scores = D(fakeDigits)
    fakeDigits = fakeDigits.cpu()
    scores = scores.cpu().numpy().flatten()
plotGenImgs(fakeDigits)

plt.figure(figsize = (10, 5))
plt.title('Conv-WGAN Generator and Discrimantor Loss during Training')
plt.plot(np.convolve(GLosses, np.ones((100,)) / 100, mode = 'valid'), label = 'G')
plt.plot(np.convolve(GLosses, np.ones((100,)) / 100, mode = 'valid'), label = 'D')
plt.xlabel('iterations')
plt.xlabel('Loss')
plt.legend()
plt.show()
