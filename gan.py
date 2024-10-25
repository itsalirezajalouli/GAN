#   Imports
import numpy as np
import torch.nn as nn
import torch.optim as optim

#   Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
latentZDim = 128
batchSize = 128
neurons = 512
outShape = (-1, 28, 28)
numEpochs = 10

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
