
import torch
from torch import nn

try:
    buf = nn.Buffer(torch.randn(5))
    print("nn.Buffer exists!")
    print(f"Buffer type: {type(buf)}")
    print(f"Buffer requires_grad: {buf.requires_grad}")
except AttributeError:
    print("nn.Buffer does NOT exist.")
except Exception as e:
    print(f"Error creating nn.Buffer: {e}")
