This file explains how to create a Jupyter notebook for interactive visualization.

1. Ensure you have matplotlib and jupyter installed:
   pip install matplotlib jupyter

2. Launch notebook:
   jupyter notebook

3. In a new notebook cell, you can import and use the helpers:
```python
from atlantean_engine import AtlanteanFieldNetAdvanced, dodecagonal_mask, spectral_target_loss
net = AtlanteanFieldNetAdvanced(H=64, W=64)
# run a few steps and visualize fields
out = net(input_spikes=None, pde_steps=4, dt=0.04)
fields = out['fields'].cpu().detach().numpy()
# use matplotlib to show fields and spectrum