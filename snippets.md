### Seed everything

```python
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Cuda if available

```python
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

### Get optimizer params for no weight decay

```python
def get_optimizer_param_groups(self, m, **optimizer_kwargs):
    no_decay_params = []
    decay_params = []
    for n, p in m.named_parameters():
        if n.endswith('.bias') or n.endswith('bn.weight'):
            # No weight decay for bias layers, and bath norm layers.
            no_decay_params.append(p)
        else:
            # Normal weight decay for the rest of the network.
            decay_params.append(p)
    return [
        {'params': no_decay_params, **{k: (0 if k == 'weight_decay' else v) for k, v in optimizer_kwargs.items()}},
        {'params': decay_params, **optimizer_kwargs},
    ]
```