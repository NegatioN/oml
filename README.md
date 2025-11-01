## OdinML - oml

Name up for debate

### Supported Pytorch Operations
All ops only with f32 so far
- nn.Linear
- nn.Relu
- plus (scalar and tensor) Unsure if all types of broadcasting works as intended.
- minus (scalar and tensor) Unsure if all types of broadcasting works as intended.
- arange(end)
- accessing constants (also anonymous) defined in forward and __init__
- detach