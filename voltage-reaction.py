







# NEURAL-NETWORK
gatgnn1 = GATGNN(heads, neurons=64, nl=4,global_attention='composition', edge_format=data_src)
gatgnn2 = GATGNN(heads, neurons=64, nl=4,global_attention='composition', edge_format=data_src)
net     = PREDICTOR(128,gatgnn1,gatgnn2,neurons=128).to(device)
