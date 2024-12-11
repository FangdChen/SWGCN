# SWGCN

This is the official SWGCN implementation for **SWGCN: Synergy Weighted Graph Convolutional Network for Multi-Behavior Recommendation**.

# Download

```shel
git clone https://github.com/FangdChen/SWGCN.git
cd SWGCN
```

# Environment

```shell
conda create -n swgcn python=3.10
conda activate swgcn

pip install -r requirements.txt
```

# Prepare Data

You can prepare the training data by using the following command to reduce the time for each run.

```shell
chmod +x scripts/preparedata.sh

# Prepare beibei dataset
./scripts/preparedata.sh beibei

# Prepare taobao dataset
./scripts/preparedata.sh taobao

# Prepare ijcai dataset
./scripts/preparedata.sh ijcai
```

# Train and Test

You can run the code with the following command.

```shell
chmod +x scripts/run.sh

# Run SWGCN on beibei
./scripts/run.sh beibei

# Run SWGCN on taobao
./scripts/run.sh taobao

# Run SWGCN on ijcai
./scripts/run.sh ijcai
```

