# Torch SB3 QuantilePPO

WIP WIP WIP WIP WIP WIP

simple for now just run

 python -m quantileppobenchmark.atariquantileppo

This repository implements **Quantile Regression**-based reinforcement learning heads and distribution classes, integrated into PPO and World Model architectures.  
It follows the ideas from Dabney et al. for **Distributional RL** and adapts them to continuous and discrete control via PPO.

---

## 📦 Requirements

- Python 3.9+ (recommended: 3.10 or 3.11)
- [PyTorch](https://pytorch.org/)
- Stable-Baselines3
- Gymnasium + Atari environments
- NumPy, TensorBoard, Matplotlib

All dependencies are listed in [`requirements.txt`](./requirements.txt).

---

## 🛠 Setting up a Local Environment

We provide a **cross-platform bootstrap** to avoid cluttering your global Python installation.

### One-time setup
Clone the repo and run:

**Linux/macOS**
```bash
bash setup_env.sh
```

**Windows PowerShell**
```powershell
.\setup_env.ps1
```

The script:
1. Creates a `.venv/` virtual environment if it doesn’t exist.
2. Upgrades `pip`, `setuptools`, and `wheel`.
3. Installs dependencies from `requirements.txt`.
4. Prints the correct activation command for your OS.

---

## ▶ Activating the Environment

After setup, activate the venv:

**Linux/macOS**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

Deactivate anytime with:
```bash
deactivate
```

---

## 🎮 Atari ROM Setup (AutoROM)

If you plan to train Atari environments, install ROMs using:
```bash
AutoROM --accept-license
```
This is required for environments like `PongNoFrameskip-v4`.

---

## 🚀 Training Examples

### Atari Pong with QuantilePPO
```bash
python atariquantileppo.py --env PongNoFrameskip-v4 --timesteps 1_000_000
```

---

## 📂 Project Structure

```
quantile_head.py          # Quantile regression head (IQN-style embedding)
quantile_distribution.py  # Distribution wrapper over quantile outputs
quantile_loss.py          # Quantile regression loss
quantileworldmodelexample.py  # Transformer-based ensemble world model using quantile heads
requirements.txt          # All dependencies
setup_env.sh / setup_env.ps1 / bootstrap_env.py  # Environment bootstrap scripts
```

---

