# QuantilePPO

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
Windows
```ps1
.\setup_env.ps1
```