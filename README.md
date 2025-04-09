
# LiLoDriver: A Lifelong Learning Framework for Closed-Loop Motion Planning in Long-tail Autonomous Driving Scenarios

<p align="center">
  <img src="assets/lilodriver_pipeline.png" width="80%" alt="LiLoDriver pipeline overview"/>
</p>

## 🧠 Introduction

**LiLoDriver** is a lifelong learning framework for closed-loop motion planning in long-tail autonomous driving scenarios. It leverages **Large Language Models (LLMs)** for adaptive planner selection and **memory-based knowledge retrieval**, enabling continual scene understanding and strategy evolution during inference without the need for retraining.

Our method integrates four core modules:
- **Environment and Perception**: Fuses map and historical trajectory data.
- **Scene Encoder**: Extracts scene embeddings via a multi-modal MLP-LSTM-Transformer stack.
- **Memory Bank**: Maintains long-tail clusters and associated few-shot planner experiences.
- **LLM Reasoning and Planner Selection**: Generates task-specific prompts and selects behavior planners in real-time.

> LiLoDriver is the first framework to introduce *lifelong learning* and *retrieval-augmented planning* into closed-loop autonomous driving at scale.

## 🚗 Key Features

- ✅ Closed-loop motion planning with long-tail generalization
- ✅ Online planner adaptation without model retraining
- ✅ Memory-augmented scenario clustering and few-shot planner retrieval
- ✅ LLM-based reasoning for scene-specific decision making
- ✅ Evaluated on the **nuPlan** real-world benchmark

---

## 📦 Installation

```bash
git clone https://github.com/your-org/LiLoDriver.git
cd LiLoDriver
conda create -n lilodriver python=3.10
conda activate lilodriver
pip install -r requirements.txt
```

We recommend using **CUDA 11.7+** and **PyTorch 2.0+** for GPU-accelerated training and inference.

---

## 📁 Project Structure

```bash
LiLoDriver/
├── assets/                     # Diagrams and figures
├── configs/                   # Config files for model and experiments
├── data/                      # Dataset loaders and preprocessing scripts
├── memory/                    # Memory bank and planner retrieval logic
├── models/                    # Scene encoder and planner network modules
├── planners/                  # Rule-based and learned behavior planners
├── prompts/                   # Prompt engineering for LLM calls
├── utils/                     # Common utilities
├── main.py                    # Entry point for training and evaluation
└── README.md
```

---

## 📊 Dataset

We evaluate LiLoDriver on the [**nuPlan benchmark**](https://github.com/motional/nuplan-devkit), which provides large-scale real-world driving data with support for **closed-loop simulation**.

To use nuPlan:
```bash
# Install nuPlan devkit
pip install nuplan-devkit
# Set up simulation environment
python setup_nuplan.py --download_data --prepare_maps
```

---

## 🧪 Evaluation

To run closed-loop simulation:

```bash
python main.py --config configs/lilodriver_nuplan.yaml --eval_only
```

Key evaluation metrics:
- Success Rate (SR)
- Collision Rate (CR)
- Comfort Index (CI)
- Long-tail Scenario Completion (LTSC)

---

## ✨ Results

LiLoDriver outperforms existing rule-based and learning-based planners across rare and complex driving scenarios:

| Method         | SR ↑ | CR ↓ | LTSC ↑ |
|----------------|------|------|--------|
| IDM (rule)     | 78.2 | 9.4  | 61.3   |
| PDM-Closed     | 82.5 | 6.1  | 65.8   |
| PlanTF (learn) | 84.7 | 5.5  | 68.2   |
| **LiLoDriver** | **89.6** | **2.9** | **75.1** |

---

## 📌 Citation

If you find this work helpful, please cite us:

```bibtex
@article{LiLoDriver2024,
  title   = {LiLoDriver: A Lifelong Learning Framework for Closed-Loop Motion Planning in Long-tail Autonomous Driving Scenarios},
  author  = {Your Name and Others},
  journal = {ArXiv preprint},
  year    = {2024},
  url     = {https://github.com/your-org/LiLoDriver}
}
```

---

## 🤝 Acknowledgements

This work builds upon [nuPlan](https://github.com/motional/nuplan-devkit), [LLaMA](https://github.com/facebookresearch/llama), and [LangChain](https://github.com/hwchase17/langchain). We thank the open-source community for their contributions.

---

## 📬 Contact

For questions or collaboration inquiries, please contact:  
📧 `yourname@institute.edu`
