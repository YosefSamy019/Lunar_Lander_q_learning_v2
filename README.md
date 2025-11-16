# Lunar Lander Q-Learning Dataset Generator

This repository contains the code used to train a **Q-learning agent** on the **LunarLander-v2** environment from OpenAI Gym and generate a dataset of quantized state-action Q-values. The dataset can be used for studying reinforcement learning behavior, building baseline policies, or teaching RL concepts.

---

## 🔹 Repository Contents

- **lunar_lander.py** – Custom wrapper for the LunarLander environment with helper functions for state/action names and rendering.  
- **quantized_q_table.py** – Implementation of a quantized Q-table with caching and access counters.  
- **q_learning.py** – Q-learning agent that interacts with the environment and updates the Q-table.  
- **video_maker.py** – Tools to generate videos from environment frames and overlay text.  
- **show_video.py** – Functions for displaying videos in notebooks.  
- **object_cache.py** – Utility for saving/loading objects (like training history).  
- **train_agent.ipynb** – Notebook demonstrating how to train the agent and export datasets/videos.  

---

## 🔹 Dataset

After training, the code generates a CSV file (`qtable.csv`) containing:

- Quantized states of the Lunar Lander (8 features: positions, velocities, angle, angular velocity, leg contact).  
- Q‑values for each action (do nothing, fire left, fire main, fire right).  
- Metadata columns:  
  - `#op getter count` – how many times this state’s Q‑values were accessed  
  - `#op setter count` – how many times this state’s Q‑values were updated (small values indicate rarely visited states)

This CSV can be used to build greedy policies, analyze agent behavior, or serve as a dataset for teaching and research.

**You can access the dataset on Kaggle here:**  
[https://www.kaggle.com/datasets/youssef019/lunar-lander-q-learning-dataset/data](https://www.kaggle.com/datasets/youssef019/lunar-lander-q-learning-dataset/data)

---

## 🔹 How to Use

```python
import pandas as pd
df = pd.read_csv("q_table_cache/qtable.csv")
```

* Use `#op setter count` to identify rarely visited states.
* Determine best actions per state:

```python
df['best_action'] = df[['action_no_op','action_fire_left','action_fire_main','action_fire_right']].idxmax(axis=1)
```

---

## 🔹 Notes

* Q-values are initialized to `0` and updated during training.
* The videos and CSV are generated automatically and can be used directly for analysis or visualization.
* The code is designed to be beginner-friendly for anyone learning reinforcement learning.

---

## 🔹 License

This repository is open-source and free to use for research and educational purposes.


---

**Note:** The code used to generate the dataset can also be explored in this repository to reproduce results or modify training parameters.
