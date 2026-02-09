# Deep Spatial Guidance Framework (DSG-RRT*)

This repository hosts the official implementation and supplementary materials for the research article:

> **Title:** Deep Spatial Guidance Framework for Robot Social Cruising in Dynamic Scenarios  
> **Authors:** Shuai Guo, Xiaorui Liu, Xuan Mu, Wanyue Jiang, Wenzheng Chi, and Shuzhi Sam Ge  
> **Abstract:** Socially aware navigation requires robots to actively interact with human groups while ensuring safety in dynamic environments. To address this, we propose a hierarchical framework that orchestrates global interaction scheduling with local neural-augmented trajectory generation.

---

## Repository Contents

This repository is organized to distinguish between the core source code and the recorded experimental data:

* **`code/`** The complete source code for the project. This directory contains:
    * **Core Algorithms:** Implementation of the proposed **DSG-RRT*** planner and the comparative baseline algorithms (Standard RRT*, Informed RRT*, Neural RRT*), alongside the neural network architectures (PointNet++ and U-Net).
    * **Simulation Scripts:** Executable scripts for the 5 simulation scenarios (`simulation_scenario1.py` through `simulation_scenario5.py`).
    * **Experiment Scripts:** Control and planning scripts used for both static verification (`experiment_static_1+2.py`) and dynamic cruising experiments (`experiment_dynamic_1.py`, `experiment_dynamic_2.py`).
    * **Environment Utils:** Tools for procedural environment generation (`generate_random_world...`) and data visualization.

* **`simulation/`** Stores the recorded trajectory data and metric logs generated from the 5 simulation scenarios (Scenario 1 to 5).

* **`experiment_static/`** Stores the recorded data from the initial verification experiments in static scenarios (Static 1 and 2).

* **`experiment_dynamic/`** Stores the recorded data from the comparative cruising experiments in dynamic scenarios (Dynamic 1 and 2).

* **`requirements.txt`** A list of Python libraries and dependencies required to configure the environment.

* **`SN-1 scale.pdf`** The 8-item **subjective evaluation questionnaire** used for social compliance analysis.

* **`LICENSE`** Proprietary license restricting usage to academic review and verification only.

---

## Subjective Evaluation (SN-1 Scale)

The **SN-1 scale** (included as `SN-1 scale.pdf`) is a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree) designed to validate the **social compliance** of the robot. The scale comprises eight items addressing four key dimensions: **Distance Maintenance**, **Position Selection**, **Path Conflict**, and **Social Comfort**.

### Reliability and Validity Verification
The scale has been rigorously tested in our study involving **17 volunteers** across two dynamic experimental scenarios. The statistical analysis confirms its suitability for **evaluating social compliance**:

* **Internal Consistency:**
    * Cronbach's $\alpha$: **0.767** (indicating satisfactory reliability).
* **Construct Validity (Factor Analysis):**
    * KMO Measure of Sampling Adequacy: **0.753**.
    * Bartlett's Test of Sphericity: $\chi^2 = 126.57$, $df = 28$, $p < 0.001$.

---

##  Environment Setup

The required dependencies are listed in `requirements.txt`. You can install them via:

```bash
pip install -r requirements.txt