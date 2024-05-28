[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=for-the-badge)](https://arxiv.org/abs/2402.01821) 


# Human-like Category Learning by Injecting Ecological Priors from Large Language Models into Neural Networks
This repository contains the code for the project Human-like category learning by injecting ecological priors from large language models into neural networks. 


<p align="center">
  <img src="ERMI.png" />
</p>

## Abstract
Ecological rationality refers to the notion that humans are rational agents adapted to their environment. However, testing this theory remains challenging due to two reasons: the difficulty in defining what tasks are ecologically valid and building rational models for these tasks. In this work, we demonstrate that large language models can generate cognitive tasks, specifically category learning tasks, that match the statistics of real-world tasks, thereby addressing the first challenge. We tackle the second challenge by deriving rational agents adapted to these tasks using the framework of meta-learning, leading to a class of models called ecologically rational meta-learned inference (ERMI). ERMI quantitatively explains human data better than seven other cognitive models in two different experiments. It additionally matches human behavior on a qualitative level: (1) it finds the same tasks difficult that humans find difficult, (2) it becomes more reliant on an exemplar-based strategy for assigning categories with learning, and (3) it generalizes to unseen stimuli in a human-like way. Furthermore, we show that ERMI's ecologically valid priors allow it to achieve state-of-the-art performance on the OpenML-CC18 classification benchmark.


## Project Structure

```bash
.
├── categorisation
│   ├── baselines # baseline models: GCM, Prototype, Rule, Rulex, and LLM
│   ├── benchmark # evaluate ERMI and other baseline models on OpenML-CC18 benchmark
│   ├── data # contains directories with data and results
│   │   ├── benchmark # results from benchmarking
│   │   ├── fitted_simulation # simulate data from MI models using parameters fitted to humans
│   │   ├── generated_tasks # generated tasks from LLM, synthetic, and OpenML-CC18
│   │   ├── human # human data from the three experiments
│   │   ├── llm # data from large language models
│   │   ├── meta_learner # simulate data from metalearned inference models
│   │   ├── model_comparison # results from model comparison
│   │   ├── stats # statistics from generated tasks
│   │   └── task_labels  # LLM synthesized problems
│   ├── mi # train, evaluate, and simulate meta-learned inference models: ERMI, MI and PFN
│   │   ├── baseline_classifiers.py # baseline classifiers
│   │   ├── envs.py # environment classes
│   │   ├── evaluate.py # evaluation functions
│   │   ├── fit_humans.py # fit human data to MI models
│   │   ├── fitted_simulations.py # simulate data from MI, ERMI and PFN models using parameters fitted to humans
│   │   ├── human_envs.py # environment classes simulating human experiments
│   │   ├── model.py # model classes
│   │   ├── model_utils.py # utility functions for models
│   │   ├── simulate_johanssen2002.py # simulate data from Johanssen et al. 2002
│   │   ├── simulate_mi.py # simulate data for different experiments from ERMI, MI and PFN models
│   │   ├── simulate_shepard1961.py # simulate data from Shepard et al. 1961
│   │   └── train_transformer.py # train ERMI, MI and PFN models
│   ├── task_generation # category learning tasks from large language models, synthetic tasks, and OpenML-CC18 tasks
│   │   ├── generate_linear_data.py # generate category learning tasks with linear decision boundaries
│   │   ├── generate_real_data.py # save category learning tasks from OpenML-CC18 benchmark
│   │   ├── generate_synthetic_data.py # generate synthetic category learning tasks
│   │   ├── generate_tasklabels.py # generate task labels for LLM synthesized problems
│   │   ├── generate_tasks.py # generate category learning tasks from large language models
│   │   ├── parse_generated_tasks.py # parse generated tasks from large language models
│   │   ├── prompts.py # prompts for large language models to generate category learning tasks
│   │   └── utils.py
│   ├── trained_models  # trained ERMI, MI and PFN models
│   ├── make_plots.py # replicate plots used in the paper
│   ├── plots.py # plot functions
│   └── utils.py
├── figures # all figures used in the paper
├── scripts # bash scripts for running experiments
├── notebooks # jupyter notebooks for playing around with data, benchmarking...
├── logs # log files from running experiments
└── README.md

```

The project also contains an .env file for storing environment variables and a requirements.txt file for installing the required Python libraries.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Installation
Clone the repository to your local machine. Then, install the required Python libraries from `requirements.text` and install the ermi package using pip:
    
```bash
git clone https://github.com/akjagadish/ermi.git
cd ermi
pip install -r requirements.txt
pip install .
```

### Configuration
The project uses a configuration file called .env to store environment variables. The .env file should be located in the root directory of the project and should contain the following variables:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
```
Replace your_anthropic_api_key with your actual Anthropic API key. You can obtain an API key by signing up for Anthropic's API service.

## Usage


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This project is for research purposes only and should not be used for any other purposes.

## Citation

If you use our work, please cite our
[paper](https://arxiv.org/abs/2402.01821) as such:

``` bibtex
@article{jagadish2024ecologically,
  title={Ecologically rational meta-learned inference explains human category learning},
  author={Jagadish, Akshay K and Coda-Forno, Julian and Thalmann, Mirko and Schulz, Eric and Binz, Marcel},
  journal={arXiv preprint arXiv:2402.01821},
  year={2024}
}
```