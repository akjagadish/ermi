# Human-like Category Learning by Injecting Ecological Priors from Large Language Models into Neural Networks
This repository contains the code for the project Human-like category learning by injecting ecological priors from large language models into neural networks. 


<p align="center">
  <img src="ERMI.png" />
</p>

## Abstract
Ecological rationality refers to the notion that humans are rational agents adapted to their environment. However, testing this theory remains challenging due to two reasons: the difficulty in defining what tasks are ecologically valid and building rational models for these tasks. In this work, we demonstrate that large language models can generate cognitive tasks, specifically category learning tasks, that match the statistics of real-world tasks, thereby addressing the first challenge. We tackle the second challenge by deriving rational agents adapted to these tasks using the framework of meta-learning, leading to a class of models called ecologically rational meta-learned inference (ERMI). ERMI quantitatively explains human data better than seven other cognitive models in two different experiments. It additionally matches human behavior on a qualitative level: (1) it finds the same tasks difficult that humans find difficult, (2) it becomes more reliant on an exemplar-based strategy for assigning categories with learning, and (3) it generalizes to unseen stimuli in a human-like way. Furthermore, we show that ERMI's ecologically valid priors allow it to achieve state-of-the-art performance on the OpenML-CC18 classification benchmark.

Link to the paper: [ArXiv](https://arxiv.org/abs/2402.01821)


## Project Structure

```
.
├── categorisation
│   ├── baselines
│   ├── data
│   ├── mi
│   ├── task_generation
│   ├── trained_models
│   ├── make_plots.py
│   ├── plots.py
│   └── utils.py
├── figures
├── logs
├── scripts
├── notebooks
└── README.md

```

The project also contains a .env file for storing environment variables and a requirements.txt file for installing the required Python libraries.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
The project uses several Python libraries such as openai, pandas, numpy, json, torch, argparse, dotenv, and anthropic. Make sure to install these before running the project.

### Installation
Clone the repository to your local machine. Then, install the required Python libraries from `requirements.text` using pip:
    
```bash
git clone https://github.com/akjagadish/ermi.git
cd ermi
pip install -r requirements.txt
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
