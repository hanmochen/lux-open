# Lux-Open



This is the official repo of the paper **[Emergent collective intelligence from massive-agent cooperation and competition](https://arxiv.org/abs/2301.01609)**.



## Directory Struture



- `agent` : our agent implementation of centralized policy
- `decentralized`: our decentralized agent as part of the ablation study
- `env`: a wrapper of the Lux-AI Season 1 environment. See more in https://www.lux-ai.org/.
- `lux`: utility functions used to interact with the Lux backend. The original code is from https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits/python/simple.
- `models`: checkpoints we used to evaluation different stages of the training process.
- `opponent`: the NO.1 winner of the Lux-S1 kaggle challenge. The original code is at  https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- `runEvaluation.py` the main entrance for runnning experiements.



## Usage



### Environment Setup

- Install the Lux environment: see instructions at https://github.com/Lux-AI-Challenge/Lux-Design-2021
- `pip install -r requirements.txt`



### Running experiements



- `python runEvaluation.py`
- Experiement Specification:
  - `mode`: evaluation mode, support self-play, different models and with opponent.
  - `model1` and `model2`: evaluation model paths
  - `num_games`: num of games in the evaluation
  - `map_size`: evaluation mapsizes
  - `seed`: random seed of the env



## Citation



```tex
@article{chen2023emergent,
  title={Emergent collective intelligence from massive-agent cooperation and competition},
  author={Chen, Hanmo and Tao, Stone and Chen, Jiaxin and Shen, Weihan and Li, Xihui and Cheng, Sikai and Zhu, Xiaolong and Li, Xiu},
  journal={arXiv preprint arXiv:2301.01609},
  year={2023}
}
```

