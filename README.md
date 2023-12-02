# Decoupling Style from Contents for Positive Text Reframing

## Natural Language Process System-set (NLPx)
The code of [paper](https://link.springer.com/chapter/10.1007/978-981-99-8178-6_6) is developed on a python package for processing NLP tasks based on transformers, pytorch, datasets etc. libraries.
Its mission is to reduce the duplicate labors when we set up and NLP models or framework in current popular deep learning framework or methodology cross multiple nodes.

## Data
<ol>
    <li> PPF: used to evaluate methods for PTR. </li>
</ol>

Source of data: [PPF](https://github.com/SALT-NLP/positive-frames)

## Installation
### Install from source
Clone the repository and install NLPx with the following commands
```shell
git clone git@github.com:codesedoc/FDSC.git
cd FDSC
pip install -e .
```
### Install with Docker
#### Preparation 
<ul>
    <li> Ubuntu (22.04 LTS) </li>
    <li> Docker (>=  23.0.5) </li>
</ul>
To protect system data during running docker container, it is recommended to creat a user belong to docker group, but without root permission.
Running follow command can create an account name "docker-1024"!

` bash sh/docker-1024 `

Running follow command to build the image of basic environment of NLPx. 

` docker compose build nlpx-env`

To use Nvidia GPU in docker containers, please install the "NVIDIA Container Toolkit" referring to [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt).

## Conduct Experiments
Running follow command to start 

` bash sh/docker-run `

To conduct different variants of method in paper, please define the value of variable "ARGS_FILE" to the path of experiment argument file.

## Citation

```
@InProceedings{sheng-etal-2023-decoupling,
    author="Xu, Sheng
        and Suzuki, Yoshimi
        and Li, Jiyi
        and Fukumoto, Fumiyo",
        editor="Luo, Biao
        and Cheng, Long
        and Wu, Zheng-Guang
        and Li, Hongyi
        and Li, Chaojie",
    title="Decoupling Style from Contents for Positive Text Reframing",
    booktitle="Neural Information Processing",
    year="2024",
    publisher="Springer Nature Singapore",
    address="Singapore",
    pages="73--84",
    isbn="978-981-99-8178-6"
}
```