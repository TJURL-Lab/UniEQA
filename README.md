This repository contains the code for our manuscript UniEQA & UniEval: A Unified Benchmark and Evaluation Platform for Multimodal Foundation Models in Embodied Question Answering.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Getting-started">Getting Started</a>
      <ul>
        <li><a href="#Installation">Installation</a></li>
        <li><a href="#Inference and Evaluation">Inference</a></li>
      </ul>
    </li>
    <li><a href="#Contributing">Contributing</a></li>
    <li><a href="#License">License</a></li>
  </ol>
</details>

## File Structure and Usage
<details>
  <summary>File Structure and Usage</summary>
  <ol>
    <li>
      <a>data: benchmark data classified by capability aspect</a>
    </li>
    <li>
      <a>scrpts</a>
      <ul>
        <li><a>UniEval: evaluation platform code for UniEval</a></li>
        <li><a>scripts: example of a model evaluation script</a></li>
<!--         <li><a>minigpt4_eval.yaml.py: configuration file for Minigpt4</a></li> -->
      </ul>
    </li>
    <li><a>LICENSE: license file</a></li>
    <li><a>README.md</a></li>
  </ol>
</details>

## Getting Started
        
### Installation

1. Build environment

 We recommend building a standalone Conda environment for the model you want to use. For example, if you want to use BLIP-2:

  ```sh
  conda create -n BLIP-2 python=3.10
  conda activate BLIP-2
  pip install -r requirement.txt
  ```

  The requirement file of BLIP-2 is at [https://github.com/salesforce/LAVIS/blob/main/requirements.txt](https://github.com/salesforce/LAVIS/blob/main/requirements.txt)
   
  We recommend building a standalone Conda environment for UniEval:

  ```sh
  conda create -n UniEval python=3.10
  conda activate UniEval
  ```
  ```sh
  conda env create -f environment.yml
  ```
2. Clone the repo

   ```sh
   git clone https://github.com/TJURL-Lab/UniEQA
   cd UniEQA
   ```
### Download the dataset

1. For the dataset, please refer to the link: [Data](https://huggingface.co/datasets/TJURL-Lab/UniEQA)

   ```sh
   mv dataset ~/UniEval/benchmarks/
   ```
   
### Inference and Evaluation

1. Run the prediction and evaluation script for a model

   ```sh
   python dataset_predict_blip.py --model blip2flant5xl --output-root "./benchmark-evaluation" --device'"cuda:0"
   python evaluation_gpt3.5_mp.py
   ```
   <p align="right">(<a href="#readme-top">back to top</a>)</p>
2. Run the evaluation platform UniEval

   ```sh
   python controller.py
   python server_demo.py
   ```
   <p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## LICENSE
Our UniEQA benchmark is released under the BSD-3-Clause license. See the "LICENSE" file for additional details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


