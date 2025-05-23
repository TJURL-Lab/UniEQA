This repository contains the code for our manuscript UniEQA & UniEval: A Unified Benchmark and Evaluation Platform for Multimodal Foundation Models in Embodied Question Answering.

For the data, please refer to the link: [Data](https://huggingface.co/datasets/TJURL-Lab/UniEQA)

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">installation</a></li>
        <li><a href="#inference">inference</a></li>
        <li><a href="#evaluation">evaluation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
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
        <li><a>chat_gpt_api.py: encapsulating ChatGPT inference</a></li>
        <li><a>dataset_predict_blip.py: inference script for BLIP-2 and InstructBLIP</a></li>
        <li><a>dataset_predict_gpt4v.py: inference script for ChatGPT(-4V)</a></li>
        <li><a>dataset_predict_llava.py: inference script for llava</a></li>
        <li><a>dataset_predict_minicpm.py: inference script for minicpm</a></li>
        <li><a>dataset_predict_minigpt4.py: inference script for minigpt4</a></li>
        <li><a>evaluate_gpt3.5_mp: using GPT-3.5 to evaluate prediction results with multithreading</a></li>
        <li><a>minigpt4_eval.yaml.py: configuration file for Minigpt4</a></li>
        <li><a>openai_cfg.json: configuration file for OpenAI api</a></li>
        <li><a>task_planning.py: Embodied Reasoning with GPT-4V</a></li>
      </ul>
    </li>
    <li><a>LICENSE: license file</a></li>
    <li><a>README.md</a></li>
  </ol>
</details>

## Getting Started
        
### installation

1. Build environment

   We recommend building a standalone Conda environment for the model you want to use. For example if you want to use BLIP-2:

  ```sh
  conda create -n BLIP-2 python=3.10
  conda activate BLIP-2
  pip install -r requirement.txt
  ```

   The requirement file of BLIP-2 is at [https://github.com/salesforce/LAVIS/blob/main/requirements.txt](https://github.com/salesforce/LAVIS/blob/main/requirements.txt)

2. Clone the repo

   ```sh
   git clone https://github.com/TJURL-Lab/UniEQA
   cd UniEQA
   ```

### inference

1. Run prediction script

   ```sh
   python dataset_predict_blip.py --model blip2flant5xl --output-root "./benchmark-evaluation" --device'"cuda:0"
   ```

### evaluation

1. Run evaluation script

   ```sh
   python evaluation_gpt3.5_mp.py
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


