# ShotBench: Expert-Level Cinematic Understanding in Vision-Language Models

<p align="center">
    <a href='https://github.com/Alexios-hub' target='_blank'>Hongbo Liu</a><sup>1, 3*</sup>,&emsp;
    <a href='https://github.com/hejingwenhejingwen' target='_blank'>Jingwen He</a><sup>2, 3*</sup>,&emsp;
    <a href='https://github.com/MQN-80' target='_blank'>Yi Jin</a><sup>1</sup>,&emsp;
    <a href='https://zhengdian1.github.io/' target='_blank'>Dian Zheng</a><sup>3</sup>,&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=kMui170AAAAJ' target='_blank'>Yuhao Dong</a><sup>4</sup>,&emsp;
    <a href='https://github.com/zhangfan-p' target='_blank'>Fan Zhang</a><sup>3</sup>,&emsp;
    <a href='https://ziqihuangg.github.io/' target='_blank'>Ziqi Huang</a><sup>4</sup>,&emsp;
    <a href='https://scholar.google.com/citations?user=EgfF_CEAAAAJ&hl=en' target='_blank'>Yinan He</a><sup>3</sup>,&emsp;
    <a href='https://yg256li.github.io/' target='_blank'>Yangguang Li</a><sup>3</sup>,&emsp;
    <a href='https://dblp.org/pid/98/120-1.html' target='_blank'>Weichao Chen</a><sup>1</sup>,&emsp;
    <a href='https://mmlab.siat.ac.cn/yuqiao' target='_blank'>Yu Qiao</a><sup>3</sup>,&emsp;
    <a href='https://wlouyang.github.io/' target='_blank'>Wanli Ouyang</a><sup>2</sup>,&emsp;
    <a href='https://orcid.org/0000-0002-4301-394X' target='_blank'>Shengjie Zhao</a><sup>1&dagger;</sup>,&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>4&dagger;</sup>&emsp;
</p>

<p align="center">
  (* equal contributions) &nbsp;&nbsp; († corresponding authors)
</p>

<p align="center">
  <sup>1</sup> Tongji University &emsp;
  <sup>2</sup> The Chinese University of Hong Kong &emsp;<br>
  <sup>3</sup> Shanghai Artificial Intelligence Laboratory &emsp;
  <sup>4</sup> S-Lab, Nanyang Technological University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.21356">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2506.21356-B31B1B?logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/Vchitect/ShotBench">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange?logo=huggingface" alt="Dataset">
  </a>
  <a href="https://huggingface.co/collections/Vchitect/shot-vl-685e541cdc5583148b36c12f">
    <img src="https://img.shields.io/badge/Model-ShotVL-green" alt="Model">
  </a>
  <a href="https://vchitect.github.io/ShotBench-project/">
    <img src="https://img.shields.io/badge/Project&nbsp;Page-Website-lightgrey?logo=googlechrome" alt="Project Page">
  </a>
</p>
<p align="center">
  <a href="https://www.youtube.com/watch?v=MJBJlJEsPFM">
    <img src="assets/shotbench_demo.gif" alt="ShotBench Demo (click to play)">
  </a>
</p>


## 🎬 Overview
- We introduce **ShotBench**, a comprehensive benchmark for evaluating VLMs’ understanding of cinematic language. It comprises over 3.5 k expert-annotated QA pairs derived from images and video clips of over 200 critically acclaimed films (predominantly Oscar-nominated), covering eight distinct cinematography dimensions. This provides a rigorous new standard for assessing fine-grained visual comprehension in film.
- We conducted an extensive evaluation of 24 leading VLMs, including prominent open-source and proprietary models, on ShotBench. Our results reveal a critical performance gap: even the most capable model, GPT-4o, achieves less than 60 % average accuracy. This systematically quantifies the current limitations of VLMs in genuine cinematographic comprehension.
- To address the identified limitations and facilitate future research, we constructed **ShotQA**, the first large-scale multimodal dataset for cinematography understanding, containing approximately 70 k high-quality QA pairs. Leveraging ShotQA, we developed **ShotVL**, a novel VLM trained using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). ShotVL significantly surpasses all tested open-source and proprietary models, establishing a new **state-of-the-art** on ShotBench.

## 🔥 News
- **2025-07-7** Release **Evaluation** code.
- **2025-07-2** Release [**ShotQA-70k**](https://huggingface.co/datasets/Vchitect/ShotQA) dataset.
- **2025-06-27** Release [**ShotBench**](https://huggingface.co/datasets/Vchitect/ShotBench) **test** split.  
- **2025-06-27** Release our paper: [**ShotBench: Expert-Level Cinematic Understanding in Vision-Language Models**](https://arxiv.org/abs/2506.21356).  
- **2025-06-27** Release[ **ShotVL-7B**](https://huggingface.co/Vchitect/ShotVL-7B) and [**ShotVL-3B**](https://huggingface.co/Vchitect/ShotVL-3B), these models are currently SOTA VLMs on cinematography understanding.

## Installation

```shell
conda create -n shotbench python=3.10
conda activate shotbench
pip install -r requirements.txt
```

## Evaluation

### 1.Preparing ShotBench Test Data

```shell
mkdir -p evaluation/data && cd evaluation/data
huggingface-cli download --repo-type dataset Vchitect/ShotBench --local-dir ShotBench
cd ShotBench
tar -xvf images.tar
tar -xvf videos.tar
cd ../../../
```

### 2.Run Evaluation Code

Evaluate ShotVL-3B with 4 GPUs:

```shell
accelerate launch --num_processes 4 evaluation/shotvl/evaluate.py --model ShotVL-3B --reasoning --output-dir eval_results
```

Evaluate ShotVL-7B with 4 GPUs:

```shell
accelerate launch --num_processes 4 evaluation/shotvl/evaluate.py --model ShotVL-7B --output-dir eval_results
```

### 3.Calculate Metrics

```shell
OPENAI_API_KEY=YOUR_OPENAI_APIKEY python evaluation/calculate_scores.py --prediction_path OUTPUT_FILE_PATH
```

## Evaluation Results

<div align="center">
<table>
  <caption>
    <small>
      Abbreviations:&nbsp;
      SS = <em>Shot&nbsp;Size</em>,&nbsp;
      SF = <em>Shot&nbsp;Framing</em>,&nbsp;
      CA = <em>Camera&nbsp;Angle</em>,&nbsp;
      LS = <em>Lens&nbsp;Size</em>,&nbsp;
      LT = <em>Lighting&nbsp;Type</em>,&nbsp;
      LC = <em>Lighting&nbsp;Conditions</em>,&nbsp;
      SC = <em>Shot&nbsp;Composition</em>,&nbsp;
      CM = <em>Camera&nbsp;Movement</em>.&nbsp;
      <u>Underline</u> marks previous best in each group.<br>
      <strong>Our <em>ShotVL</em> models establish new SOTA.</strong>
    </small>
  </caption><thead>
    <tr>
      <th>Models</th><th>SS</th><th>SF</th><th>CA</th><th>LS</th><th>LT</th>
      <th>LC</th><th>SC</th><th>CM</th><th>Avg</th>
    </tr>
  </thead><tbody>
  <tr><th colspan="10"><em>Open-Sourced&nbsp;VLMs</em></th></tr>
                            <tr><td>Qwen2.5-VL-3B-Instruct</td><td>54.6</td><td>56.6</td><td>43.1</td><td>36.6</td><td>59.3</td><td>45.1</td><td>41.5</td><td>31.9</td><td>46.1</td></tr>
                            <tr><td>Qwen2.5-VL-7B-Instruct</td><td>69.1</td><td>73.5</td><td>53.2</td><td>47.0</td><td>60.5</td><td>47.4</td><td>49.9</td><td>30.2</td><td>53.8</td></tr>
                            <tr><td>LLaVA-NeXT-Video-7B</td><td>35.9</td><td>37.1</td><td>32.5</td><td>27.8</td><td>50.9</td><td>31.7</td><td>28.0</td><td>31.3</td><td>34.4</td></tr>
                            <tr><td>LLaVA-Video-7B-Qwen2</td><td>56.9</td><td>65.4</td><td>45.1</td><td>36.0</td><td>63.5</td><td>45.4</td><td>37.4</td><td>35.3</td><td>48.1</td></tr>
                            <tr><td>LLaVA-Onevision-Qwen2-7B-Ov-Chat</td><td>58.4</td><td>71.0</td><td>52.3</td><td>38.7</td><td>59.5</td><td>44.9</td><td>50.9</td><td>39.7</td><td>51.9</td></tr>
                            <tr><td>InternVL2.5-8B</td><td>56.3</td><td>70.3</td><td>50.8</td><td>41.1</td><td>60.2</td><td>45.1</td><td>50.1</td><td>33.6</td><td>50.9</td></tr>
                            <tr><td>InternVL3-2B</td><td>56.3</td><td>56.0</td><td>44.4</td><td>34.6</td><td>56.8</td><td>44.6</td><td>43.0</td><td>38.1</td><td>46.7</td></tr>
                            <tr><td>InternVL3-8B</td><td>62.1</td><td>65.8</td><td>46.8</td><td>42.9</td><td>58.0</td><td>44.3</td><td>46.8</td><td>44.2</td><td>51.4</td></tr>
                            <tr><td>InternVL3-14B</td><td>59.6</td><td>82.2</td><td>55.4</td><td>40.7</td><td>61.7</td><td>44.6</td><td>51.1</td><td>38.2</td><td>54.2</td></tr>
                            <tr><td>Internlm-xcomposer2d5-7B</td><td>51.1</td><td>71.0</td><td>39.8</td><td>32.7</td><td>59.3</td><td>35.7</td><td>35.7</td><td>38.8</td><td>45.5</td></tr>
                            <tr><td>Ovis2-8B</td><td>35.9</td><td>37.1</td><td>32.5</td><td>27.8</td><td>50.9</td><td>31.7</td><td>28.0</td><td>35.3</td><td>34.9</td></tr>
                            <tr><td>VILA1.5-3B</td><td>33.4</td><td>44.9</td><td>32.1</td><td>28.6</td><td>50.6</td><td>35.7</td><td>28.4</td><td>21.5</td><td>34.4</td></tr>
                            <tr><td>VILA1.5-8B</td><td>40.6</td><td>44.5</td><td>39.1</td><td>29.7</td><td>48.9</td><td>32.9</td><td>34.4</td><td>36.9</td><td>38.4</td></tr>
                            <tr><td>VILA1.5-13B</td><td>36.7</td><td>54.6</td><td>40.7</td><td>34.8</td><td>52.8</td><td>35.4</td><td>34.2</td><td>31.3</td><td>40.1</td></tr>
                            <tr><td>Instructblip-vicuna-7B</td><td>27.0</td><td>27.9</td><td>34.5</td><td>29.4</td><td>44.4</td><td>29.7</td><td>27.1</td><td>25.0</td><td>30.6</td></tr>
                            <tr><td>Instructblip-vicuna-13B</td><td>26.8</td><td>29.2</td><td>27.9</td><td>28.0</td><td>39.0</td><td>24.0</td><td>27.1</td><td>22.0</td><td>28.0</td></tr>
                            <tr><td>InternVL2.5-38B</td><td>67.8</td><td><u>85.4</u></td><td>55.4</td><td>41.7</td><td>61.7</td><td>48.9</td><td>52.4</td><td>44.0</td><td>57.2</td></tr>
                            <tr><td>InternVL3-38B</td><td>68.0</td><td>84.0</td><td>51.9</td><td>43.6</td><td>64.4</td><td>46.9</td><td>54.7</td><td>44.6</td><td>57.3</td></tr>
                            <tr><td>Qwen2.5-VL-32B-Instruct</td><td>62.3</td><td>76.6</td><td>51.0</td><td>48.3</td><td>61.7</td><td>44.0</td><td>52.2</td><td>43.8</td><td>55.0</td></tr>
                            <tr><td>Qwen2.5-VL-72B-Instruct</td><td><u>75.1</u></td><td>82.9</td><td>56.7</td><td>46.8</td><td>59.0</td><td><u>49.4</u></td><td>54.1</td><td><u>48.9</u></td><td>59.1</td></tr>
                            <tr><td>InternVL3-78B</td><td>69.7</td><td>80.0</td><td>54.5</td><td>44.0</td><td><u>65.5</u></td><td>47.4</td><td>51.8</td><td>44.4</td><td>57.2</td></tr>
<tr><th colspan="10"><em>Proprietary&nbsp;VLMs</em></th></tr>
                            <tr><td>Gemini-2.0-flash</td><td>48.9</td><td>75.5</td><td>44.6</td><td>31.9</td><td>62.2</td><td>48.9</td><td>52.4</td><td>47.4</td><td>51.5</td></tr>
                            <tr><td>Gemini-2.5-flash-preview-04-17</td><td>57.7</td><td>82.9</td><td>51.4</td><td>43.8</td><td>65.2</td><td>45.7</td><td>45.9</td><td>43.5</td><td>54.5</td></tr>
                            <tr><td>GPT-4o</td><td>69.3</td><td>83.1</td><td><u>58.2</u></td><td><u>48.9</u></td><td>63.2</td><td>48.0</td><td><u>55.2</u></td><td>48.3</td><td><u>59.3</u></td></tr>
<tr><th colspan="10"><em>Ours</em></th></tr>
<tr>
  <td>ShotVL-3B
    <a href="https://huggingface.co/Vchitect/ShotVL-3B">
      <img src="https://img.shields.io/badge/Model-HF-yellow?logo=huggingface" alt="HF">
    </a>
  </td>
  <td>77.9</td><td>85.6</td><td>68.8</td><td>59.3</td><td>65.7</td>
  <td>53.1</td><td>57.4</td><td>51.7</td><td>65.1</td>
</tr>
<tr>
  <td>ShotVL-7B
    <a href="https://huggingface.co/Vchitect/ShotVL-7B">
      <img src="https://img.shields.io/badge/Model-HF-yellow?logo=huggingface" alt="HF">
    </a>
  </td>
  <td>81.2</td><td>90.1</td><td>78.0</td><td>68.5</td><td>70.1</td>
  <td>64.3</td><td>45.7</td><td>62.9</td><td>70.1</td>
</tr>  </tbody>
</table></div>

## Open-Sourcing Plan

- [ ] Release Training code.
- [x] Release Evaluation code.
- [x] Release **ShotQA-70k** dataset.
- [x] Release **ShotBench** test set.
- [x] Release **ShotVL** models.

## BibTeX

```
@misc{
      liu2025shotbench,
      title={ShotBench: Expert-Level Cinematic Understanding in Vision-Language Models}, 
      author={Hongbo Liu and Jingwen He and Yi Jin and Dian Zheng and Yuhao Dong and Fan Zhang and Ziqi Huang and Yinan He and Yangguang Li and Weichao Chen and Yu Qiao and Wanli Ouyang and Shengjie Zhao and Ziwei Liu},
      year={2025},
      eprint={2506.21356},
      achivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.21356}, 
    }
```

