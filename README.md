# ShotBench: Expert-Level Cinematic Understanding in Vision-Language Models

<div>
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
</div>

<div style="text-align:center;">
  <span>(* equal contributions)&nbsp;&nbsp;(† corresponding authors)</span>
</div>

<div style="text-align:center;">
  <sup>1</sup>Tongji University,&emsp;
  <sup>2</sup>The Chinese University of Hong Kong,&emsp;<br>
  <sup>3</sup>Shanghai Artificial Intelligence Laboratory,&emsp;
  <sup>4</sup>S-Lab, Nanyang Technological University
</div>
<div style="text-align:center; margin:1rem 0;">
  <a href="https://arxiv.org/abs/2506.21356"
     target="_blank">Paper</a>&emsp;
  <a href="https://huggingface.co/datasets/Vchitect/ShotBench"
     target="_blank">Dataset</a>&emsp;
  <a href="https://huggingface.co/collections/Vchitect/shot-vl-685e541cdc5583148b36c12f"
     target="_blank">Model</a>&emsp;
  <a href="https://vchitect.github.io/ShotBench-project/"
     target="_blank">Project&nbsp;Page</a>
</div>

<div align="center">
  <iframe
    src="https://www.youtube.com/embed/MJBJlJEsPFM?rel=0&modestbranding=1&playsinline=1"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen
    title="YouTube video player"
    style="width:75%;max-width:960px;aspect-ratio:16/9;border:0;">
  </iframe>
</div>

## 🎬 Overview

- We introduce **ShotBench**, a comprehensive benchmark for evaluating VLMs’ understanding of cinematic language. It comprises over 3.5k expert-annotated QA pairs derived from images and video clips of over 200 critically acclaimed films (predominantly Oscar-nominated), covering eight distinct cinematography dimensions. This provides a rigorous new standard for assessing fine-grained visual comprehension in film.
- We conducted an extensive evaluation of 24 leading VLMs, including prominent open-source and proprietary models, on ShotBench. Our results reveal a critical performance gap: even the most capable model, GPT-4o, achieves less than 60% average accuracy. This systematically quantifies the current limitations of VLMs in genuine cinematographic comprehension.
- To address the identified limitations and facilitate future research, we constructed **ShotQA**, the first large-scale multimodal dataset for cinematography understanding, containing approximately 70k high-quality QA pairs. Leveraging ShotQA, we developed **ShotVL**, a novel VLM trained using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). ShotVL significantly surpasses all tested open-source and proprietary models, establishing a new **state-of-the-art** on ShotBench.

## 🔥 News

- [2025.06.27] Release ShotBench **test** split.
- [2025.06.27] Release our paper: **ShotBench: Expert-Level Cinematic Understanding in Vision-Language Models**.
- [2025.06.27] Release **ShotVL-7B** and **ShotVL-3B**, these models are currently SOTA VLMs on cinematography understanding.

