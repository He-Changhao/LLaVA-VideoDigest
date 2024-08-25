# LLaVA-VideoDigest

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Output](#output)
- [License](#license)

## Overview
LLaVA-VideoDigest is a comprehensive tool for generating video summaries based on extracted keyframes. This project utilizes the LLaVA model for keyframe summarization and video summary generation. Additionally, it provides an optional feature to translate the summaries into multiple languages using the MBart50 model.

## Features
- **Keyframe Extraction**: Automatically extracts keyframes from a video based on visual similarity.
- **Keyframe Summarization**: Generates detailed descriptions of each keyframe using the LLaVA model.
- **Video Summary Generation**: Produces concise summary of the entire video.
- **Translation Support**: Optional feature to translate the generated summaries into multiple languages.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/He-Changhao/LLaVA-VideoDigest.git
   cd LLaVA-VideoDigest
   ```

2. Install the required dependencies:
   ```bash
   conda create -n llava_videodigest python=3.10 -y
   conda activate llava_videodigest
   pip install -e .
   pip install opencv-python
   pip install ftfy
   ```
   You can also install dependencies via `pip install -r requirements.txt`. Note that the cuda version needs to be higher than 12.0 to support LLaVA inference.

## Usage
To use the LLaVA-VideoDigest tool, run the following command:

```bash
python main.py --video_path path/to/video.mp4 --output_dir outputs/ --keyframe_sensitivity 0.2 --llava_path liuhaotian/llava-v1.5-7b --llava_path facebook/mbart-large-50-one-to-many-mmt --is_translate True --language zh_CN
```

## Arguments
- `--video_path`: Path to the video file to be processed.
- `--output_dir`: Directory where the outputs will be saved.
- `--llava_path`: Path to the LLaVA model.
- `--translation_model_path`: Path to the MBart50 model for translation.
- `--is_translate`: Whether to translate the summaries or not.
- `--keyframe_sensitivity`: Sensitivity for keyframe extraction. A larger value extracts more keyframes. (default: `0.2`)
- `--language`: Language code for translation. Choices: `zh_CN`, `en_XX`, `fr_XX`, `de_DE`, `es_XX`, `ru_RU`, `ja_XX`, `ar_AR`. (default: `zh_CN`)

## Output
The outputs are saved in the specified output directory, organized as follows:
- `keyframes/`: Directory containing the extracted keyframes.
- `concatenated_keyframes.jpg`: Image after keyframe concatenation.
- `output-en.json`: JSON file containing the keyframes' paths, keyframes' summaries, concise summary, and detailed summary in English.
- `output-<language>.json`: JSON file containing translated summaries (if translation is enabled).

In each JSON file, the output will include the following files:

- **keyframes_dir**: A list of paths to the extracted keyframes.
- **keyframes_summary**: A list of summaries corresponding to each keyframe.
- **concise summary**: A brief summary of the entire video content.

### Example Output

#### English Output:
```json
{
    "keyframes_dir": [
        "outputs/village/keyframes/keyframe_1.jpg",
        "outputs/village/keyframes/keyframe_2.jpg",
        "outputs/village/keyframes/keyframe_3.jpg",
        "outputs/village/keyframes/keyframe_4.jpg",
        "outputs/village/keyframes/keyframe_5.jpg",
        "outputs/village/keyframes/keyframe_6.jpg"
    ],
    "keyframes_summary": [
        "This video frame describes a group of people walking along a path near a river...",
        "This video frame describes a group of people riding horses through a shallow body of water, likely a river or stream...",
        "This video frame describes a peaceful scene of a large herd of sheep grazing on a lush green hillside...",
        "This video frame describes a small waterfall in a river, with a brick building in the background...",
        "This video frame describes a peaceful scene of a small village with a river running through it...",
        "This video frame describes a scene of two people riding horses through a shallow river..."
    ],
    "concise summary": "This video describes a picturesque scene of a small village with a river running through it. The village is surrounded by lush greenery..."
}
```

#### Chinese Output:
```json
{
    "keyframes_dir": [
        "outputs/village/keyframes/keyframe_1.jpg",
        "outputs/village/keyframes/keyframe_2.jpg",
        "outputs/village/keyframes/keyframe_3.jpg",
        "outputs/village/keyframes/keyframe_4.jpg",
        "outputs/village/keyframes/keyframe_5.jpg",
        "outputs/village/keyframes/keyframe_6.jpg"
    ],
    "keyframes_summary": [
        "这个视频框架描述了一群人沿着一条河边的小径走去。在场景中有三个人,一个人靠近左边,另一个人在中间...",
        "这个视频框架描述一群人骑马通过浅水体,可能是一条河流或溪流...",
        "这个视频框架描述了在茂密的绿色山坡上牧羊的一大群牧羊的和平景象。图片中至少有14只绵羊可见...",
        "这个视频框架描述了一条河里的小瀑布,背景有一座砖楼...",
        "这个视频框架描述了一个小村庄的和平景象,有一条河流穿过村庄...",
        "这个视频框架描述了两个人在浅河中骑马的场景,马在水里散步..."
    ],
    "concise summary": "这个视频描述了一个小村庄的景象,有一条河流穿过它。村庄被茂密的绿地包围,河流充满了人们享受各种活动..."
}
```

## Citation
```bibtex
@article{liu2024visual,
  title={Visual instruction tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  journal={Advances in neural information processing systems},
  volume={36},
  year={2024}
}

@article{tang2020multilingual,
    title={Multilingual Translation with Extensible Multilingual Pretraining and Finetuning},
    author={Yuqing Tang and Chau Tran and Xian Li and Peng-Jen Chen and Naman Goyal and Vishrav Chaudhary and Jiatao Gu and Angela Fan},
    year={2020},
    eprint={2008.00401},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```

## License
This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.