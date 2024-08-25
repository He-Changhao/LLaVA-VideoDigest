from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import cv2
import os
import clip
from PIL import Image
from scipy.spatial.distance import cosine
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for LLaVA-VideoDigest.")

    parser.add_argument("--video_path", type=str, default="village.mp4", help="Path to the video file.")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Directory to save outputs.")
    parser.add_argument("--keyframe_sensitivity", type=float, default=0.2, help="Sensitivity for keyframe extraction. The larger the value, the more keyframes are extracted.")
    parser.add_argument("--llava_path", type=str, default="/media/ssd2/hech/Video_Abstract/LLaVA/liuhaotian/llava-v1.5-7b", help="Path to the LLaVA model.")
    parser.add_argument("--translation_model_path", type=str, default="facebook/mbart-large-50-one-to-many-mmt", help="Path to the translation model.")
    parser.add_argument("--is_translate", type=bool, default=True, help="Whether to translate the summaries or not.")
    parser.add_argument("--language", type=str, default="zh_CN", help="Language code for translation.",choices=["zh_CN", "en_XX", "fr_XX", "de_DE", "es_XX", "ru_RU", "ja_XX", "ar_AR"])

    return parser.parse_args()

def video_digest(keyframes_paths, video_output_dir, model_path):
    """
    Generates summaries for the keyframes and the entire video using the LLaVA model.

    Args:
        keyframes_paths (list of str): Paths to the extracted keyframes.
        model_path (str): Path to the pre-trained LLaVA model.

    Returns:
        tuple: Contains the list of keyframe summaries and concise video summary.
    """
    
    print("\033[1;32mkeyframes summaries extracting...\033[0m")
    keyframes_summaries =[]
    prompt = "This is a keyframe extracted from the video. Please answer as detailed as possible what this keyframe describes. Please start with: This video frame describes..."
    for i, keyframe_path in enumerate(keyframes_paths):
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": keyframe_path,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512,
        })()
        print(f"\033[1;32mKeyframe {i+1} summary generating...\033[0m")
        keyframes_summaries.append(eval_model(args))
    print("\033[1;32mCompleted generating keyframes summaries!\033[0m")

    #####################################################################################

    print("\033[1;32mConcise summary generating...\033[0m")
    keyframes = [Image.open(keyframe_path) for keyframe_path in keyframes_paths]

    total_height = sum(img.height for img in keyframes)
    max_width = max(img.width for img in keyframes)
    
    concatenated_keyframes = Image.new('RGB', (max_width, total_height))
    
    y_offset = 0
    for img in keyframes:
        concatenated_keyframes.paste(img, (0, y_offset))
        y_offset += img.height
    concatenated_keyframes_path = os.path.join(video_output_dir, 'concatenated_keyframes.jpg')
    concatenated_keyframes.save(concatenated_keyframes_path)

    prompt = "I have provided you with a series of key frames captured from the video, in order, from top to bottom. Please use them to learn as much as you can about what the video describes and to generate a detailed video summary. The video summary should begin with: This video describes..."
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": concatenated_keyframes_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
    })()
    concise_summary = eval_model(args)
    print("\033[1;32mCompleted generating concise summary!\033[0m")

    return keyframes_summaries, concise_summary

def extract_keyframes(video_path, keyframes_dir, similarity_threshold):
    """
    Extracts keyframes from the video based on a similarity threshold.

    Args:
        video_path (str): Path to the input video file.
        keyframes_dir (str): Directory to save the extracted keyframes.
        similarity_threshold (float): Threshold for determining similarity between frames. A lower value results in more keyframes.

    Returns:
        list of str: Paths to the saved keyframe images.
    """

    print("\033[1;32mKeyframes extracting...\033[0m")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    keyframes = []
    previous_feature = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.eval().float()
    keyframes_count = 0
    for frame in frames:
        frame_preprocessed = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        with torch.no_grad():
            current_feature = model.encode_image(frame_preprocessed).cpu().numpy()
        if previous_feature is None:
            keyframes_count += 1
            print(f"\033[1;32mKeyframe {keyframes_count} detected\033[0m")
            keyframes.append(frame)
            previous_feature = current_feature
        else:
            similarity = 1 - cosine(previous_feature.ravel(), current_feature.ravel())
            if similarity < similarity_threshold:
                keyframes_count += 1
                print(f"\033[1;32mKeyframe {keyframes_count} detected\033[0m")
                keyframes.append(frame)
                previous_feature = current_feature

    keyframes_paths = []
    for i, frame in enumerate(keyframes):
        frames_i_path = os.path.join(keyframes_dir,f'keyframe_{i+1}.jpg')
        cv2.imwrite(frames_i_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        keyframes_paths.append(frames_i_path)
    print(f"\033[1;32mCompleted extracting keyframes! {keyframes_count} frames in total.\033[0m")

    return keyframes_paths

def translation(language, translation_model_path, video_output_dir, output_json_file_path):
    """
    Translates the generated summaries into the specified language if translation is enabled.

    Args:
        is_translate (bool): Whether to translate the summaries.
        language (str): The target language code for translation.
        translation_model_path (str): Path to the translation model.
        video_output_dir (str): Directory to save the translated summaries.
        output_json_file_path (str): Path to the JSON file containing the original summaries.
    """

    translation_tokenizer = MBart50TokenizerFast.from_pretrained(translation_model_path)
    translation_model = MBartForConditionalGeneration.from_pretrained(translation_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translation_model.to(device)

    def translate_text(text, src_lang="en_XX", tgt_lang=language):
        """
        Translates the given text from the source language to the target language.

        Args:
            text (str): The text to translate.
            src_lang (str): The source language code.
            tgt_lang (str): The target language code.

        Returns:
            str: The translated text.
        """

        translation_tokenizer.src_lang = src_lang
        model_inputs = translation_tokenizer(text, return_tensors="pt").to(device)
        generated_tokens = translation_model.generate(
            **model_inputs,
            forced_bos_token_id=translation_tokenizer.lang_code_to_id[tgt_lang]
        )
        translated_text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text

    with open(output_json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data["keyframes_summary"] = [translate_text(summary) for summary in data["keyframes_summary"]]
    data["concise summary"] = translate_text(data["concise summary"])

    with open(os.path.join(video_output_dir, f"output_{language}.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("\033[1;32mCompleted translating!\033[0m")

if __name__ == "__main__":
    args = parse_args()
 
    video_name = os.path.basename(args.video_path).split(".")[0]
    video_output_dir = os.path.join(args.output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    keyframes_dir = os.path.join(video_output_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    keyframes_paths = extract_keyframes(args.video_path, keyframes_dir, similarity_threshold=1-args.keyframe_sensitivity)
    keyframes_summaries, concise_summary = video_digest(keyframes_paths, video_output_dir, args.llava_path)
    
    data = {
        "keyframes_dir": keyframes_paths,
        "keyframes_summary": keyframes_summaries, 
        "concise summary": concise_summary,
    }

    output_json_file_path = os.path.join(video_output_dir, "output-en.json")
    with open(output_json_file_path, 'w', encoding='utf-8') as output:
        json.dump(data, output, ensure_ascii=False, indent=4)
        
    if args.is_translate:
        translation(args.language, args.translation_model_path, video_output_dir, output_json_file_path)