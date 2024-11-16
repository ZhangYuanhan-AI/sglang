"""
Usage:

python3 -m sglang.launch_server --model-path lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 --port=30000 --tp-size=4 --chat-template=chatml-llava

python3 http_llava_onevision_test.py
"""

import base64
import io
import os
import sys
import time

import numpy as np
import openai
import requests
from decord import VideoReader, cpu
from PIL import Image

# pip install httpx==0.23.3
# pip install decord
# pip install protobuf==3.20.0


def download_video(url, cache_dir):
    file_path = os.path.join(cache_dir, "jobs.mp4")
    os.makedirs(cache_dir, exist_ok=True)

    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"File downloaded and saved to: {file_path}")
    return file_path


def create_openai_client(base_url):
    return openai.Client(api_key="EMPTY", base_url=base_url)




def video_stream_request_test(client, video_path):
    print("------------------------Video Stream Request Test----------------------")
    messages = prepare_video_messages(video_path)

    video_request = client.chat.completions.create(
        model="default",
        messages=messages,
        temperature=0,
        max_tokens=1024,
        stream=True,
    )
    print("-" * 30)
    video_response = ""

    for chunk in video_request:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            video_response += content
            sys.stdout.write(content)
            sys.stdout.flush()
    print("-" * 30)




def video_speed_test(client, video_path):
    print("------------------------Video Speed Test------------------------")
    messages = prepare_video_messages(video_path)

    start_time = time.time()
    video_request = client.chat.completions.create(
        model="default",
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
    end_time = time.time()
    video_response = video_request.choices[0].message.content
    print(video_response)
    print("-" * 30)
    print_speed_test_results(video_request, start_time, end_time)


def prepare_video_messages(video_path):
    max_frames_num = 32
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    uniform_sampled_frames = np.linspace(
        0, total_frame_num - 1, max_frames_num, dtype=int
    )
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    base64_frames = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG")
        base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        base64_frames.append(base64_str)

    messages = [{"role": "user", "content": []}]
    for base64_frame in base64_frames:
        frame_format = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"},
            "modalities": "video",
        }
        messages[0]["content"].append(frame_format)

    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    prompt = {"type": "text", "text": f"{time_instruciton}\nPlease provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."}
    # prompt = {"type": "text", "text": f"Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."}
    print(prompt)
    messages[0]["content"].append(prompt)

    return messages


def print_speed_test_results(request, start_time, end_time):
    total_tokens = request.usage.total_tokens
    completion_tokens = request.usage.completion_tokens
    prompt_tokens = request.usage.prompt_tokens

    print(f"Total tokens: {total_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Token per second: {total_tokens / (end_time - start_time)}")
    print(f"Completion token per second: {completion_tokens / (end_time - start_time)}")
    print(f"Prompt token per second: {prompt_tokens / (end_time - start_time)}")


def main():
    url = "https://raw.githubusercontent.com/EvolvingLMMs-Lab/sglang/dev/onevision_local/assets/jobs.mp4"
    cache_dir = os.path.expanduser("~/.cache")
    video_path = download_video(url, cache_dir)
    # video_path="/mnt/bn/vl-research/data/llava_video/seed_videos/Gatech.mp4"

    client = create_openai_client("http://127.0.0.1:30000/v1")

    # image_stream_request_test(client)
    # multi_image_stream_request_test(client)
    video_stream_request_test(client, video_path)
    # image_speed_test(client)
    # video_speed_test(client, video_path)


if __name__ == "__main__":
    main()
