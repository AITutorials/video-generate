import fileinput
# rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
# rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
# yum install ffmpeg

path = "./input.md"
img_path = "./img/"
voice_path = "./voice/"
video_path = "./video/"


import cv2
import numpy as np
import urllib.request


def get_content_dict(path):
    content = list(map(lambda x: x.strip(), fileinput.FileInput(path)))
    content_dict = dict()
    for i, c in enumerate(content):
        try:
            if c[0] == "!" and content[i+1][0] != "!" and content[i+1]:
                pic_url = c[:-1].replace("![在这里插入图片描述](", "")
                res = urllib.request.urlopen(pic_url)
                img = np.asarray(bytearray(res.read()), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                pic_name = img_path + str(i) + ".jpg"
                cv2.imwrite(pic_name, img)   
                content_dict[pic_name] = content[i+1]
        except:
            pass
            
    return content_dict

'''
import torch
import torchaudio

torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# 传统合成器
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
#bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

def _get_voice_en(text, path):
    """很慢的语音转换"""
    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, _ = vocoder(spec, spec_lengths)
        torchaudio.save(path, waveforms.to("cpu"), vocoder.sample_rate)

'''

# 模型合成器
'''
waveglow = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub",
    "nvidia_waveglow",
    model_math="fp32",
    pretrained=False,
)
checkpoint = torch.hub.load_state_dict_from_url(
    "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",  # noqa: E501
    progress=False,
    map_location=device,
)
state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

waveglow.load_state_dict(state_dict)
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to(device)
waveglow.eval()

with torch.no_grad():
    waveforms = waveglow.infer(spec)
'''




from paddlespeech.cli.tts.infer import TTSExecutor
tts = TTSExecutor()

def _get_voice_cn(text, path):
    tts(text=text, output=path)



def get_voice_dict(content_dict):
    voice_dict = dict()
    for i, (k, v) in enumerate(content_dict.items()):
        path = voice_path + k.replace(img_path, "") + '.mp3'
        _get_voice_cn(v, path)
        voice_dict[k] = path 
    return voice_dict


import cv2
import numpy as np
import os

from math import ceil

from moviepy.editor import *

def img_convert_video(i_v_dict):
    """{'./img/2.jpg': './voice/2.jpg.mp3', './img/6.jpg': './voice/6.jpg.mp3'}"""
    out_video = dict()
    for k, v in i_v_dict.items():
        img = cv2.imread(k)
        height, width, layers = img.shape
        size = (width, height)
        # 视频比音频多Ns
        N = 1
        time_l = int(AudioFileClip(v).duration) + N      
        print(time_l) 
        fps_coef = 0.5
        video_path_out = video_path + k.replace(img_path, "") + ".mp4"
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'),1*fps_coef, size) 
        for i in range(0, ceil(time_l*fps_coef)):       
            out.write(img)
        out.release()
        out_video[video_path_out] = v
    return out_video




l_video_path = "./l_video/"


# 通过循环设置空白音长度
blank_audio = AudioFileClip("./voice/blank.m4a")
blank_audio = afx.audio_loop(blank_audio, duration=0.5)


def video_add_audio(v_a_dict):
    video_list = list()
    for k, v in v_a_dict.items():
        video = VideoFileClip(k)
        audio = AudioFileClip(v)
        print(video.duration)
        print(audio.duration) 
        audio = concatenate_audioclips([blank_audio, audio])
        print(audio.duration)
        video = video.set_audio(audio)
        print(video.duration)
        print("********")
        lv_path = l_video_path + k.replace(video_path, "")
        #video.set_duration(video.duration)
        print(video.fps)
        video.write_videofile(lv_path, audio_codec='aac', fps=1)
        audio.close()
        video.close()
        video_list.append(lv_path)
    return video_list


def video_concatenate(video_list, out_path="./fin.mp4"):
    clip_list = []
    for vl in video_list:
        clip_list.append(VideoFileClip(vl).resize(width=640, height=360))
    videos = concatenate_videoclips(clip_list)
    videos.write_videofile(out_path, audio_codec='aac')
    return out_path



if __name__ == "__main__":
    content_dict = get_content_dict(path)
    print(content_dict)
    print(len(content_dict))
    i_v_dict = get_voice_dict(content_dict)
    v_a_dict = img_convert_video(i_v_dict) 
    video_list = video_add_audio(v_a_dict)
    video_concatenate(video_list)
    """ 
    video = VideoFileClip("./video/0.jpg.mp4")
    print(video.duration)
    video.set_duration(video.duration)
   
    video.write_videofile("./l_video/1.jpg.mp4", audio_codec='aac')
    video = VideoFileClip("./l_video/1.jpg.mp4")
    print(video.duration)
    #video_concatenate(video_list)
    """

