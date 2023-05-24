import os
import fileinput
# rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
# rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
# yum install ffmpeg

pdf_path = "./test1.pdf"
img_path = "./img/"
voice_path = "./voice/"
video_path = "./video/"


import urllib.request

from pdf2image import convert_from_path
# pip install pdf2image
# https://github.com/Belval/pdf2image
# conda install -c conda-forge poppler


def pdf_convert_img(pdf_path, img_path):
    images = convert_from_path(pdf_path, dpi=200)
    for image in images:
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        image.save(img_path + f'{images.index(image)}.png', 'PNG')


"""
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
"""

import cv2
import numpy as np

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
    pdf_convert_img(pdf_path, img_path)
    i_v_dict = dict()
    for img in os.listdir(img_path):
        i_v_dict[img_path + img] = voice_path + img.split(".")[0] + ".m4a"

    print(i_v_dict) 
    v_a_dict = img_convert_video(i_v_dict) 
    print(v_a_dict)
    video_list = video_add_audio(v_a_dict)
    print(video_list)
    #video_concatenate(video_list)
    """ 
    video = VideoFileClip("./video/0.jpg.mp4")
    print(video.duration)
    video.set_duration(video.duration)
   
    video.write_videofile("./l_video/1.jpg.mp4", audio_codec='aac')
    video = VideoFileClip("./l_video/1.jpg.mp4")
    print(video.duration)
    #video_concatenate(video_list)
    """

