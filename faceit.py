import sys
sys.path.append('faceswap')

import os
from argparse import Namespace
from moviepy.video.io.VideoFileClip import VideoFileClip
import youtube_dl
import cv2
import time

from lib.utils import FullHelpArgumentParser
from scripts.extract import ExtractTrainingData


class Model:
    VIDEO_PATH = 'data/videos'
    PERSON_PATH = 'data/persons'    
    
    def __init__(self, person_a, person_b):
        self._person_a = person_a
        self._person_b = person_b

        self._faces = {
            self._person_a : os.path.join(Model.PERSON_PATH, person_a + '.jpg'),
            self._person_b : os.path.join(Model.PERSON_PATH, person_b + '.jpg')
        }
        self._videos = {
            self._person_a : [],
            self._person_b : []
        }
        if not os.path.exists(os.path.join(Model.VIDEO_PATH)):
            os.makedirs(Model.VIDEO_PATH)
        

    def add_video(self, person, name, url=None):
        self._videos[person].append({
            'name' : name,
            'url' : url
        })

    def fetch(self):
        self._process_videos(self._fetch_video)

    def extract_frames(self):
        self._process_videos(self._extract_frames)

    def extract_faces(self):        
        self._process_videos(self._extract_faces)        

    def _process_videos(self, func):
        for person, videos in self._videos.items():
            for video in videos:
                func(person, video)

    def _fetch_video(self, person, video):
        options = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio',
            'outtmpl': os.path.join(Model.VIDEO_PATH, video['name']),
            'merge_output_format' : 'mp4'
        }
        with youtube_dl.YoutubeDL(options) as ydl:
            x = ydl.download([video['url']])

    def _extract_frames(self, person, video):
        video_path = os.path.join(Model.VIDEO_PATH, video['name'])
        video_frames_dir = video_path + '_frames'
        video = VideoFileClip(video_path)

        start_time = time.time()
        print('[extract-frames] about to extract_frames for {}, fps {}, length {}s'.format(video_frames_dir, video.fps, video.duration))
        
        if os.path.exists(video_frames_dir):
            print('[extract-frames] frames already exist, skipping extraction: {}'.format(video_frames_dir))
            return
        
        os.makedirs(video_frames_dir)
        frame_num = 0
        for frame in video.iter_frames(fps=2):
            video_frame_file = os.path.join(video_frames_dir, 'frame_%03d.jpg' % (frame_num))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
            cv2.imwrite(video_frame_file, frame)
            frame_num += 1

        print('[extract] finished extract_frames for {}, total frames {}, time taken {:.0f}s'.format(
            video_frames_dir, frame_num-1, time.time() - start_time))            

    def _extract_faces(self, person, video):
        video_path = os.path.join(Model.VIDEO_PATH, video['name'])
        video_frames_dir = video_path + '_frames'        
        video_faces_dir = video_path + '_faces'

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(video_faces_dir))
        
        if os.path.exists(video_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(video_faces_dir))
            return
        
        os.makedirs(video_faces_dir)

        parser = FullHelpArgumentParser()
        subparser = parser.add_subparsers()
        extract = ExtractTrainingData(
            subparser, "extract", "Extract the faces from a pictures.")

        args = Namespace(input_dir = video_frames_dir,
                         output_dir = video_faces_dir,
                         processes = 1,
                         filter = self._faces[person],
                         detector = 'cnn',
                         verbose = False)
        
        extract.process_arguments(args)

model = Model('oren', 'trump')

# Add video data
model.add_video('trump', 'trump_speech_compilation.mp4', 'https://www.youtube.com/watch?v=f0UB06v7yLY')
model.add_video('oren', 'oren_speech_stevens_institute.mp4', 'https://www.youtube.com/watch?v=V2V0Yiy0Afs')

# When getting ready to train
model.fetch()
model.extract_frames()
model.extract_faces()

# Interactive for now
model.train()


# Some day later
#model.add_video('oren', 'input/oren/oren.mp4') # Existing video file
#model.add_images('oren', 'input/oren_images') # Folder of images
