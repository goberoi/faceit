import os
from argparse import Namespace
import youtube_dl
import cv2
import time
import tqdm
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import AudioFileClip
import shutil
from pathlib import Path
import sys
sys.path.append('faceswap')

from lib.utils import FullHelpArgumentParser
from scripts.extract import ExtractTrainingData
from scripts.train import TrainingProcessor
from scripts.convert import ConvertImage
from lib.faces_detect import detect_faces
from plugins.PluginLoader import PluginLoader
from lib.FaceFilter import FaceFilter

class FaceIt:
    VIDEO_PATH = 'data/videos'
    PERSON_PATH = 'data/persons'
    PROCESSED_PATH = 'data/processed'
    MODEL_PATH = 'models'
    
    def __init__(self, name, person_a, person_b):
        self._faceswap = FaceSwapInterface()

        self._name = name
        
        self._person_a = person_a
        self._person_b = person_b

        self._faces = {
            self._person_a : os.path.join(FaceIt.PERSON_PATH, person_a + '.jpg'),
            self._person_b : os.path.join(FaceIt.PERSON_PATH, person_b + '.jpg')
        }
        self._videos = {
            self._person_a : [],
            self._person_b : []
        }
        if not os.path.exists(os.path.join(FaceIt.VIDEO_PATH)):
            os.makedirs(FaceIt.VIDEO_PATH)
        if True:
            # Magic incantation to let tensorflow use more GPU memory
            import tensorflow as tf
            import keras.backend.tensorflow_backend as K
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list="0"
            K.set_session(tf.Session(config=config))
            

    def add_video(self, person, name, url=None, fps=20):
        self._videos[person].append({
            'name' : name,
            'url' : url,
            'fps' : fps
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

    def _video_path(self, video):
        return os.path.join(FaceIt.VIDEO_PATH, video['name'])        

    def _video_frames_path(self, video):
        return os.path.join(FaceIt.PROCESSED_PATH, video['name'] + '_frames')        

    def _video_faces_path(self, video):
        return os.path.join(FaceIt.PROCESSED_PATH, video['name'] + '_faces')

    def _model_path(self, use_gan = False):
        path = FaceIt.MODEL_PATH
        if use_gan:
            path += "_gan"
        return os.path.join(path, self._name)

    def _model_data_path(self):
        return os.path.join(FaceIt.PROCESSED_PATH, "model_data_" + self._name)
    
    def _model_person_data_path(self, person):
        return os.path.join(self._model_data_path(), person)

    def _fetch_video(self, person, video):
        options = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio',
            'outtmpl': os.path.join(FaceIt.VIDEO_PATH, video['name']),
            'merge_output_format' : 'mp4'
        }
        with youtube_dl.YoutubeDL(options) as ydl:
            x = ydl.download([video['url']])

    def _extract_frames(self, person, video):
        video_frames_dir = self._video_frames_path(video)
        video_clip = VideoFileClip(self._video_path(video))
        
        start_time = time.time()
        print('[extract-frames] about to extract_frames for {}, fps {}, length {}s'.format(video_frames_dir, video_clip.fps, video_clip.duration))
        
        if os.path.exists(video_frames_dir):
            print('[extract-frames] frames already exist, skipping extraction: {}'.format(video_frames_dir))
            return
        
        os.makedirs(video_frames_dir)
        frame_num = 0
        for frame in tqdm.tqdm(video_clip.iter_frames(fps=video['fps']), total = video_clip.fps * video_clip.duration):
            video_frame_file = os.path.join(video_frames_dir, 'frame_{:03d}.jpg'.format(frame_num))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
            cv2.imwrite(video_frame_file, frame)
            frame_num += 1

        print('[extract] finished extract_frames for {}, total frames {}, time taken {:.0f}s'.format(
            video_frames_dir, frame_num-1, time.time() - start_time))            

    def _extract_faces(self, person, video):
        video_faces_dir = self._video_faces_path(video)

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(video_faces_dir))
        
        if os.path.exists(video_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(video_faces_dir))
            return
        
        os.makedirs(video_faces_dir)
        self._faceswap.extract(self._video_frames_path(video), video_faces_dir, self._faces[person])

    def preprocess(self):
        self.fetch()
        self.extract_frames()
        self.extract_faces()
    
    def _symlink_faces_for_model(self, person, video):
        for face_file in os.listdir(self._video_faces_path(video)):
            target_file = os.path.join(self._model_person_data_path(person), video['name'] + "_" + face_file)
            face_file_path = os.path.join(os.getcwd(), self._video_faces_path(video), face_file)
            os.symlink(face_file_path, target_file)

    def train(self, use_gan = False):
        # Setup directory structure for model, and create one director for person_a faces, and
        # another for person_b faces containing symlinks to all faces.
        if not os.path.exists(self._model_path(use_gan)):
            os.makedirs(self._model_path(use_gan))

        if os.path.exists(self._model_data_path()):
            shutil.rmtree(self._model_data_path())

        os.makedirs(self._model_person_data_path(self._person_a))
        os.makedirs(self._model_person_data_path(self._person_b))            
        self._process_videos(self._symlink_faces_for_model)

        self._faceswap.train(self._model_person_data_path(self._person_a), self._model_person_data_path(self._person_b), self._model_path(use_gan), use_gan)

    def convert(self, video_file, swap_model = False, max_frames = None, use_gan = False):
        video_path = self._video_path({ 'name' : video_file })
        video = VideoFileClip(video_path)

        model_name = "Original"
        converter_name = "Masked"
        if use_gan:
            model_name = "GAN"
            converter_name = "GAN"
        model = PluginLoader.get_model(model_name)(Path(self._model_path(use_gan)))

        if not model.load(swap_model):
            print('model Not Found! A valid model must be provided to continue!')
            exit(1)


        converter = PluginLoader.get_converter(converter_name)
        converter = converter(model.converter(False),
                              blur_size=8,
                              seamless_clone=True,
                              mask_type="facehullandrect",
                              erosion_kernel_size=None,
                              smooth_mask=True,
                              avg_color_adjust=True)

        filter_person = self._person_a
        if swap_model:
            filter_person = self._person_b
        filter = FaceFilter(self._faces[filter_person])

        def _convert_helper(get_frame, t):
            return _convert_frame(get_frame(t))

        frames_converted = 0
        def _convert_frame(frame):
            nonlocal frames_converted
            if max_frames and frames_converted > max_frames:
                return frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV            
            for face in detect_faces(frame, "cnn"):
                if filter.check(face):
                    frame = converter.patch_image(frame, face)
                    frame = frame.astype(numpy.float32)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
            frames_converted += 1            
            return frame

        # Convert frames 
        new_video = video.fl(_convert_helper)
        
        new_video.write_videofile(video_file, rewrite_audio = True)
        del video
        del new_video                

class FaceSwapInterface:
    def __init__(self):
        self._parser = FullHelpArgumentParser()
        self._subparser = self._parser.add_subparsers()

    def extract(self, input_dir, output_dir, filter_path):
        extract = ExtractTrainingData(
            self._subparser, "extract", "Extract the faces from a pictures.")
        args_str = "extract --input-dir {} --output-dir {} --filter {} --processes 1 --detector cnn"
        args_str = args_str.format(input_dir, output_dir, filter_path)
        self._run_script(args_str)

    def train(self, input_a_dir, input_b_dir, model_dir, gan = False):
        model_type = "Original"
        if gan:
            model_type = "GAN"
        train = TrainingProcessor(
            self._subparser, "train", "This command trains the model for the two faces A and B.")
        args_str = "train --input-A {} --input-B {} --model-dir {} --trainer {} --batch-size {} --write-image"
        args_str = args_str.format(input_a_dir, input_b_dir, model_dir, model_type, 512)
        self._run_script(args_str)

    def _run_script(self, args_str):
        args = self._parser.parse_args(args_str.split(' '))
        args.func(args)




#faceit = FaceIt('pikotaro_to_jacob', 'pikotaro', 'jacob')
#faceit.add_video('pikotaro', 'pikotaro_music_video.mp4', 'https://www.youtube.com/watch?v=Ct6BUPvE2sM', fps=20)
#faceit.add_video('jacob', 'jacob_rolex.mp4', 'https://www.youtube.com/watch?v=HPcbjLJXelU')
#faceit.add_video('jacob', 'jacob_wall.mp4', 'https://www.youtube.com/watch?v=91LULLWBRqk')
#faceit.add_video('jacob', 'jacob_pitch.mp4', 'https://www.youtube.com/watch?v=smRCM5Smwls')
#faceit.add_video('jacob', 'jacob_interview.mp4', 'https://www.youtube.com/watch?v=Y-mYHCO9lF8')

#faceit = FaceIt('pikotaro_to_jacob', 'pikotaro', 'jacob')

#faceit = FaceIt('rick_to_jacob', 'rick', 'jacob')
#faceit.add_video('rick', 'rick_never_gonna_give_you_up.mp4', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 20)
#faceit.add_video('jacob', 'jacob_rolex.mp4', 'https://www.youtube.com/watch?v=HPcbjLJXelU')
#faceit.add_video('jacob', 'jacob_wall.mp4', 'https://www.youtube.com/watch?v=91LULLWBRqk')
#faceit.add_video('jacob', 'jacob_pitch.mp4', 'https://www.youtube.com/watch?v=smRCM5Smwls')
#faceit.add_video('jacob', 'jacob_interview.mp4', 'https://www.youtube.com/watch?v=Y-mYHCO9lF8')

#faceit = FaceIt('bezos_to_jacob', 'bezos', 'jacob')
#faceit.add_video('bezos', 'bezos_alexa_lost_her_voice.mp4', 'https://www.youtube.com/watch?v=YLPNjaAOrBw')
#faceit.add_video('bezos', 'bezos_ice_bucket.mp4', 'https://www.youtube.com/watch?v=DFVezzjAhFY', fps=5)
#faceit.add_video('jacob', 'jacob_rolex.mp4', 'https://www.youtube.com/watch?v=HPcbjLJXelU')
#faceit.add_video('jacob', 'jacob_wall.mp4', 'https://www.youtube.com/watch?v=91LULLWBRqk')
#faceit.add_video('jacob', 'jacob_pitch.mp4', 'https://www.youtube.com/watch?v=smRCM5Smwls')
#faceit.add_video('jacob', 'jacob_interview.mp4', 'https://www.youtube.com/watch?v=Y-mYHCO9lF8')

faceit = FaceIt('trump_to_oren', 'trump', 'oren')
faceit.add_video('trump', 'trump_speech_compilation.mp4', 'https://www.youtube.com/watch?v=f0UB06v7yLY')
faceit.add_video('oren', 'oren_speech_stevens_institute.mp4', 'https://www.youtube.com/watch?v=V2V0Yiy0Afs')
faceit.add_video('oren', 'oren_future_of_data_mining.mp4', 'https://www.youtube.com/watch?v=ZCsrUI9kGII', fps=5)

# When getting ready to train
faceit.preprocess()

# Interactive for now
#faceit.train(use_gan = False)

faceit.convert('trump_speech_compilation.mp4')

#faceit.convert('bezos_alexa_lost_her_voice.mp4', max_frames = 30000, use_gan = False)

#faceit.convert('bezos_ice_bucket.mp4', max_frames = 600, use_gan = False)

#faceit.convert('jacob_wall.mp4', max_frames = 300, swap_model = True, use_gan = True)

#faceit.convert('rick_never_gonna_give_you_up.mp4', swap_model = True, max_frames = 600)

#faceit.convert('pikotaro_music_video.mp4')
