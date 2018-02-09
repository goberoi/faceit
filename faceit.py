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
import sys
sys.path.append('faceswap')

from lib.utils import FullHelpArgumentParser
from scripts.extract import ExtractTrainingData
from scripts.train import TrainingProcessor
from scripts.convert import ConvertImage
from lib.faces_detect import detect_faces
from plugins.PluginLoader import PluginLoader


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

        # Magic incantation to let tensorflow use more GPU memory
        if False:
            import tensorflow as tf
            import keras.backend.tensorflow_backend as K
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list="0"
            K.set_session(tf.Session(config=config))

    def add_video(self, person, name, url=None, fps=2):
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

    def _model_path(self):
        return os.path.join(FaceIt.MODEL_PATH, self._name)

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
        for frame in tqdm.tqdm(video_clip.iter_frames(fps=video['fps'])):
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

    
    def _symlink_faces_for_model(self, person, video):
        if not os.path.exists(self._model_person_data_path(person)):
            os.makedirs(self._model_person_data_path(person))
        for face_file in os.listdir(self._video_faces_path(video)):
            target_file = os.path.join(self._model_person_data_path(person), video['name'] + "_" + face_file)
            face_file_path = os.path.join(os.getcwd(), self._video_faces_path(video), face_file)
            print("face_file_path {}, target_file {}".format(face_file_path, target_file))
            os.symlink(face_file_path, target_file)

    def train(self):
        # Setup directory structure for model, and create one director for person_a faces, and
        # another for person_b faces containing symlinks to all faces.
        if not os.path.exists(self._model_path()):
            os.makedirs(self._model_path())

        if not os.path.exists(self._model_data_path()):
            self._process_videos(self._symlink_faces_for_model)

        self._faceswap.train(self._model_person_data_path(self._person_a), self._model_person_data_path(self._person_b), self._model_path())

    def convert(self, video_file, swap_model = False):
        video_path = self._video_path({ 'name' : video_file })
        video = VideoFileClip(video_path)
        
        model = PluginLoader.get_model("Original")(self._model_path())
        if not model.load(swap_model):
            print('model Not Found! A valid model must be provided to continue!')
            exit(1)

        converter = PluginLoader.get_converter("Masked")
        converter = converter(model.converter(swap_model),
                              blur_size=8,
                              seamless_clone=True,
                              mask_type="facehullandrect",
                              erosion_kernel_size=None,
                              smooth_mask=True,
                              avg_color_adjust=True)

        def _convert_helper(get_frame, t):
            return _convert_frame(get_frame(t))
        
        def _convert_frame(frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV            
            for face in detect_faces(frame, "cnn"):
                # TODO: add filter by face. See self.filter.check(face) in cli.py 
                frame = converter.patch_image(frame, face)
                frame = frame.astype(numpy.float32)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV                            
            return frame

        # Convert frames one by one
        #frames = []
        #for frame in tqdm.tqdm(video.iter_frames(), total = video.fps * video.duration, desc = '[converting video] {}'.format(video_file)):
        #    frames.append(_convert_frame(frame))
        #new_video = ImageSequenceClip(frames, fps = video.fps)
        new_video = video.fl(_convert_helper)

        # Add audio
#        audio = AudioFileClip(video_path)
#        new_video = new_video.set_audio(audio)
        
        new_video.write_videofile(video_file)
        del video
        del new_video                

class FaceSwapInterface:
    def __init__(self):
        self._parser = FullHelpArgumentParser()
        subparser = self._parser.add_subparsers()
        extract = ExtractTrainingData(
            subparser, "extract", "Extract the faces from a pictures.")
        train = TrainingProcessor(
            subparser, "train", "This command trains the model for the two faces A and B.")
        convert = ConvertImage(
            subparser, "convert", "Convert a source image to a new one with the face swapped.")

    def extract(self, input_dir, output_dir, filter_path):
        args_str = "extract --input-dir {} --output-dir {} --filter {} --processes 1 --detector cnn"
        args_str = args_str.format(input_dir, output_dir, filter_path)
        self._run_script(args_str)

    def train(self, input_a_dir, input_b_dir, model_dir):
        args_str = "train --input-A {} --input-B {} --model-dir {} --batch-size {} --write-image"
        args_str = args_str.format(input_a_dir, input_b_dir, model_dir, 256)
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

faceit = FaceIt('rick_to_jacob', 'rick', 'jacob')
faceit.add_video('rick', 'rick_never_gonna_give_you_up.mp4', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ')
faceit.add_video('jacob', 'jacob_rolex.mp4', 'https://www.youtube.com/watch?v=HPcbjLJXelU')
faceit.add_video('jacob', 'jacob_wall.mp4', 'https://www.youtube.com/watch?v=91LULLWBRqk')
faceit.add_video('jacob', 'jacob_pitch.mp4', 'https://www.youtube.com/watch?v=smRCM5Smwls')
faceit.add_video('jacob', 'jacob_interview.mp4', 'https://www.youtube.com/watch?v=Y-mYHCO9lF8')


#faceit = FaceIt('trump_to_oren', 'trump', 'oren')
#faceit.add_video('trump', 'trump_speech_compilation.mp4', 'https://www.youtube.com/watch?v=f0UB06v7yLY')
#faceit.add_video('oren', 'oren_speech_stevens_institute.mp4', 'https://www.youtube.com/watch?v=V2V0Yiy0Afs')

# When getting ready to train
faceit.fetch()
faceit.extract_frames()
faceit.extract_faces()

# Interactive for now
faceit.train()


#faceit.convert('pikotaro_music_video.mp4')
