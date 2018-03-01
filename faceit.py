import os
from argparse import Namespace
import argparse
import youtube_dl
import cv2
import time
import tqdm
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.fx.all import crop
from moviepy.editor import AudioFileClip, clips_array, TextClip, CompositeVideoClip
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
    OUTPUT_PATH = 'data/output'
    MODEL_PATH = 'models'
    MODELS = {}

    @classmethod
    def add_model(cls, model):
        FaceIt.MODELS[model._name] = model
    
    def __init__(self, name, person_a, person_b):
        def _create_person_data(person):
            return {
                'name' : person,
                'videos' : [],
                'faces' : os.path.join(FaceIt.PERSON_PATH, person + '.jpg'),
                'photos' : []
            }
        
        self._name = name

        self._people = {
            person_a : _create_person_data(person_a),
            person_b : _create_person_data(person_b),
        }
        self._person_a = person_a
        self._person_b = person_b
        
        self._faceswap = FaceSwapInterface()

        if not os.path.exists(os.path.join(FaceIt.VIDEO_PATH)):
            os.makedirs(FaceIt.VIDEO_PATH)            

    def add_photos(self, person, photo_dir):
        self._people[person]['photos'].append(photo_dir)
            
    def add_video(self, person, name, url=None, fps=20):
        self._people[person]['videos'].append({
            'name' : name,
            'url' : url,
            'fps' : fps
        })

    def fetch(self):
        self._process_media(self._fetch_video)

    def extract_frames(self):
        self._process_media(self._extract_frames)

    def extract_faces(self):        
        self._process_media(self._extract_faces)
        self._process_media(self._extract_faces_from_photos, 'photos')        

    def all_videos(self):
        return self._people[self._person_a]['videos'] + self._people[self._person_b]['videos']

    def _process_media(self, func, media_type = 'videos'):
        for person in self._people:
            for video in self._people[person][media_type]:
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
        self._faceswap.extract(self._video_frames_path(video), video_faces_dir, self._people[person]['faces'])

    def _extract_faces_from_photos(self, person, photo_dir):
        photo_faces_dir = self._video_faces_path({ 'name' : photo_dir })

        start_time = time.time()
        print('[extract-faces] about to extract faces for {}'.format(photo_faces_dir))
        
        if os.path.exists(photo_faces_dir):
            print('[extract-faces] faces already exist, skipping face extraction: {}'.format(photo_faces_dir))
            return
        
        os.makedirs(photo_faces_dir)
        self._faceswap.extract(self._video_path({ 'name' : photo_dir }), photo_faces_dir, self._people[person]['faces'])


    def preprocess(self):
        self.fetch()
        self.extract_frames()
        self.extract_faces()
    
    def _symlink_faces_for_model(self, person, video):
        if isinstance(video, str):
            video = { 'name' : video }
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

        for person in self._people:
            os.makedirs(self._model_person_data_path(person))
        self._process_media(self._symlink_faces_for_model)

        self._faceswap.train(self._model_person_data_path(self._person_a), self._model_person_data_path(self._person_b), self._model_path(use_gan), use_gan)

    def convert(self, video_file, swap_model = False, duration = None, start_time = None, use_gan = False, face_filter = False, photos = True, crop_x = None, width = None, side_by_side = False):
        # Magic incantation to not have tensorflow blow up with an out of memory error.
        import tensorflow as tf
        import keras.backend.tensorflow_backend as K
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list="0"
        K.set_session(tf.Session(config=config))

        # Load model
        model_name = "Original"
        converter_name = "Masked"
        if use_gan:
            model_name = "GAN"
            converter_name = "GAN"
        model = PluginLoader.get_model(model_name)(Path(self._model_path(use_gan)))
        if not model.load(swap_model):
            print('model Not Found! A valid model must be provided to continue!')
            exit(1)

        # Load converter
        converter = PluginLoader.get_converter(converter_name)
        converter = converter(model.converter(False),
                              blur_size=8,
                              seamless_clone=True,
                              mask_type="facehullandrect",
                              erosion_kernel_size=None,
                              smooth_mask=True,
                              avg_color_adjust=True)

        # Load face filter
        filter_person = self._person_a
        if swap_model:
            filter_person = self._person_b
        filter = FaceFilter(self._people[filter_person]['faces'])

        # Define conversion method per frame
        def _convert_frame(frame, convert_colors = True):
            if convert_colors:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
            for face in detect_faces(frame, "cnn"):
                if (not face_filter) or (face_filter and filter.check(face)):
                    frame = converter.patch_image(frame, face)
                    frame = frame.astype(numpy.float32)
            if convert_colors:                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Swap RGB to BGR to work with OpenCV
            return frame
        def _convert_helper(get_frame, t):
            return _convert_frame(get_frame(t))

        media_path = self._video_path({ 'name' : video_file })
        if not photos:
            # Process video; start loading the video clip
            video = VideoFileClip(media_path)

            # If a duration is set, trim clip
            if duration:
                video = video.subclip(start_time, start_time + duration)
            
            # Resize clip before processing
            if width:
                video = video.resize(width = width)

            # Crop clip if desired
            if crop_x:
                video = video.fx(crop, x2 = video.w / 2)

            # Kick off convert frames for each frame
            new_video = video.fl(_convert_helper)

            # Stack clips side by side
            if side_by_side:
                def add_caption(caption, clip):
                    text = (TextClip(caption, font='Amiri-regular', color='white', fontsize=80).
                            margin(40).
                            set_duration(clip.duration).
                            on_color(color=(0,0,0), col_opacity=0.6))
                    return CompositeVideoClip([clip, text])
                video = add_caption("Original", video)
                new_video = add_caption("Swapped", new_video)                
                final_video = clips_array([[video], [new_video]])
            else:
                final_video = new_video

            # Resize clip after processing
            #final_video = final_video.resize(width = (480 * 2))

            # Write video
            output_path = os.path.join(self.OUTPUT_PATH, video_file)
            final_video.write_videofile(output_path, rewrite_audio = True)
            
            # Clean up
            del video
            del new_video
            del final_video
        else:
            # Process a directory of photos
            for face_file in os.listdir(media_path):
                face_path = os.path.join(media_path, face_file)
                image = cv2.imread(face_path)
                image = _convert_frame(image, convert_colors = False)
                cv2.imwrite(os.path.join(self.OUTPUT_PATH, face_file), image)

class FaceSwapInterface:
    def __init__(self):
        self._parser = FullHelpArgumentParser()
        self._subparser = self._parser.add_subparsers()

    def extract(self, input_dir, output_dir, filter_path):
        extract = ExtractTrainingData(
            self._subparser, "extract", "Extract the faces from a pictures.")
        args_str = "extract --input-dir {} --output-dir {} --processes 1 --detector cnn --filter {}"
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


if __name__ == '__main__':
    faceit = FaceIt('fallon_to_oliver', 'fallon', 'oliver')
    faceit.add_video('oliver', 'oliver_trumpcard.mp4', 'https://www.youtube.com/watch?v=JlxQ3IUWT0I')
    faceit.add_video('oliver', 'oliver_taxreform.mp4', 'https://www.youtube.com/watch?v=g23w7WPSaU8')
    faceit.add_video('oliver', 'oliver_zazu.mp4', 'https://www.youtube.com/watch?v=Y0IUPwXSQqg')
    faceit.add_video('oliver', 'oliver_pastor.mp4', 'https://www.youtube.com/watch?v=mUndxpbufkg')
    faceit.add_video('oliver', 'oliver_cookie.mp4', 'https://www.youtube.com/watch?v=H916EVndP_A')
    faceit.add_video('oliver', 'oliver_lorelai.mp4', 'https://www.youtube.com/watch?v=G1xP2f1_1Jg')
    faceit.add_video('fallon', 'fallon_mom.mp4', 'https://www.youtube.com/watch?v=gjXrm2Q-te4')
    faceit.add_video('fallon', 'fallon_charlottesville.mp4', 'https://www.youtube.com/watch?v=E9TJsw67OmE')
    faceit.add_video('fallon', 'fallon_dakota.mp4', 'https://www.youtube.com/watch?v=tPtMP_NAMz0')
    faceit.add_video('fallon', 'fallon_single.mp4', 'https://www.youtube.com/watch?v=xfFVuXN0FSI')
    faceit.add_video('fallon', 'fallon_sesamestreet.mp4', 'https://www.youtube.com/watch?v=SHogg7pJI_M')
    faceit.add_video('fallon', 'fallon_emmastone.mp4', 'https://www.youtube.com/watch?v=bLBSoC_2IY8')
    faceit.add_video('fallon', 'fallon_xfinity.mp4', 'https://www.youtube.com/watch?v=7JwBBZRLgkM')
    faceit.add_video('fallon', 'fallon_bank.mp4', 'https://www.youtube.com/watch?v=q-0hmYHWVgE')
    FaceIt.add_model(faceit)

    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices = ['preprocess', 'train', 'convert'])
    parser.add_argument('model', choices = FaceIt.MODELS.keys())
    parser.add_argument('video', nargs = '?')
    parser.add_argument('--duration', type = int, default = None)
    parser.add_argument('--photos', action = 'store_true', default = False)    
    parser.add_argument('--swap-model', action = 'store_true', default = False)
    parser.add_argument('--face-filter', action = 'store_true', default = False)
    parser.add_argument('--start-time', type = int, default = 0)
    parser.add_argument('--crop-x', type = int, default = None)
    parser.add_argument('--width', type = int, default = None)
    parser.add_argument('--side-by-side', action = 'store_true', default = False)    
    args = parser.parse_args()

    faceit = FaceIt.MODELS[args.model]
    
    if args.task == 'preprocess':
        faceit.preprocess()
    elif args.task == 'train':
        faceit.train()
    elif args.task == 'convert':
        if not args.video:
            print('Need a video to convert. Some ideas: {}'.format(", ".join([video['name'] for video in faceit.all_videos()])))
        else:
            faceit.convert(args.video, duration = args.duration, swap_model = args.swap_model, face_filter = args.face_filter, start_time = args.start_time, photos = args.photos, crop_x = args.crop_x, width = args.width, side_by_side = args.side_by_side)


