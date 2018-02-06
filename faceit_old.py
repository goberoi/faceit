import sys
sys.path.append('faceswap')

import numpy
import cv2
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from lib.faces_detect import detect_faces
from plugins.PluginLoader import PluginLoader


class Model:
    BASE_PATH = 'data'
    MEDIA_PATH = 'media'
    TRAINING_PATH = 'training'
    OUTPUT_PATH = 'output'

    @classmethod
    def _media_path(cls, filename):
            return os.path.join(Model.BASE_PATH, Model.MEDIA_PATH, filename)        

    @classmethod        
    def _training_path(cls, filename):
            return os.path.join(Model.BASE_PATH, Model.TRAINING_PATH, filename)        

    @classmethod        
    def _output_path(cls, filename):
            return os.path.join(Model.BASE_PATH, Model.OUTPUT_PATH, filename)        

    def __init__(self, name, media_a, media_b):
        self._name = name
        self._media_a = media_a
        self._media_b = media_b

    def prepare_data(self):
        for filename in (self._media_a + self._media_b):
            if os.path.isfile(self._media_path(filename)):
                self._process_video(filename)
            else:
                self._process_images(filename)                
        return

    def _process_video(self, filename):
        print("Processing video: %s" % self._media_path(filename))
        
        if os.path.exists(self._training_path(filename)):
            print("Processing video already done, found results at: %s" % self._training_path(filename))
            return

        training_dir = self._training_path(filename)
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            
        video = VideoFileClip(self._media_path(filename))
        
        extractor = PluginLoader.get_extractor("Align")()
        frame_num = 0
        for frame in video.iter_frames():
            print("Processing frame: %03d" % frame_num)
            for (face_num, face) in enumerate(detect_faces(frame)):
                print("Processing face: %03d" % face_num)
                training_file = os.path.join(training_dir, 'frame%03d_face%02d.jpg' % (frame_num, face_num))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_image = extractor.extract(frame, face, 256)
                cv2.imwrite(training_file, resized_image)
            frame_num += 1

    def _process_images(self, media):
        raise NotImplementedError("directories of images not yet supported")        

if __name__ == '__main__':
    oren_trump = Model('oren_trump', ['oren.mp4'], ['trump_speech.webm'])
    oren_trump.prepare_data()

#    trump_oren = Model('trump_oren', ['trump_speech.webm'], ['oren.mp4'])
#    trump_oren.prepare_data()



