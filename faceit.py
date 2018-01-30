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
            print("Proccessing video already done, found results at: %s" % self._training_path(filename))
            return

        output_dir = self._output_path(filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        video = VideoFileClip(self._media_path(filename))
        
        extractor = PluginLoader.get_extractor("Align")()
        frame_num = 0
        for frame in video.iter_frames():
            print("Processing frame: %03d" % frame_num)
            for (face_num, face) in enumerate(detect_faces(frame)):
                print("Processing face: %03d" % face_num)                
                resized_image = extractor.extract(frame, face, 256)
                output_file = os.path.join(output_dir, 'frame_%03d_%02d.jpg' % (frame_num, face_num))
                print(output_file)
                cv2.imwrite(output_file, resized_image)
            frame_num += 1

    def _process_images(self, media):
        raise NotImplementedError("directories of images not yet supported")        

if __name__ == '__main__':
    oren_trump = Model('oren_trump', ['oren.mp4'], ['trump_speech.webm'])
    oren_trump.prepare_data()

#    trump_oren = Model('trump_oren', ['trump_speech.webm'], ['oren.mp4'])
#    trump_oren.prepare_data()



