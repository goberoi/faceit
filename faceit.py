import os
from moviepy import VideoFileClip
from lib.faceswap.lib.faces_detect import detect_faces
from lib.faceswap.plugins.PluginLoader import PluginLoader

class Model
    BASE_PATH = 'data':
    MEDIA_PATH = 'media'
    TRAINING_PATH = 'training'
    OUTPUT_PATH = 'output'
    
    def __init__(name, person_a, person_b, media_a, media_b):
        self._name = name
        self._media_a = media_a
        self._media_b = media_b

    def prepare_data():
        for media in (self._media_a + self._media_b):
            if os.path.isfile(media):
                self._process_video(media)
            else:
                self._process_images(media)                
        return

    def _process_video(filename):
        video_path = os.path.join(BASE_PATH, MEDIA_PATH, filename)
        training_path = os.path.join(BASE_PATH, TRAINING_PATH, filename)
        print("Processing video: %s" % video_path)
        
        if os.path.exists(training_path):
            print("Proccessing video already done, found results at: %s" % training_path)
            return

        output_dir = os.path.join(BASE_PATH, TRAINING_PATH, filename)
        video = VideoFileClip(video_path)
        
        extractor = PluginLoader.get_extractor("Align")()
        frame_num = 0
        for frame in video.iter_frames():
            for (face_num, face) in enumerate(detect_faces(frame)):
                resized_image = extractor.extract(frame, face, 256)
                output_file = os.path.join(output_dir, 'frame_%03d_%02d.jpg' % (frame_num, face_num))
                cv2.imwrite(output_file, resized_image)
            frame_num += 1

    def _process_images(media):
        raise NotImplementedError("directories of images not yet supported")        

if __name__ == '__main__':
    pass


