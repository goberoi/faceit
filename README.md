## FaceIt

FaceIt lets you swap one face for another in a YouTube video.

This code relies on the deepfakes/faceswap library.

## Goals

Goals:
1. Figure out minimum requirements for decent quality results in terms of: training data, and time needed.
2. Understand the cost on AWS to generate models and images.
3. Evaluate whether we can build a viral consumer app on this technology based on cost, feasilibity, and fun.

Approach:
1. Build a tool to quickly test lots of training sizes and times.
2. Define a set of experiments, collecting initial data, and kick it off.
3. Review with vision team to understand what we can improve on the model.

## Usage

```
```

## UX Ideas

Home:

* People [add person]
  * Oren [add video] [add image] [add facebook] [add instagram]
  * Trump
  * Gaurav
  * Harrison Ford

* Models [add model]
  * Oren-Trump 1 hour, 1 video each [convert video] [convert photo] [train more]
  * Oren-Trump 2 hours, 1 video each
  * Oren-Trump 3 hours, 1 video each
  * Oren --> Harrison Ford
  * Trump --> Oren
  * Harrison Ford --> Oren

## Database Schema

Person
* id
* name
* facebook_username
* instagram_username

Media
* id
* person_id
* url
* name
* type (image, video)

Model
* id
* name
* person_a_id
* person_b_id
* url
* last_accuracy
* training_time
* trained_on (list of media)

## Notes

add_video
* download
* extract each frame
* find face in each frame
* filter each face
* save frame file


* Model
  * person_a
  * person_b

* Person
  * name
  * videos
  * models

class Video:
  * url
  * filename
  * frames_dir
  * extracted_faces_dir
  * filtered_extracted
  * __init__(url)


## Data Directory Structure

```
data/persons/trump.jpg
data/persons/oren.jpg

data/videos/trump_speech_compilation.mp4


data/processed/trump_speech_compilation.mp4_frames/
data/processed/trump_speech_compilation.mp4_faces/
data/processed/trump_oren_simple/trump
data/processed/trump_oren_simple/oren

models/trump_oren_simple/decoder_A.h5
models/trump_oren_simple/decoder_B.h5
models/trump_oren_simple/encoder.h5



convert/foo.mp4
convert/foo_converted.mp4
```

```
oren_trump_simple.json
model = {
    'name' : 'oren_trump_simple',
    'person_a' : {
        'name' : 'trump',
        'videos' : [
            ('trump_speech_compilation.mp4', 'https://www.youtube.com/watch?v=f0UB06v7yLY' ),
        ]
    },
    'person_b' : {
        'name' : 'oren',
        'videos' : [
            ('oren_speech_stevens_institute.mp4', 'https://www.youtube.com/watch?v=V2V0Yiy0Afs')
        ]
    }
}

oren_trump_complex.json
model = {
    'name' : 'oren_trump_complex',
#    'base_model' : None,
    'base_model' : 'oren_trump_simple',    
    'person_a' : {
        'name' : 'trump',
        'videos' : [
            ('trump_speech_compilation.mp4', 'https://www.youtube.com/watch?v=f0UB06v7yLY' ),
            ('foobar.mp4', 'https://www.youtube.com/watch?v=f0UB06v7yLY' ),            
        ]
    },
    'person_b' : {
        'name' : 'oren',
        'videos' : [
            ('oren_speech_stevens_institute.mp4', 'https://www.youtube.com/watch?v=V2V0Yiy0Afs')
        ]
    }
}

python faceit.py train oren_trump_simple.json
python faceit.py convert oren_trump_simple foo.mp4
```


## Notes

Download video command:
```
youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio' --merge-output-format mp4 -o "foobar.%(ext)s" https://www.youtube.com/watch?v=f0UB06v7yLY
```

Output template for using youtube id and extension:
```
            'outtmpl': os.path.join(Model.VIDEO_PATH, person + '_%(id)s.%(ext)s'),
```


# Some day later
#model.add_video('oren', 'input/oren/oren.mp4') # Existing video file
#model.add_images('oren', 'input/oren_images') # Folder of images
