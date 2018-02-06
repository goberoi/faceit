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
trump_gaurav_simple = Model(
  name = 'trump_gaurav_simple',
  media_a = [ 'data/trump_speech.mp4', 'data/trump_photos' ],
  media_b = [ 'data/gaurav-short-video.mp4', 'data/gaurav-fb-photos' ],
)

trump_gaurav_simple.prepare_data()

trump_gaurav_simple.train(
  total_time = '240',
  checkpoint_every = '15'
)

trump_gaurav_simple.convert('gaurav-short-video.mp4')
```

```
python faceit.py --add_video gaurav1.mp4

python faceit.py --add_video trump https://www.youtube.com/watch?v=f0UB06v7yLY
python faceit.py --add_video oren https://www.youtube.com/watch?v=V2V0Yiy0Afs
python faceit.py --train "oren trump 1 hour" oren trump
python faceit.py --convert trump oren https://www.youtube.com/watch?v=f0UB06v7yLY

python faceit.py --add_video oren https://www.youtube.com/watch?v=ZCsrUI9kGII&t=2659s

python faceit.py --convert trump oren https://www.youtube.com/watch?v=f0UB06v7yLY
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
data/media/
    trump_speech.mp4
    trump_photos/
        photo1.jpg
        photo2.jpg
    gaurav_short_video.mp4
    oren_machine_learning_speech.mp4

data/training/
    trump_speech.mp4/
        frame0.jpg
	frame2.jpg
	...

data/output/
    
```

```
data/
    people/
        videos/
	    video_0/
	        url.txt
		video_0.mp4
	faces
    models/
```


def fetch_video(url):
    """Download or load video from cache and return filename."""
    return filename

## Notes

Download video command:
```
youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio' --merge-output-format mp4 -o "foobar.%(ext)s" https://www.youtube.com/watch?v=f0UB06v7yLY
```

Output template for using youtube id and extension:
```
            'outtmpl': os.path.join(Model.VIDEO_PATH, person + '_%(id)s.%(ext)s'),
```