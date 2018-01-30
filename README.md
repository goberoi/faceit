## FaceIt

FaceIt lets you swap one face for another in a YouTube video.

This code relies on the deepfakes/faceswap library.

## Usage

```
gaurav_simple = Person(
  name = 'gaurav_simple'
  videos = [ 'data/gaurav-short-video.mp4' ],
  images = [ 'data/gaurav_images_from_fb/' ]
)

trump_gaurav_simple = Model(
  name = 'trump_gaurav_simple',
  person_a = 'trump',
  person_b = 'gaurav',
  media_a = [ 'data/trump_speech.mp4', 'data/trump_photos' ],
  media_b = [ 'data/gaurav-short-video.mp4', 'data/gaurav-fb-photos' ],
)

trump_gaurav_simple.prepare_data()

trump_gaurav_simple.train(
  total_time = '240',
  checkpoint_every = '15'
)
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
