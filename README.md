## FaceIt

FaceIt lets you swap one face for another in a YouTube video.

This code relies on the deepfakes/faceswap library.

## Product Notes


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
