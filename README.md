# Easy Face Tracker

Install [pyannote-video](https://github.com/pyannote/pyannote-video) to use this.

## Example Usage

1. Download a sample video with [youtube-dl](http://ytdl-org.github.io/youtube-dl/download.html)
```
youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]' \
 -o sample \
 https://www.youtube.com/watch?v=MevKTPN4ozw 
```

2. Detect faces with pyannote-video

To use pyannote-video as the face tracker use the code below:
```
python pyannote-video/scripts/pyannote-structure.py shot \
        --verbose \
        sample.mkv \
        sample.shots.json

python pyannote-video/scripts/pyannote-face.py \
        track --verbose \
        sample.mkv \
        sample.shots.json \
        sample.track.txt
```

To use Easy Face Tracker as the face tracker use the code below:
```

```

3. Build face track videos from tracked videos
```

```
