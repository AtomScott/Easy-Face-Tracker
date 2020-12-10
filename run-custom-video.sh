echo $1


python  ~/github/japanese-audio-emotion/Easy-Face-Tracker/pyannote-video/scripts/pyannote-structure.py shot \
        --verbose \
        "$1" \
        "$1.shots.json"

python  ~/github/japanese-audio-emotion/Easy-Face-Tracker/pyannote-video/scripts/pyannote-face.py \
        track --verbose \
        "$1" \
        "$1.shots.json" \
        "$1.track.txt"
        

