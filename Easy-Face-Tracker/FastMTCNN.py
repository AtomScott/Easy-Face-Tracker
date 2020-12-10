"""
FastMTCNN from https://www.kaggle.com/timesler/fast-mtcnn-detector-55-fps-at-full-resolution
All credits to this guy.
"""

from cv2 import cv2
from facenet_pytorch import MTCNN
from copy import deepcopy
from track_tools import Face

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
                      
        boxes, probs, landmarks = self.mtcnn.detect(frames[::self.stride], landmarks=True)

        faces = [[]] * len(frames)
        prev_ind = -1
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            if prev_ind == box_ind:
                faces[i] = deepcopy(faces[i-1])

            else:
                for box, ls in zip(boxes[box_ind], landmarks[box_ind]):
                    box = [[int(box[1]), int(box[3])], [int(box[0]), int(box[2])]]
                    ls = [[int(l[0]), int(l[1])] for l in ls]
                    
                    face = Face(idx=i, img=frame, box=box, landmarks=landmarks)
                    faces[i].append(face)

            prev_ind = box_ind

        return faces