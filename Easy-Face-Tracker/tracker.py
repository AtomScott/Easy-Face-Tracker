import glob
import itertools as it
import time
import sys
from collections import defaultdict
from functools import cached_property

import cv2
import numpy as np
import PIL
import torch
from cv2 import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten
from imutils.video import FileVideoStream
from loguru import logger
from PIL import Image
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis
from tqdm.notebook import tqdm
from fire import Fire

# My scripts
from FastMTCNN import FastMTCNN
from track_tools import Face, Track

logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO", colorize=True)

# logger.add(lambda msg: tqdm.write(msg, end=""))
logger.info("Initialzing")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# mtcnn = MTCNN(keep_all=True, device=device)


def format_input(X, y):
    X = [X[np.where(y == t)] for t in np.unique(y)]
    return X, np.unique(y)


class FaceTracker(object):
    """"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fast_mtcnn = FastMTCNN(
        stride=4,
        resize=1,
        margin=14,
        factor=0.6,
        keep_all=True,
        device=device,
        post_process=False,
    )

    def __init__(self):
        pass
    
    def demo(self):
        videofile = "./data/sample/sample.mkv"
        face_tracker = FaceTracker()
        faces = face_tracker.run_detection(videofile)
        tracks = face_tracker.run_association(faces)

        for track in tracks:
            print(track.dataframe)

    def run_detection(self, videofile):
        """
        Based on https://www.kaggle.com/timesler/fast-mtcnn-detector-55-fps-at-full-resolution
        """
        frames = []        
        faces = []
        start = time.time()
        batch_size = 60

        v_cap = FileVideoStream(videofile).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in tqdm(range(v_len)):

            frame = v_cap.read()
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if len(frames) >= batch_size or j == v_len - 1:
                faces += FaceTracker.fast_mtcnn(frames)
                frames = []
                print(
                    f'Frames per second: {len(faces) / (time.time() - start):.3f}',
                    f'Frames Processed: {len(faces)} / {v_len}\r',
                    end=''
                )

        v_cap.stop()

        return faces

    def run_association(
        self,
        facess,
        lam=0.1,
        thresh_1=0.5,
        thresh_2=0.5,
        kill_thresh=2,
        frame_idx_thresh=29,
    ):

        tracks_alive = []
        tracks_dead = []
        salt = 1 / 10 ** 9  # avoids zero division
        pepper = 10 ** 6  # avoids infinity
        ID = 0
        prev_frame_idx = 0
        for frame_idx, faces in tqdm(enumerate(facess)):
            # Print tracks
            logger.debug(f'Frame: {frame_idx}, Tracks alive {len(tracks_alive)}, Tracks dead: {len(tracks_dead)}')
            for track in tracks_alive:
                logger.debug(f'Track ID: {track.ID}')

            ###########################
            # * Calculate cost matrices
            ###########################
            C = association_cost_matrix(faces, tracks_alive, lam)
            G = gate_matrix(faces, tracks_alive, thresh_1, thresh_2) + salt
            gated_cost_matrix = C / G
            assert C.shape == G.shape, f"{C.shape}, {G.shape}"

            face_idxs, track_idxs = linear_sum_assignment(gated_cost_matrix)

            ##########################
            # * Assign faces to tracks
            ##########################
            # - faces[face_idx] is assigned to tracks_alive[track_idx]
            # - Remember faces that have not been assigned
            # - Remember tracks that are to be killed
            faces_unassigned = (
                list(face_idxs) if face_idxs != [] else list(range(len(faces)))
            )
            tracks_to_kill = []
            for face_idx, track_idx in zip(face_idxs, track_idxs):
                if gated_cost_matrix[face_idx, track_idx] < pepper:
                    tracks_alive[track_idx].update(faces[face_idx])
                    faces_unassigned.remove(face_idx)
                else:
                    tracks_alive[track_idx].update(None)

            for track_idx, track in enumerate(tracks_alive):
                if track_idx not in track_idxs:
                    tracks_alive[track_idx].update(None)
                if track.none_count >= kill_thresh:
                    tracks_dead.append(track)
                    tracks_to_kill.append(track_idx)

            ####################################
            # * Kill tracks/Generate new tracks
            ####################################
            # - Generate new tracks for new faces
            # - Terminate track with continuous null updates
            for track_idx in sorted(tracks_to_kill, reverse=True):
                tracks_dead.append(tracks_alive[track_idx])
                del tracks_alive[track_idx]

            for face_idx in faces_unassigned:
                new_track = Track(faces[face_idx], ID=ID)
                ID+=1
                tracks_alive.append(new_track)

            ################
            # * Reset tracks
            ################
            # - Tracks are reset if large gap between previous frame
            if frame_idx - prev_frame_idx > frame_idx_thresh:
                for track in tracks_alive:
                    tracks_dead.append(track)
                tracks_alive = []

            prev_frame_idx = frame_idx

        for track in tracks_alive:
            tracks_dead.append(track)

        return tracks_dead


def _d_1(face, track):
    """Motion Descriptor"""
    assert type(face) == Face
    assert type(track) == Track

    z = track.format_measurement(face)  # new measurement vector

    y, S = track.predict_state()  # should be next (x,y) predict by kalman

    # normalize each array
    z = z / np.linalg.norm(z)
    y = y / np.linalg.norm(y)
    S = S / np.linalg.norm(S)

    dist = mahalanobis(z, y, S)
    return dist


def _d_2(face, track):
    """Appearance Descriptor"""
    return 0
    assert type(face) == Face
    assert type(track) == Track

    dist = 1
    r_j = face.embedding
    for _face in track.gallery:
        r_i = _face.embedding
        r_j_dist = r_j.T @ r_i
        # assert 0 <= r_j_dist <= 1
        if not 0 <= r_j_dist <= 1:
            logger.warning(r_j_dist)
        dist = min(dist, r_j_dist)
    return 1 - dist


def association_cost_matrix(faces, tracks, lam=0.1):
    n_faces = len(faces)
    n_tracks = len(tracks)
    C = np.zeros((n_faces, n_tracks))
    for i, j in it.product(range(n_faces), range(n_tracks)):
        C[i, j] = lam * _d_1(faces[i], tracks[j]) + (1 - lam) * _d_2(
            faces[i], tracks[j]
        )
    return C


def gate_matrix(faces, tracks, _thresh_1=50, _thresh_2=1):
    n_faces = len(faces)
    n_tracks = len(tracks)

    G = np.zeros((n_faces, n_tracks))
    for i, j in it.product(range(n_faces), range(n_tracks)):
        G[i, j] = (_d_1(faces[i], tracks[j]) <= _thresh_1) * (
            _d_2(faces[i], tracks[j]) <= _thresh_2
        )
    G = G.astype(int)

    return G

if __name__ == "__main__":
    Fire(FaceTracker)
