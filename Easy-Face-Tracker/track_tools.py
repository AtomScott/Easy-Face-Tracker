from functools import cached_property

import facenet_pytorch
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten
from scipy.interpolate import interp1d
from cv2 import cv2
from torchvision.transforms import functional as F
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


class Face:
    def __init__(self, idx, img, box, landmarks):
        self.idx = idx
        self.roi = img[box[0][0] : box[0][1], box[1][0] : box[1][1]]
        self.box = box
        self.landmarks = landmarks

    @cached_property
    def embedding(self):
        img = cv2.resize(self.roi, (160, 160), interpolation=cv2.INTER_AREA).copy()
        tensor = F.to_tensor(np.float32(img))
        aligned = torch.stack([prewhiten(tensor)]).to(device)
        embedding = resnet(aligned).detach().cpu().numpy()[0]
        return embedding


class Track:
    def __init__(self, face, ID=0):
        
        self.ID=ID

        # Kalman stuff
        x1, x2, y1, y2 = np.ravel(face.box)
        x = np.mean((x1, x2))
        y = np.mean((y1, y2))
        s = np.linalg.norm((x1 - x2, y1 - y2))

        self.dt = dt = 0.1
        
        self.state_x = np.array([x, y, s, 0, 0, 0])
        self.state_prev_x = self.state_x

        self.state_cov = P = np.diag(np.ones(self.state_x.shape))

        self.H = np.asarray(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # confidence probably needs tuning
        conf = 1
        self.R = np.diag(np.ones(len(self.state_x))) * conf

        # Deep sort stuff
        self.gallery = []
        self.none_count = 0
        pass
    
    @property
    def dataframe(self):
        _df = []
        for face in self.gallery:
            _df.append({
                'Frame No.': face.idx,
                'Track ID': self.ID,
                'x1': face.box[0][0],
                'y1': face.box[0][1],
                'x2': face.box[1][0],
                'y2': face.box[1][1],
            })

        df = pd.DataFrame(_df)
        return df

    def get_track_start(self):
        return self.gallery[0].idx

    def get_track_end(self):
        return self.gallery[-1].idx

    def predict_state(self):
        x_now = self.state_x
        P_now = self.state_cov
        H = self.H

        x_pred = H @ x_now
        P_pred = H @ P_now @ H.T

        return x_pred, P_pred

    def update(self, face):
        if face is None:
            self.none_count += 1
        else:
            self.update_gallery(face)
            self.update_state(face)
            self.none_count = 0
        return

    def update_state(self, face):
        z = self.format_measurement(face)

        x_now = self.state_x
        P_now = self.state_cov

        H = self.H
        R = self.R

        K = P_now @ H.T @ np.linalg.inv(H @ P_now @ H.T + R)

        x_next = x_now + K @ (z - H @ x_now)
        P_next = P_now - K @ H @ P_now

        self.state_prev_x = self.state_x
        self.state_x = x_next
        self.state_cov = P_next / np.linalg.norm(P_next)

        return

    def format_measurement(self, face):
        x1, x2, y1, y2 = np.ravel(face.box)
        _, _, _, x_prev, y_prev, s_prev = self.state_prev_x

        x = np.mean((x1, x2))
        y = np.mean((y1, y2))
        s = np.linalg.norm((x1 - x2, y1 - y2))
        xv = x - x_prev
        yv = y - y_prev
        sv = s - s_prev
        z = np.array([x, y, s, xv, yv, sv])
        return z

    def update_gallery(self, face):
        self.gallery.append(face)
        pass

    def get_bboxes(self):
        """
        x1, y1, x2, y2 = get_bboxes(track)
        """

        start_t = self.gallery[0].idx
        end_t = self.gallery[-1].idx

        xs = np.zeros((end_t + 1 - start_t, 4))

        idxs_old = []
        bboxes_old = []
        for face in self.gallery:
            idxs_old.append(face.idx)
            bboxes_old.append(face.box)

        idxs_new = np.arange(start_t, end_t)
        bboxes_new = []

        ys = np.asarray(bboxes_old)
        for i in range(4):
            f = interp1d(np.asarray(idxs_old), ys[:, i])
            bboxes_new.append(f(idxs_new))

        return np.asarray(bboxes_new)
