import json
import math
import cv2
import numpy as np


def render_posenet(
    input_file: str,
    output_file: str,
    size=(768, 1024),
    stickwidth=4,
):
    width, height = size
    img = np.zeros((height, width, 3))
    pose_keypoints: list[float] = json.load(open(input_file))["people"][0][
        "pose_keypoints_2d "
    ]

    poses = [[pose_keypoints[i : i + 3] for i in range(0, len(pose_keypoints), 3)]]

    limb_seq = [
        [1, 0],
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 5],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [8, 12],
        [12, 13],
        [13, 14],
        [0, 15],
        [0, 16],
        [15, 17],
        [16, 18],
        [11, 24],
        [11, 22],
        [14, 21],
        [14, 19],
        [22, 23],
        [19, 20],
    ]
    njoint = 25

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 255, 255],
        [170, 255, 255],
        [85, 255, 255],
        [0, 255, 255],
    ]
    for i in range(njoint):
        for n in range(len(poses)):
            _pose = poses[n][i]
            if _pose[2] <= 0:
                continue
            x, y = _pose[:2]
            cv2.circle(img, (int(x), int(y)), 4, colors[i], thickness=-1)

    for pose in poses:
        for limb, color in zip(limb_seq, colors):
            p1 = pose[limb[0]]
            p2 = pose[limb[1]]
            if p1[2] <= 0 or p2[2] <= 0:
                continue
            cur_canvas = img.copy()
            X = [p1[1], p2[1]]
            Y = [p1[0], p2[0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = np.array(
                cv2.ellipse2Poly(
                    (int(mY), int(mX)),
                    (int(length / 2), stickwidth),
                    int(angle),
                    0,
                    360,
                    1,
                )
            )
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img = np.array(cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0))
    cv2.imwrite(output_file, img)
