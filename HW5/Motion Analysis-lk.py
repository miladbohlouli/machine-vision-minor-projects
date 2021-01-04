import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def show_frame(frame, name):
    cv2.imshow(name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_video(frames: list, name="frame"):
    for i, frame in enumerate(frames):
        cv2.imshow(name, frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def read_video(path="videoplayback.mp4", show=False):
    cap = cv2.VideoCapture(path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()

    if show:
        show_video(frames, "original movie")
    return frames


def save_video(frames, save_url="saved_video.avi", fps=25):
    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(save_url, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, (frame_width, frame_height))
    print("Saving the video...")
    for frame in tqdm(frames):
        out.write(frame)
    out.release()


def preprocess_frame(frame, smooth=True):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if smooth:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame


if __name__ == '__main__':
    frames = read_video()[102:]

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    feature_params = dict(
        maxCorners = 100,
        qualityLevel = 0.1,
        minDistance = 7,
        blockSize = 7
    )

    old_gray = preprocess_frame(frames[0], smooth=False)
    old_corners = cv2.goodFeaturesToTrack(old_gray, **feature_params)
    mask = np.zeros_like(frames[0])
    color = np.random.randint(0, 255, (100, 3))

    # processing the video
    for i, frame in tqdm(enumerate(frames[1:])):
        new_gray = preprocess_frame(frame, smooth=False)
        new_corners, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_corners, None, **lk_params)

        print(st)
        new_good = new_corners[st == 1]
        old_good = old_corners[st == 1]

        for i, (new, old) in enumerate(zip(new_good, old_good)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow("frame", img)

        if cv2.waitKey(30) & 0xff == ord("q"):
            break

        old_gray = new_gray.copy()
        if i % 50 == 0:
            mask = np.zeros_like(frames[0])
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            old_corners = cv2.goodFeaturesToTrack(new_gray, **feature_params)
        else:
            old_corners = new_good.reshape((-1, 1, 2))




