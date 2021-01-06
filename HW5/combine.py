import cv2
import numpy as np
from tqdm import tqdm

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
    for frame in tqdm(frames):
        out.write(frame)
    out.release()


if __name__ == '__main__':
    frames = []
    for i in range(1, 11):
        frames += read_video(f"results1/scene{i}.avi")

    save_video(frames, "result1.avi")
