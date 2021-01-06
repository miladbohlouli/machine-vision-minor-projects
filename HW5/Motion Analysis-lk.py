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
    for frame in tqdm(frames):
        out.write(frame)
    out.release()


def preprocess_frame(frame, smooth=False, return_red=False):
    if return_red:
        frame = frame[:, :, 2]
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if smooth:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame


def check_in_region(a, b):
    return lambda x: np.logical_and(a < x, x < b)


lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

feature_params = dict(
    maxCorners=30,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7
)
recalculate_every_n_frame = 30


# Choose the scene from the below cases:
scene = 10
name = "scene" + str(scene)
##############################################################################
#                               Scene 1
##############################################################################
if scene == 1:
    recalculate_every_n_frame = 100
    first_frame, final_frame = 102, 291

##############################################################################
#                               Scene 2
##############################################################################
elif scene == 2:
    first_frame, final_frame = 292, 448

##############################################################################
#                               Scene 3
##############################################################################
elif scene == 3:
    first_frame, final_frame = 449, 600
    recalculate_every_n_frame = 20
    feature_params = dict(
        maxCorners=30,
        qualityLevel=0.01,
        minDistance=200,
        blockSize=7
    )

##############################################################################
#                               Scene 4
##############################################################################
elif scene == 4:
    first_frame, final_frame = 601, 816
    recalculate_every_n_frame = 20
    feature_params = dict(
        maxCorners=40,
        qualityLevel=0.01,
        minDistance=300,
        blockSize=7
    )

##############################################################################
#                               Scene 5
##############################################################################
elif scene == 5:
    first_frame, final_frame = 817, 1077
    recalculate_every_n_frame = 20
    feature_params = dict(
        maxCorners=30,
        qualityLevel=0.01,
        minDistance=100,
        blockSize=7
    )

##############################################################################
#                               Scene 6
##############################################################################
elif scene == 6:
    first_frame, final_frame = 1078, 1328
    recalculate_every_n_frame = 100
    feature_params = dict(
        maxCorners=50,
        qualityLevel=0.001,
        minDistance=10,
        blockSize=7
    )


##############################################################################
#                               Scene 7
##############################################################################
elif scene == 7:
    first_frame, final_frame = 1329, 1525
    recalculate_every_n_frame=10
    feature_params = dict(
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=200,
        blockSize=7
    )

##############################################################################
#                               Scene 8
##############################################################################
elif scene == 8:
    first_frame, final_frame = 1526, 1745
    recalculate_every_n_frame = 30
    feature_params = dict(
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=200,
        blockSize=7
    )

##############################################################################
#                               Scene 9
##############################################################################
elif scene == 9:
    first_frame, final_frame = 1746, 2009
    recalculate_every_n_frame = 10
    feature_params = dict(
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=200,
        blockSize=7
    )

##############################################################################
#                               Scene 10
##############################################################################
elif scene == 10:
    first_frame, final_frame = 2010, 2242
    recalculate_every_n_frame = 20
    feature_params = dict(
        maxCorners=30,
        qualityLevel=0.01,
        minDistance=100,
        blockSize=7
    )

if __name__ == '__main__':
    frames = read_video()[first_frame:final_frame]

    old_gray = preprocess_frame(frames[0])

    old_corners = cv2.goodFeaturesToTrack(old_gray, **feature_params)
    initial_corners = old_corners[:, 0, :]
    displacement = np.zeros(len(old_corners))
    mask = np.zeros_like(frames[0])
    colors = np.random.randint(0, 255, (100, 3))
    new_frames_scene = []

    # processing the video
    for i, frame in tqdm(enumerate(frames[1:])):
        new_gray = preprocess_frame(frame)
        new_corners, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_corners, None, **lk_params)
        displacement += (np.linalg.norm(new_corners[:, 0, :] - initial_corners, axis=1) / recalculate_every_n_frame)

        for j, (new, old) in enumerate(zip(new_corners[:, 0, :], old_corners[:, 0, :])):
            a, b = new.ravel()
            c, d = old.ravel()

            if st[j] and displacement[j] > 0.5:
                mask = cv2.line(mask, (a, b), (c, d), colors[j].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, colors[j].tolist(), -1)
        img = cv2.add(frame, mask)
        new_frames_scene.append(img)
        cv2.imshow("frame", img)

        if cv2.waitKey(30) & 0xff == ord("q"):
            break

        old_gray = new_gray.copy()
        if i % recalculate_every_n_frame == 1:
            old_corners = cv2.goodFeaturesToTrack(new_gray, **feature_params)
            initial_corners = old_corners[:, 0, :]
            displacement = np.zeros(len(old_corners))

        else:
            old_corners = new_corners.reshape((-1, 1, 2))

    save_video(new_frames_scene, f"results1/{name}.avi")
