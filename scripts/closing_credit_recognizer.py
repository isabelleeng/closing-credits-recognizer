import sys, os
import cv2
import math
import numpy as np
import datetime
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.get_logger().setLevel(logging.ERROR)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s: %(message)s')
LOGGER = logging.getLogger("ClosingCredits")
STARTING_POINT = 0.8
ACCEPTANCE_THRESHOLD = 0.9
AREAS = 3

# if len(sys.argv) < 2:
#     LOGGER.error("Missing arguments! You should provide to arguments to the script. First, path to the video and then path to the model.")
#     exit()

# video_path = sys.argv[1]
# model_path = sys.argv[2]

def predict(model, frames):
    prediction_classes = (model.predict(frames) > 0.5).astype("int32")
    estimates = np.array([x[0] for x in prediction_classes])
    return estimates

def get_starting_index(estimates, window_size=100):
    window = np.zeros((window_size,))
    count = 0
    for i in range(estimates.shape[0]-window_size):
        if count == 10:
            return index + 4
        if np.sum(estimates[i:(i+window_size)] == window)/window_size > 0.95:
            if count == 0:
                index = i
            count += 1
        else:
            count = 0
            index = None
    return None

def zero_pad(variable):
    if len(variable) > 3:
        return variable[:3]
    if int(variable) < 10:
        return f'0{int(variable)}'
    else:
        return str(int(variable))

def format_time(time_progress):
    formatted_time = str(datetime.timedelta(milliseconds=time_progress))
    formatted_time_split = formatted_time.split(':')

    hour = formatted_time_split[0]
    minute = formatted_time_split[1]
    second_split = formatted_time_split[2].split('.')
    second = second_split[0]
    ms = second_split[1]
    hour, minute, second, ms = zero_pad(hour), zero_pad(minute), zero_pad(second), zero_pad(ms)
    formatted_time = f"{hour}:{minute}:{second}.{ms}"
    return formatted_time

def save_frame(video_path, frame_id, time):
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = capture.read()
    capture.release()

    cv2.putText(frame, time, org=(100,100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2 , color=(255, 255, 0), thickness=2)
    cv2.imwrite(f"scripts/result/{video_path[12:-4]}_frame_{frame_id}.jpg", frame)

def main():
    video_path = 'data/videos/The_Tunnel-S01E03.mp4'
    model_path = 'notebooks/models/closing_credits_Resnet50.h5'

    model = load_model(model_path) # loading the pretrained model
    metadata = [] # Contains the timestamp (in milliseconds) and frame ID of all frames fed into the model
    frames = [] # Contains the frames themselves

    capture = cv2.VideoCapture(video_path)

    width = capture.get(3)
    height = capture.get(4)
    cutoff = int((width - height)/2)
    frame_rate = capture.get(5)
    total_frames = capture.get(7)
    
    LOGGER.info(f"\nMovie metadata - width: {width}, height: {height}, framerate: {frame_rate}, "
            f"total_frames: {total_frames}, currentframe: {capture.get(1)}")

    while(capture.isOpened()):
        frame_info = {"time_progress": capture.get(0),
                    "frame_id": capture.get(1)}
        ret, frame = capture.read()
        if ret != True:
            break
        if frame_info['frame_id']/total_frames > 0.75 and frame_info['frame_id'] % math.floor(frame_rate/10) == 0:
            metadata.append(frame_info)
            frame = frame[:, cutoff:-cutoff, :]
            frame = cv2.resize(frame, (224, 224))/255.0
            frames.append(frame)

    frames = np.array(frames)
    capture.release()

    LOGGER.info("finished preprocessing frames.")

    estimates = predict(model, frames)
    LOGGER.info("prediction stage is done.")

    credits_info = metadata[get_starting_index(estimates)]

    formatted_time = format_time(credits_info['time_progress'])
    LOGGER.info(f"Credits started rolling at {formatted_time}, at {int(credits_info['frame_id'])} frame.")

    save_frame(video_path, int(credits_info['frame_id']), formatted_time)


if __name__ == '__main__':
    main()