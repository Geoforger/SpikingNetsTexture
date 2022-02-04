import shutil
import json
import time
import threading
import os
from core.sensor.tactile_sensor_neuro_pixels import TacTip_neuro
from cri.controller import ABBController
from cri.robot import SyncRobot, AsyncRobot
import pickle
import pandas as pd
import numpy as np
# -*- coding: utf-8 -*-


# from vsp.video_stream import CvVideoCamera, CvVideoDisplay, CvVideoOutputFile
# from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

######################################################################
# TO DO LIST:
# 1) Lower the first (0) test by 1mm to exert equal pressure [x]
# 2) Find way to easily visualise data [x]
######################################################################

def make_meta(meta_file,
              robot_tcp=[0, 0, 75, 0, 0, 0],
              base_frame=[0, 0, 0, 0, 0, 0],
              home_pose=[400, 0, 300, 180, 0, 180],
              work_frame=[452, -125, 53, 180, 0, 180],
              linear_speed=10,
              angular_speed=10,
              num_frames=1,
              tap_move=[[0, 0, 8, 0, 0, 0], [
                  50, 0, 8, 0, 0, 0], [50, 0, 0, 0, 0, 0]],
              poses_rng=[[-5, 0, -3, -15, -15, -45], [5, 0, 0, 15, 15, 45]],
              obj_poses=[125, 50, 0, 0, 0, 0],
              num_poses=1,
              num_trials=1,
              ):

    # Set differing heights for misprinted textures
    obj_poses[0] = [125, 50, 0, 0, 0, 0]

    data_dir = os.path.dirname(meta_file)

    video_dir = os.path.join(data_dir, 'videos')
    video_df_file = os.path.join(data_dir, 'targets_video.csv')

    meta = locals().copy()
    del meta['data_dir']
    return meta


def make_robot():
    return AsyncRobot(SyncRobot(ABBController(ip='192.168.125.1')))


def make_sensor():
    return TacTip_neuro(port=33859)


def collect_data(data_dir, tap_move, obj_poses, home_pose, base_frame, work_frame, robot_tcp, linear_speed, angular_speed, num_trials, **kwargs):
    with make_robot() as robot, make_sensor() as sensor:

        # move to home position
        print("Moving to home position ...")
        robot.coord_frame = base_frame
        robot.move_linear(home_pose)

        for r in range(num_trials):

            robot.tcp = robot_tcp

            # Set TCP, linear speed,  angular speed and coordinate frame
            robot.linear_speed = linear_speed
            robot.angular_speed = angular_speed

            # Display robot info
            print("Robot info: {}".format(robot.info))

            # Display initial pose in work frame
            print("Initial pose in work frame: {}".format(robot.pose))

            # Move to origin of work frame
            print("Moving to origin of work frame ...")
            robot.coord_frame = work_frame
            robot.linear_speed = 50
            robot.move_linear((0, 0, 0, 0, 0, 0))
            obj_idx = 0

            for pose in obj_poses:

                sensor.set_variables()

                robot.linear_speed = 50
                robot.move_linear(pose)

                # time.sleep(1)
                robot.linear_speed = 10

                # Tap
                print("Trial 1/1 - Texture 11")
                robot.coord_frame = base_frame
                robot.coord_frame = robot.pose
                robot.move_linear(tap_move[0])
                time.sleep(0.5)   # Delay to prevent recording of initial press

                # Start sensor recording
                sensor.start_logging()
                t = threading.Thread(target=sensor.get_pixel_events, args=())
                t.start()

                robot.move_linear(tap_move[1])
                # Stop sensor recording
                sensor.stop_logging()
                t.join()

                robot.move_linear(tap_move[2])
                robot.coord_frame = work_frame

                # print("tap finished")

                # Collate proper timestamp values in ms.
                sensor.value_cleanup()
                # print("value cleaned")

                # Save data
                # pickle_out = open(os.path.join(data_dir, 'Artificial Dataset ' +
                #                                str(r) + 'Texture No. ' + str(obj_idx) + '.pickle'), 'wb')
                # pickle.dump(sensor.data, pickle_out)
                # pickle_out.close()
                # print("saved data")
                obj_idx += 1

            # # # Unwind sensor
            # print("Unwinding")
            # robot.move_linear([sum(x)
            #                    for x in zip(robot.pose, [0, 0, 0, 0, 0, -170])])
            # # robot.move_linear([sum(x) for x in zip(robot.pose, [0,0,0,0,0,-170])])

        # Move to home position
        print("Moving to home position ...")
        robot.coord_frame = base_frame
        robot.move_linear(home_pose)


def main():

    # Make and save metadata
    data_dir = os.path.join(os.environ['DATAPATH'], 'TacTip_NM', os.path.basename(
        __file__)[:-3]+'_'+time.strftime('%m%d%H%M'))
    meta_file = os.path.join(data_dir, 'meta.json')
    meta = make_meta(meta_file)
    os.makedirs(os.path.dirname(meta_file))
    with open(meta_file, 'w') as f:
        json.dump(meta, f)

    # Collect data
    collect_data(data_dir, **meta)


if __name__ == '__main__':
    main()
