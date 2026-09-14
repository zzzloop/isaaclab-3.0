# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run in the existing ROS1 environment; subscribe to raw Pico topics, send UDP only.

Deliberately compatible with ROS Noetic Python 3.8. No robot command publishers.
"""

import argparse
import json
import math
import socket
import threading
import time
import uuid


def main():
    """Forward fresh, synchronized-enough Pico samples to the simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15051)
    parser.add_argument("--rate", type=float, default=30)
    parser.add_argument("--max_age", type=float, default=0.25, help="Maximum per-topic receipt age [s]")
    parser.add_argument("--max_skew", type=float, default=0.15, help="Maximum pose header timestamp skew [s]")
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.rate, args.max_age, args.max_skew)):
        parser.error("rate, max_age and max_skew must be finite and positive")
    import rospy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Joy

    rospy.init_node("am_dp123_pico_sim_bridge", anonymous=True)
    samples, lock = {}, threading.Lock()

    def receive(message, key):
        with lock:
            samples[key] = (message, time.monotonic())

    topics = {
        "head": ("/teleop/pose/hmd", PoseStamped),
        "left": ("/teleop/pose/left_controller", PoseStamped),
        "right": ("/teleop/pose/right_controller", PoseStamped),
        "left_joy": ("/teleop/controller/left_joy", Joy),
        "right_joy": ("/teleop/controller/right_joy", Joy),
    }
    subscribers = [
        rospy.Subscriber(topic, kind, receive, callback_args=key, queue_size=1) for key, (topic, kind) in topics.items()
    ]
    session, seq, last_receipts = uuid.uuid4().hex, 0, None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rospy.loginfo("Pico -> simulation UDP %s:%s; subscribes only", args.host, args.port)
    try:
        while not rospy.is_shutdown():
            start = time.monotonic()
            with lock:
                frame = dict(samples)
            if len(frame) == len(topics) and all(start - value[1] <= args.max_age for value in frame.values()):
                receipts = tuple(frame[key][1] for key in ("head", "left", "right"))
                stamps = [frame[key][0].header.stamp.to_sec() for key in ("head", "left", "right")]
                if receipts != last_receipts and max(stamps) - min(stamps) <= args.max_skew:
                    packet = {"version": 1, "session": session, "seq": seq}
                    for key in ("head", "left", "right"):
                        pose = frame[key][0].pose
                        packet[key] = [
                            pose.position.x,
                            pose.position.y,
                            pose.position.z,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ]
                    complete = True
                    for side in ("left", "right"):
                        joy = frame[side + "_joy"][0]
                        if len(joy.axes) < 1 or len(joy.buttons) < 4:
                            complete = False
                            break
                        packet[side + "_joy"] = {
                            "trigger": float(joy.axes[0]),
                            "a": bool(joy.buttons[2]),
                            "b": bool(joy.buttons[3]),
                        }
                    if complete:
                        try:
                            sock.sendto(json.dumps(packet, allow_nan=False).encode(), (args.host, args.port))
                            seq += 1
                            last_receipts = receipts
                        except ValueError:
                            rospy.logwarn_throttle(2, "Non-finite Pico sample discarded")
            else:
                rospy.logwarn_throttle(3, "Waiting for fresh messages on all five Pico topics")
            time.sleep(max(0, 1 / args.rate - (time.monotonic() - start)))
    finally:
        sock.close()
        for subscriber in subscribers:
            subscriber.unregister()


if __name__ == "__main__":
    main()
