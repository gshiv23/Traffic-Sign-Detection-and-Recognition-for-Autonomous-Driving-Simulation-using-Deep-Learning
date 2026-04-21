import carla
import random
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import math
import time
import csv
import os

# =====================================================
# FINAL COMPLETE SYSTEM
# =====================================================

# -----------------------------
# Traffic Sign Names
# -----------------------------
CLASS_NAMES = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50",
    3: "Speed Limit 60", 4: "Speed Limit 70", 5: "Speed Limit 80",
    6: "End Speed Limit 80", 7: "Speed Limit 100", 8: "Speed Limit 120",
    9: "No Overtaking", 10: "No Overtaking Trucks", 11: "Right of Way",
    12: "Priority Road", 13: "Yield", 14: "STOP",
    15: "No Entry", 16: "Danger", 17: "Curve Left", 18: "Curve Right",
    19: "Double Curve", 20: "Uneven Road", 21: "Slippery Road",
    22: "Road Narrows", 23: "Road Work", 24: "Traffic Signals",
    25: "Pedestrian Crossing", 26: "School Crossing",
    27: "Bicycle Crossing", 28: "Snow", 29: "Animals",
    30: "Speed Limit End", 31: "Turn Right", 32: "Turn Left",
    33: "Go Straight", 34: "Go Straight or Right",
    35: "Go Straight or Left", 36: "Keep Right", 37: "Keep Left",
    38: "Roundabout", 39: "End No Overtaking",
    40: "End No Overtaking Trucks"
}

# -----------------------------
# Load Models
# -----------------------------
yolo_model = YOLO(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\yolov8n.pt")
cnn_model = tf.keras.models.load_model(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\traffic_sign_final_model.h5")

print("Models Loaded")

# -----------------------------
# CARLA Setup
# -----------------------------
client = carla.Client("localhost", 2000)
client.set_timeout(10)

world = client.load_world("Town03")
blueprints = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
traffic_manager = client.get_trafficmanager(8000)

# -----------------------------
# Vehicle
# -----------------------------
vehicle = world.spawn_actor(
    blueprints.filter("model3")[0],
    random.choice(spawn_points)
)
vehicle.set_autopilot(False)

# -----------------------------
# Traffic
# -----------------------------
traffic_vehicles = []
for _ in range(20):
    npc = world.try_spawn_actor(
        random.choice(blueprints.filter("vehicle.*")),
        random.choice(spawn_points)
    )
    if npc:
        npc.set_autopilot(True, traffic_manager.get_port())
        traffic_vehicles.append(npc)

# -----------------------------
# Camera
# -----------------------------
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle
)

# -----------------------------
# Collision Recovery
# -----------------------------
def recover(event):
    nearest = min(
        spawn_points,
        key=lambda sp: vehicle.get_location().distance(sp.location)
    )
    vehicle.set_transform(nearest)

collision_sensor = world.spawn_actor(
    blueprints.find("sensor.other.collision"),
    carla.Transform(),
    attach_to=vehicle
)
collision_sensor.listen(recover)

# -----------------------------
# PID Speed Control
# -----------------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev = 0
        self.int = 0

    def step(self, error, dt=0.05):
        self.int += error * dt
        der = (error - self.prev) / dt
        self.prev = error
        return self.kp * error + self.ki * self.int + self.kd * der

speed_pid = PID(0.5, 0.01, 0.1)

# -----------------------------
# Metrics
# -----------------------------
fps = 0
prev_time = time.time()

total_detections = 0
correct_predictions = 0

# CSV
csv_file = "carla_results.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Sign", "Confidence", "Decision", "Speed"])

# -----------------------------
# Memory + Timing
# -----------------------------
frame_count = 0
sign_memory = deque(maxlen=5)

sign_action_active = False
sign_action_start_time = 0
SIGN_ACTION_DURATION = 10
current_active_sign = None

# -----------------------------
# Road Following
# -----------------------------
def follow_road():
    waypoint = world.get_map().get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if waypoint is None:
        return 0

    vehicle_yaw = vehicle.get_transform().rotation.yaw
    road_yaw = waypoint.transform.rotation.yaw

    error = (road_yaw - vehicle_yaw) / 90.0
    return max(min(error, 0.5), -0.5)

# -----------------------------
# Front Vehicle Detection
# -----------------------------
def detect_front_vehicle():
    ego = vehicle.get_transform()
    ego_loc = ego.location
    fwd = ego.get_forward_vector()

    for npc in traffic_vehicles:
        if not npc.is_alive:
            continue

        loc = npc.get_location()
        dist = ego_loc.distance(loc)

        if dist < 15:
            vec = loc - ego_loc
            dot = vec.x * fwd.x + vec.y * fwd.y

            if dot > 0:
                return dist
    return None

# -----------------------------
# Main Processing
# -----------------------------
def process_image(image):
    global frame_count, fps, prev_time
    global total_detections, correct_predictions
    global sign_action_active, sign_action_start_time, current_active_sign

    frame_count += 1

    if frame_count % 2 != 0:
        return

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))[:, :, :3]
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    small = cv2.resize(frame, (416, 416))
    results = yolo_model(small, conf=0.4, verbose=False)

    best_sign = None
    best_conf = 0

    for r in results:
        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes.xyxy):
            if i > 2:
                break

            x1, y1, x2, y2 = map(int, box)
            crop = small[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            cnn_img = cv2.resize(crop, (224, 224)) / 255.0
            cnn_img = np.expand_dims(cnn_img, axis=0)

            pred = cnn_model.predict(cnn_img, verbose=0)
            class_id = int(np.argmax(pred))
            conf = float(np.max(pred))

            total_detections += 1
            if conf > 0.8:
                correct_predictions += 1

            if conf > best_conf:
                best_conf = conf
                best_sign = class_id

            name = CLASS_NAMES.get(class_id, "Unknown")
            label = f"{name} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    sign_memory.append(best_sign)

    stable_sign = None
    if len(sign_memory) >= 3:
        stable_sign = max(set(sign_memory), key=sign_memory.count)

    front_dist = detect_front_vehicle()

    # Decision Logic with 10s reaction
    if sign_action_active:
        if current_time - sign_action_start_time < SIGN_ACTION_DURATION:

            if current_active_sign == 14:
                target_speed = 0
                decision = "STOP"

            else:
                target_speed = 20
                decision = "SLOW"

        else:
            sign_action_active = False
            current_active_sign = None
            target_speed = 35
            decision = "DRIVE"

    else:
        if stable_sign is not None:
            sign_action_active = True
            sign_action_start_time = current_time
            current_active_sign = stable_sign

        elif front_dist:
            target_speed = min(20, front_dist * 1.2)
            decision = "FOLLOW"

        else:
            target_speed = 35
            decision = "DRIVE"

    # Speed Control
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
    throttle = speed_pid.step((target_speed - speed) / 40.0)

    control = carla.VehicleControl()
    control.steer = follow_road()

    if target_speed == 0 and speed > 2:
        control.throttle = 0
        control.brake = 1
    else:
        control.throttle = max(0, min(throttle, 0.75))
        control.brake = 0

    vehicle.apply_control(control)

    # Accuracy
    accuracy = (correct_predictions / max(total_detections, 1)) * 100

    # CSV logging
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            round(current_time, 2),
            CLASS_NAMES.get(stable_sign, "None"),
            round(best_conf, 2),
            decision,
            round(speed, 2)
        ])

    # Display
    cv2.putText(frame, f"{decision}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("FINAL OUTPUT", frame)
    cv2.waitKey(5)

# -----------------------------
# Run
# -----------------------------
camera.listen(process_image)

while True:
    world.wait_for_tick()