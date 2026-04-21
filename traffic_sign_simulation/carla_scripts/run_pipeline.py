# import carla
# import random
# import cv2
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# from collections import deque
# import math
# import time
# import csv
# import os

# # =====================================================
# # FINAL STABLE VERSION (NO CRASH + FULL FEATURES)
# # =====================================================

# # -----------------------------
# # Traffic Sign Names
# # -----------------------------
# CLASS_NAMES = {
#     0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50",
#     3: "Speed Limit 60", 4: "Speed Limit 70", 5: "Speed Limit 80",
#     6: "End Speed Limit 80", 7: "Speed Limit 100", 8: "Speed Limit 120",
#     9: "No Overtaking", 10: "No Overtaking Trucks", 11: "Right of Way",
#     12: "Priority Road", 13: "Yield", 14: "STOP",
#     15: "No Entry", 16: "Danger", 17: "Curve Left", 18: "Curve Right",
#     19: "Double Curve", 20: "Uneven Road", 21: "Slippery Road",
#     22: "Road Narrows", 23: "Road Work", 24: "Traffic Signals",
#     25: "Pedestrian Crossing", 26: "School Crossing",
#     27: "Bicycle Crossing", 28: "Snow", 29: "Animals",
#     30: "Speed Limit End", 31: "Turn Right", 32: "Turn Left",
#     33: "Go Straight", 34: "Go Straight or Right",
#     35: "Go Straight or Left", 36: "Keep Right", 37: "Keep Left",
#     38: "Roundabout", 39: "End No Overtaking",
#     40: "End No Overtaking Trucks"
# }

# # -----------------------------
# # Load Models
# # -----------------------------
# yolo_model = YOLO(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\yolov8n.pt")
# cnn_model = tf.keras.models.load_model(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\traffic_sign_final_model.h5")

# print("Models Loaded")

# # -----------------------------
# # CARLA Setup
# # -----------------------------
# client = carla.Client("localhost", 2000)
# client.set_timeout(10)

# world = client.load_world("Town03")
# blueprints = world.get_blueprint_library()
# spawn_points = world.get_map().get_spawn_points()
# traffic_manager = client.get_trafficmanager(8000)

# # -----------------------------
# # Vehicle
# # -----------------------------
# vehicle = world.spawn_actor(
#     blueprints.filter("model3")[0],
#     random.choice(spawn_points)
# )
# vehicle.set_autopilot(False)

# # -----------------------------
# # Traffic
# # -----------------------------
# traffic_vehicles = []
# for _ in range(20):
#     npc = world.try_spawn_actor(
#         random.choice(blueprints.filter("vehicle.*")),
#         random.choice(spawn_points)
#     )
#     if npc:
#         npc.set_autopilot(True, traffic_manager.get_port())
#         traffic_vehicles.append(npc)

# # -----------------------------
# # Camera
# # -----------------------------
# camera_bp = blueprints.find("sensor.camera.rgb")
# camera_bp.set_attribute("image_size_x", "640")
# camera_bp.set_attribute("image_size_y", "480")

# camera = world.spawn_actor(
#     camera_bp,
#     carla.Transform(carla.Location(x=1.5, z=2.4)),
#     attach_to=vehicle
# )

# # -----------------------------
# # Collision Recovery
# # -----------------------------
# def recover(event):
#     nearest = min(
#         spawn_points,
#         key=lambda sp: vehicle.get_location().distance(sp.location)
#     )
#     vehicle.set_transform(nearest)

# collision_sensor = world.spawn_actor(
#     blueprints.find("sensor.other.collision"),
#     carla.Transform(),
#     attach_to=vehicle
# )
# collision_sensor.listen(recover)

# # -----------------------------
# # PID Speed Control
# # -----------------------------
# class PID:
#     def __init__(self, kp, ki, kd):
#         self.kp, self.ki, self.kd = kp, ki, kd
#         self.prev = 0
#         self.int = 0

#     def step(self, error, dt=0.05):
#         self.int += error * dt
#         der = (error - self.prev) / dt
#         self.prev = error
#         return self.kp * error + self.ki * self.int + self.kd * der

# speed_pid = PID(0.5, 0.01, 0.1)

# # -----------------------------
# # Metrics
# # -----------------------------
# fps = 0
# prev_time = time.time()

# total_detections = 0
# correct_predictions = 0

# csv_file = "carla_results.csv"
# if not os.path.exists(csv_file):
#     with open(csv_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Time", "Sign", "Confidence", "Decision", "Speed"])

# # -----------------------------
# # Memory + Timing
# # -----------------------------
# frame_count = 0
# sign_memory = deque(maxlen=5)

# sign_action_active = False
# sign_action_start_time = 0
# SIGN_ACTION_DURATION = 10
# current_active_sign = None

# # -----------------------------
# # Road Following
# # -----------------------------
# def follow_road():
#     waypoint = world.get_map().get_waypoint(
#         vehicle.get_location(),
#         project_to_road=True,
#         lane_type=carla.LaneType.Driving
#     )

#     if waypoint is None:
#         return 0

#     vehicle_yaw = vehicle.get_transform().rotation.yaw
#     road_yaw = waypoint.transform.rotation.yaw

#     error = (road_yaw - vehicle_yaw) / 90.0
#     return max(min(error, 0.5), -0.5)

# # -----------------------------
# # Front Vehicle Detection
# # -----------------------------
# def detect_front_vehicle():
#     ego = vehicle.get_transform()
#     ego_loc = ego.location
#     fwd = ego.get_forward_vector()

#     for npc in traffic_vehicles:
#         if not npc.is_alive:
#             continue

#         loc = npc.get_location()
#         dist = ego_loc.distance(loc)

#         if dist < 15:
#             vec = loc - ego_loc
#             dot = vec.x * fwd.x + vec.y * fwd.y

#             if dot > 0:
#                 return dist
#     return None

# # -----------------------------
# # Main Processing
# # -----------------------------
# def process_image(image):
#     global frame_count, fps, prev_time
#     global total_detections, correct_predictions
#     global sign_action_active, sign_action_start_time, current_active_sign

#     frame_count += 1
#     if frame_count % 2 != 0:
#         return

#     # FPS
#     current_time = time.time()
#     fps = 1 / (current_time - prev_time)
#     prev_time = current_time

#     img = np.frombuffer(image.raw_data, dtype=np.uint8)
#     img = img.reshape((image.height, image.width, 4))[:, :, :3]
#     frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     small = cv2.resize(frame, (416, 416))
#     results = yolo_model(small, conf=0.4, verbose=False)

#     best_sign = None
#     best_conf = 0

#     for r in results:
#         if r.boxes is None:
#             continue

#         for i, box in enumerate(r.boxes.xyxy):
#             if i > 2:
#                 break

#             x1, y1, x2, y2 = map(int, box)
#             crop = small[y1:y2, x1:x2]

#             if crop.size == 0:
#                 continue

#             cnn_img = cv2.resize(crop, (224, 224)) / 255.0
#             cnn_img = np.expand_dims(cnn_img, axis=0)

#             pred = cnn_model.predict(cnn_img, verbose=0)
#             class_id = int(np.argmax(pred))
#             conf = float(np.max(pred))

#             total_detections += 1
#             if conf > 0.8:
#                 correct_predictions += 1

#             if conf > best_conf:
#                 best_conf = conf
#                 best_sign = class_id

#             name = CLASS_NAMES.get(class_id, "Unknown")
#             label = f"{name} ({conf:.2f})"

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame, label, (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     sign_memory.append(best_sign)

#     stable_sign = None
#     if len(sign_memory) >= 3:
#         stable_sign = max(set(sign_memory), key=sign_memory.count)

#     front_dist = detect_front_vehicle()

#     # -----------------------------
#     # SAFE Decision Logic
#     # -----------------------------
#     target_speed = 35
#     decision = "DRIVE"

#     if sign_action_active:
#         if current_time - sign_action_start_time < SIGN_ACTION_DURATION:

#             if current_active_sign == 14:
#                 target_speed = 0
#                 decision = "STOP"

#             else:
#                 target_speed = 20
#                 decision = "SLOW"

#         else:
#             sign_action_active = False
#             current_active_sign = None

#     if not sign_action_active:
#         if stable_sign is not None:
#             sign_action_active = True
#             sign_action_start_time = current_time
#             current_active_sign = stable_sign

#         elif front_dist:
#             target_speed = min(20, front_dist * 1.2)
#             decision = "FOLLOW"

#     # Speed control
#     vel = vehicle.get_velocity()
#     speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
#     throttle = speed_pid.step((target_speed - speed) / 40.0)

#     control = carla.VehicleControl()
#     control.steer = follow_road()

#     if target_speed == 0 and speed > 2:
#         control.throttle = 0
#         control.brake = 1
#     else:
#         control.throttle = max(0, min(throttle, 0.75))
#         control.brake = 0

#     vehicle.apply_control(control)

#     # Accuracy
#     accuracy = (correct_predictions / max(total_detections, 1)) * 100

#     # CSV
#     with open(csv_file, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             round(current_time, 2),
#             CLASS_NAMES.get(stable_sign, "None"),
#             round(best_conf, 2),
#             decision,
#             round(speed, 2)
#         ])

#     # Display
#     cv2.putText(frame, decision, (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

#     cv2.putText(frame, f"FPS: {fps:.2f}", (20,80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#     cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (20,120),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#     cv2.imshow("FINAL OUTPUT", frame)
#     cv2.waitKey(5)

# # -----------------------------
# # Run
# # -----------------------------
# camera.listen(process_image)

# while True:
#     world.wait_for_tick()

#======================================================================================#

# import carla
# import random
# import cv2
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# from collections import deque
# import math
# import time
# import csv
# import os

# # =====================================================
# # FINAL COMPLETE SYSTEM (SPAWN + WEATHER + ACCURACY)
# # =====================================================

# # -----------------------------
# # Traffic Sign Names
# # -----------------------------
# CLASS_NAMES = {
#     0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50",
#     3: "Speed Limit 60", 4: "Speed Limit 70", 5: "Speed Limit 80",
#     6: "End Speed Limit 80", 7: "Speed Limit 100", 8: "Speed Limit 120",
#     9: "No Overtaking", 10: "No Overtaking Trucks", 11: "Right of Way",
#     12: "Priority Road", 13: "Yield", 14: "STOP",
#     15: "No Entry", 16: "Danger", 17: "Curve Left", 18: "Curve Right",
#     19: "Double Curve", 20: "Uneven Road", 21: "Slippery Road",
#     22: "Road Narrows", 23: "Road Work", 24: "Traffic Signals",
#     25: "Pedestrian Crossing", 26: "School Crossing",
#     27: "Bicycle Crossing", 28: "Snow", 29: "Animals",
#     30: "Speed Limit End", 31: "Turn Right", 32: "Turn Left",
#     33: "Go Straight", 34: "Go Straight or Right",
#     35: "Go Straight or Left", 36: "Keep Right", 37: "Keep Left",
#     38: "Roundabout", 39: "End No Overtaking",
#     40: "End No Overtaking Trucks"
# }

# # -----------------------------
# # Load Models
# # -----------------------------
# yolo_model = YOLO(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\yolov8n.pt")
# cnn_model = tf.keras.models.load_model(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\traffic_sign_final_model.h5")

# print("Models Loaded")

# # -----------------------------
# # CARLA Setup
# # -----------------------------
# client = carla.Client("localhost", 2000)
# client.set_timeout(10)

# world = client.load_world("Town03")
# blueprints = world.get_blueprint_library()
# spawn_points = world.get_map().get_spawn_points()
# traffic_manager = client.get_trafficmanager(8000)

# # -----------------------------
# # WEATHER CONDITIONS
# # -----------------------------
# weather_conditions = {
#     "Clear": carla.WeatherParameters.ClearNoon,
#     "Cloudy": carla.WeatherParameters.CloudyNoon,
#     "Rain": carla.WeatherParameters.HardRainNoon,
#     "Wet": carla.WeatherParameters.WetCloudyNoon,
#     "Sunset": carla.WeatherParameters.ClearSunset,
# }

# current_weather_name = random.choice(list(weather_conditions.keys()))
# world.set_weather(weather_conditions[current_weather_name])

# print("Weather:", current_weather_name)

# # -----------------------------
# # SMART SPAWN (NEAR SIGNS)
# # -----------------------------
# def get_best_spawn_point():
#     for sp in spawn_points:
#         waypoint = world.get_map().get_waypoint(sp.location)

#         # Prefer junctions
#         if waypoint.is_junction:
#             return sp

#         # Or near traffic lights
#         for tl in world.get_actors().filter('*traffic_light*'):
#             if sp.location.distance(tl.get_location()) < 30:
#                 return sp

#     return random.choice(spawn_points)

# vehicle = world.spawn_actor(
#     blueprints.filter("model3")[0],
#     get_best_spawn_point()
# )
# vehicle.set_autopilot(False)

# # -----------------------------
# # Traffic
# # -----------------------------
# traffic_vehicles = []
# for _ in range(20):
#     npc = world.try_spawn_actor(
#         random.choice(blueprints.filter("vehicle.*")),
#         random.choice(spawn_points)
#     )
#     if npc:
#         npc.set_autopilot(True, traffic_manager.get_port())
#         traffic_vehicles.append(npc)

# # -----------------------------
# # Camera
# # -----------------------------
# camera_bp = blueprints.find("sensor.camera.rgb")
# camera_bp.set_attribute("image_size_x", "640")
# camera_bp.set_attribute("image_size_y", "480")

# camera = world.spawn_actor(
#     camera_bp,
#     carla.Transform(carla.Location(x=1.5, z=2.4)),
#     attach_to=vehicle
# )

# # -----------------------------
# # Collision Recovery
# # -----------------------------
# def recover(event):
#     nearest = min(spawn_points, key=lambda sp: vehicle.get_location().distance(sp.location))
#     vehicle.set_transform(nearest)

# collision_sensor = world.spawn_actor(
#     blueprints.find("sensor.other.collision"),
#     carla.Transform(),
#     attach_to=vehicle
# )
# collision_sensor.listen(recover)

# # -----------------------------
# # PID Speed
# # -----------------------------
# class PID:
#     def __init__(self, kp, ki, kd):
#         self.kp, self.ki, self.kd = kp, ki, kd
#         self.prev = 0
#         self.int = 0

#     def step(self, error, dt=0.05):
#         self.int += error * dt
#         der = (error - self.prev) / dt
#         self.prev = error
#         return self.kp * error + self.ki * self.int + self.kd * der

# speed_pid = PID(0.5, 0.01, 0.1)

# # -----------------------------
# # Metrics + CSV
# # -----------------------------
# fps = 0
# prev_time = time.time()

# total_detections = 0
# correct_predictions = 0

# csv_file = "carla_results.csv"
# if not os.path.exists(csv_file):
#     with open(csv_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Time", "Weather", "Sign", "Confidence", "Decision", "Speed"])

# # -----------------------------
# # Memory
# # -----------------------------
# frame_count = 0
# sign_memory = deque(maxlen=5)

# sign_action_active = False
# sign_action_start_time = 0
# SIGN_ACTION_DURATION = 10
# current_active_sign = None

# # -----------------------------
# # Road Follow
# # -----------------------------
# def follow_road():
#     waypoint = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)
#     if waypoint is None:
#         return 0

#     vehicle_yaw = vehicle.get_transform().rotation.yaw
#     road_yaw = waypoint.transform.rotation.yaw

#     error = (road_yaw - vehicle_yaw) / 90.0
#     return max(min(error, 0.5), -0.5)

# # -----------------------------
# # Front Vehicle Detection
# # -----------------------------
# def detect_front_vehicle():
#     ego = vehicle.get_transform()
#     ego_loc = ego.location
#     fwd = ego.get_forward_vector()

#     for npc in traffic_vehicles:
#         if not npc.is_alive:
#             continue

#         loc = npc.get_location()
#         dist = ego_loc.distance(loc)

#         if dist < 15:
#             vec = loc - ego_loc
#             dot = vec.x * fwd.x + vec.y * fwd.y

#             if dot > 0:
#                 return dist
#     return None

# # -----------------------------
# # MAIN LOOP
# # -----------------------------
# def process_image(image):
#     global frame_count, fps, prev_time
#     global total_detections, correct_predictions
#     global sign_action_active, sign_action_start_time, current_active_sign

#     frame_count += 1
#     if frame_count % 2 != 0:
#         return

#     current_time = time.time()
#     fps = 1 / (current_time - prev_time)
#     prev_time = current_time

#     img = np.frombuffer(image.raw_data, dtype=np.uint8)
#     img = img.reshape((image.height, image.width, 4))[:, :, :3]
#     frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     small = cv2.resize(frame, (416, 416))
#     results = yolo_model(small, conf=0.4, verbose=False)

#     best_sign = None
#     best_conf = 0

#     for r in results:
#         if r.boxes is None:
#             continue

#         for i, box in enumerate(r.boxes.xyxy):
#             if i > 2:
#                 break

#             x1, y1, x2, y2 = map(int, box)
#             crop = small[y1:y2, x1:x2]

#             if crop.size == 0:
#                 continue

#             cnn_img = cv2.resize(crop, (224, 224)) / 255.0
#             cnn_img = np.expand_dims(cnn_img, axis=0)

#             pred = cnn_model.predict(cnn_img, verbose=0)
#             class_id = int(np.argmax(pred))
#             conf = float(np.max(pred))

#             total_detections += 1
#             if conf > 0.8:
#                 correct_predictions += 1

#             if conf > best_conf:
#                 best_conf = conf
#                 best_sign = class_id

#             name = CLASS_NAMES.get(class_id, "Unknown")
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     sign_memory.append(best_sign)

#     stable_sign = None
#     if len(sign_memory) >= 3:
#         stable_sign = max(set(sign_memory), key=sign_memory.count)

#     front_dist = detect_front_vehicle()

#     target_speed = 35
#     decision = "DRIVE"

#     if sign_action_active:
#         if current_time - sign_action_start_time < SIGN_ACTION_DURATION:
#             if current_active_sign == 14:
#                 target_speed = 0
#                 decision = "STOP"
#             else:
#                 target_speed = 20
#                 decision = "SLOW"
#         else:
#             sign_action_active = False
#             current_active_sign = None

#     if not sign_action_active:
#         if stable_sign is not None:
#             sign_action_active = True
#             sign_action_start_time = current_time
#             current_active_sign = stable_sign
#         elif front_dist:
#             target_speed = min(20, front_dist * 1.2)
#             decision = "FOLLOW"

#     # Control
#     vel = vehicle.get_velocity()
#     speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6
#     throttle = speed_pid.step((target_speed - speed) / 40.0)

#     control = carla.VehicleControl()
#     control.steer = follow_road()

#     if target_speed == 0 and speed > 2:
#         control.brake = 1
#         control.throttle = 0
#     else:
#         control.throttle = max(0, min(throttle, 0.75))
#         control.brake = 0

#     vehicle.apply_control(control)

#     accuracy = (correct_predictions / max(total_detections, 1)) * 100

#     # CSV
#     with open(csv_file, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([round(current_time,2), current_weather_name,
#                          CLASS_NAMES.get(stable_sign,"None"),
#                          round(best_conf,2), decision, round(speed,2)])

#     # Display
#     cv2.putText(frame, decision, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
#     cv2.putText(frame, f"FPS: {fps:.2f}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
#     cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#     cv2.imshow("FINAL OUTPUT", frame)
#     cv2.waitKey(5)

# # -----------------------------
# # Run
# # -----------------------------
# camera.listen(process_image)

# while True:
#     world.wait_for_tick()

#======================================================================================#

import carla
import random
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import math
import time
import os

# -----------------------------
# SCREENSHOT FOLDER
# -----------------------------
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
screenshot_count = 0

# -----------------------------
# FULL CLASS LABELS
# -----------------------------
CLASS_NAMES = {0:"Give Way",1:"No Entry",2:"One Way",3:"One Way",
4:"No Vehicle",5:"No Entry",6:"No Entry",7:"No Entry",8:"No Entry",
9:"No Entry",10:"No Entry",11:"Height Limit",12:"Weight Limit",
13:"Axle Weight",14:"STOP",15:"No Left Turn",16:"No Right Turn",
17:"No Overtaking",18:"Max Speed",19:"Max Speed",20:"Horn Prohibited",
21:"No Parking",22:"No Stopping",23:"Turn Left",24:"Turn Right",
25:"Steep Descent",26:"Steep Ascent",27:"Narrow Road",28:"Narrow Bridge",
29:"Unprotected",30:"Road Hump",31:"Dip",32:"Loose Gravel",
33:"Falling Rocks",34:"Cattle",35:"Crossroads",36:"Side Road Junction",
37:"Side Road",38:"Oblique Side",39:"Oblique Side",40:"T Junction",
41:"Y Junction",42:"Staggered Junction",43:"Staggered Junction",
44:"Roundabout",45:"Guarded Crossing",46:"Unguarded Crossing",
47:"Railway Crossing",48:"Railway Crossing",49:"Railway Crossing",
50:"Railway Crossing",51:"Parking",52:"Bus Stop",53:"First Aid",
54:"Telephone",55:"Fuel Station",56:"Hotel",57:"Restaurant",58:"Refreshment"}

# -----------------------------
# LOAD MODELS (UNCHANGED PATHS)
# -----------------------------
yolo_model = YOLO(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\yolov8n.pt")
cnn_model = tf.keras.models.load_model(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\traffic_sign_final_model.h5")

# -----------------------------
# CARLA SETUP
# -----------------------------
client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world("Town03")

blueprints = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# -----------------------------
# WEATHER
# -----------------------------
weather_conditions = {
    "Clear": carla.WeatherParameters.ClearNoon,
    "Rain": carla.WeatherParameters.HardRainNoon,
    "Cloudy": carla.WeatherParameters.CloudyNoon
}
current_weather_name = random.choice(list(weather_conditions.keys()))
world.set_weather(weather_conditions[current_weather_name])

# -----------------------------
# VEHICLE
# -----------------------------
vehicle = world.spawn_actor(
    blueprints.filter("model3")[0],
    random.choice(spawn_points)
)
vehicle.set_autopilot(False)

# -----------------------------
# CAMERA
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
# COLLISION HANDLING
# -----------------------------
def recover(event):
    print("⚠ Collision → Switching road")

    current_loc = vehicle.get_location()
    far_points = [sp for sp in spawn_points if sp.location.distance(current_loc) > 80]
    new_spawn = random.choice(far_points) if far_points else random.choice(spawn_points)

    vehicle.set_transform(new_spawn)

    control = carla.VehicleControl()
    control.throttle = 0
    control.brake = 1
    vehicle.apply_control(control)

collision_sensor = world.spawn_actor(
    blueprints.find("sensor.other.collision"),
    carla.Transform(),
    attach_to=vehicle
)
collision_sensor.listen(recover)

# -----------------------------
# PID CONTROLLER
# -----------------------------
class PID:
    def __init__(self):
        self.prev = 0
        self.int = 0

    def step(self, error, dt=0.05):
        self.int += error * dt
        der = (error - self.prev) / dt
        self.prev = error
        return 0.5*error + 0.01*self.int + 0.1*der

speed_pid = PID()

# -----------------------------
# ROAD FOLLOWING
# -----------------------------
def follow_road():
    waypoint = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)
    if waypoint is None:
        return 0
    return (waypoint.transform.rotation.yaw - vehicle.get_transform().rotation.yaw)/90

# -----------------------------
# MEMORY + METRICS
# -----------------------------
sign_memory = deque(maxlen=5)
fps, prev_time = 0, time.time()
total_det, correct_pred = 0, 0

sign_active = False
start_time = 0
SIGN_DURATION = 10
active_sign = None

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def process_image(image):
    global fps, prev_time, screenshot_count
    global total_det, correct_pred
    global sign_active, start_time, active_sign

    now = time.time()
    fps = 1/(now-prev_time)
    prev_time = now

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))[:, :, :3]
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    small = cv2.resize(frame,(416,416))
    results = yolo_model(small, conf=0.4, verbose=False)

    best_sign, best_conf = None, 0

    for r in results:
        if r.boxes is None: continue

        for box in r.boxes.xyxy:
            x1,y1,x2,y2 = map(int, box)
            crop = small[y1:y2, x1:x2]
            if crop.size==0: continue

            cnn_img = cv2.resize(crop,(224,224))/255.0
            cnn_img = np.expand_dims(cnn_img,axis=0)

            pred = cnn_model.predict(cnn_img,verbose=0)
            cid = int(np.argmax(pred))
            conf = float(np.max(pred))

            total_det +=1
            if conf>0.8: correct_pred+=1

            if conf>best_conf:
                best_conf, best_sign = conf, cid

            name = CLASS_NAMES.get(cid,"Unknown")
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{name} ({conf:.2f})",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # stable sign
    sign_memory.append(best_sign)
    stable = None
    if len(sign_memory)>=3:
        stable = max(set(sign_memory),key=sign_memory.count)

    # decision (10 sec)
    speed_target, decision = 35, "DRIVE"

    if sign_active:
        if now-start_time < SIGN_DURATION:
            if active_sign==14:
                speed_target, decision = 0,"STOP"
            else:
                speed_target, decision = 20,"SLOW"
        else:
            sign_active=False

    if not sign_active and stable is not None:
        sign_active=True
        start_time=now
        active_sign=stable

    # control
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2+vel.y**2+vel.z**2)*3.6
    throttle = speed_pid.step((speed_target-speed)/40.0)

    control = carla.VehicleControl()
    control.steer = follow_road()

    if speed_target==0:
        control.brake=1
    else:
        control.throttle=max(0,min(throttle,0.75))

    vehicle.apply_control(control)

    acc = (correct_pred/max(total_det,1))*100

    # DISPLAY (RESULT SECTION)
    cv2.putText(frame,f"Decision: {decision}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.putText(frame,f"FPS: {fps:.2f}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(frame,f"Accuracy: {acc:.2f}%",(20,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(frame,f"Speed: {speed:.2f}",(20,160),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    cv2.putText(frame,f"Weather: {current_weather_name}",(20,200),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    if best_sign is not None:
        cv2.putText(frame,f"Detected: {CLASS_NAMES.get(best_sign)}",(20,240),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    # screenshot
    if (best_conf>0.9 or decision!="DRIVE") and screenshot_count%20==0:
        cv2.imwrite(f"{SCREENSHOT_DIR}/shot_{int(time.time())}.png",frame)

    screenshot_count+=1

    cv2.imshow("FINAL OUTPUT",frame)
    cv2.waitKey(5)

# -----------------------------
# RUN
# -----------------------------
camera.listen(process_image)

while True:
    world.wait_for_tick()