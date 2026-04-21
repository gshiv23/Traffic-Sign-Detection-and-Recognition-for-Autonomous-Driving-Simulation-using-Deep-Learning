import carla

def apply_decision(vehicle, class_id):

    STOP_SIGNS = [1]
    SPEED_SIGNS = [18,19]
    WARNING_SIGNS = [33]

    if class_id in STOP_SIGNS:

        print("Stop sign detected")

        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0)
        )

    elif class_id in SPEED_SIGNS:

        print("Speed limit detected")

        vehicle.apply_control(
            carla.VehicleControl(throttle=0.4)
        )

    elif class_id in WARNING_SIGNS:

        print("Warning sign")

        vehicle.apply_control(
            carla.VehicleControl(throttle=0.2)
        )