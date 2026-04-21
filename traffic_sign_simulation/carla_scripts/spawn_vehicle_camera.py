import carla

def spawn_vehicle_with_camera(world):

    blueprint = world.get_blueprint_library()

    vehicle_bp = blueprint.filter("model3")[0]

    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint.find("sensor.camera.rgb")

    camera_bp.set_attribute("image_size_x","640")
    camera_bp.set_attribute("image_size_y","480")

    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.4)
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )

    return vehicle, camera