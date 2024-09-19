import glob
import os
import sys
import random
import carla
import time
import pygame
import logging
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from collections import deque
import math
import pandas as pd

# 添加CARLA Python API路径
try:
    sys.path.append(glob.glob('D:/CODE/carla/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# 添加agents模块路径
sys.path.append('D:/CODE/carla/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla')

# 从agents模块导入BehaviorAgent
from agents.navigation.behavior_agent import BehaviorAgent

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_camera(world, ego_vehicle):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_transform = carla.Transform(carla.Location(x=-8, y=0, z=4),
                                       carla.Rotation(pitch=10, yaw=0, roll=0))
    camera = world.spawn_actor(camera_bp, camera_transform,
                               attach_to=ego_vehicle, attachment_type=carla.AttachmentType.SpringArm)
    return camera

def setup_lidar(world, ego_vehicle):
    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '56000')
    lidar_transform = carla.Transform(carla.Location(x=0, z=3.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    return lidar

def process_img(image, display):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

def check_spawn_collision(world, spawn_point, clearance_distance=5.0):
    actors = world.get_actors()
    for actor in actors:
        if actor.get_location().distance(spawn_point.location) < clearance_distance:
            return True  # 表示该生成点附近有车辆或障碍物
    return False  # 该生成点安全

def generate_spawn_points_from_road(world, waypoints, road_width, max_points=500, min_distance=10.0):
    spawn_points = []
    count = 0
    last_point = None
    for wp in waypoints:
        if count >= max_points:
            break
        spawn_point = wp.transform
        right_vector = spawn_point.get_right_vector()
        offset_distance = road_width / 4.0  # 偏移距离为道路宽度的1/4
        spawn_point.location += carla.Location(x=right_vector.x * offset_distance, y=right_vector.y * offset_distance)
        
        # 检查新的生成点是否与其他对象或生成点冲突
        if last_point is None or spawn_point.location.distance(last_point.location) >= min_distance:
            if not check_spawn_collision(world, spawn_point, clearance_distance=5.0):  # 增加了clearance_distance以确保安全生成
                spawn_points.append(spawn_point)
                count += 1
                last_point = spawn_point
            
    return spawn_points

def setup_vehicles(world, client, spawn_points, reserved_points):
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.audi.*')
    number_of_vehicles = 0
    batch = []
    reserved_points_locations = [point.location for point in reserved_points]
    traffic_manager = client.get_trafficmanager()
    selected_spawn_points = random.sample(spawn_points, min(len(spawn_points), number_of_vehicles))  # 随机选择生成点
    for spawn_point in selected_spawn_points:
        if spawn_point.location in reserved_points_locations:
            continue
        
        if check_spawn_collision(world, spawn_point):
            continue  # 如果检测到碰撞，则跳过该生成点

        vehicle_bp = random.choice(vehicle_blueprints)
        batch.append(carla.command.SpawnActor(vehicle_bp, spawn_point))

    other_vehicle_list = []
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            logging.error(response.error)
        else:
            vehicle = world.get_actor(response.actor_id)
            other_vehicle_list.append(vehicle)
            vehicle.set_autopilot(True)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 80) 
    logging.info(f"Total vehicles generated: {len(other_vehicle_list)}")
    return other_vehicle_list

def monitor_obstacles_and_lane_change(vehicle, world, traffic_manager):
    while True:
        if check_lane_change_needed(vehicle, world):
            print("Obstacle detected, attempting to change lane...")
            time.sleep(0.5)  # Give some time to reduce speed
            while check_lane_change_needed(vehicle, world):
                if check_safe_to_change_lane(vehicle, world, True):
                    print("Safe to change to left lane. Changing lane...")
                    traffic_manager.force_lane_change(vehicle, True)
                    time.sleep(0.1)  # Give some time to complete lane change
                    traffic_manager.force_lane_change(vehicle, False)
                    break
                else:
                    print("No safe lane to change to.")
                    traffic_manager.force_lane_change(vehicle, False)
            time.sleep(0.5)  # Cooldown period to avoid immediate re-checking

def check_lane_change_needed(vehicle, world):
    vehicle_list = world.get_actors().filter('*vehicle*')
    vehicle_location = vehicle.get_location()
    vehicle_transform = vehicle.get_transform()
    vehicle_waypoint = world.get_map().get_waypoint(vehicle_location)

    for other_vehicle in vehicle_list:
        if other_vehicle.id != vehicle.id:
            other_vehicle_location = other_vehicle.get_location()
            other_vehicle_waypoint = world.get_map().get_waypoint(other_vehicle_location)
            distance = vehicle_location.distance(other_vehicle_location)
            if distance < 30.0 and vehicle_waypoint.lane_id == other_vehicle_waypoint.lane_id:
                return True
    return False

def check_safe_to_change_lane(vehicle, world, direction):
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_waypoint = world.get_map().get_waypoint(vehicle_location)
    
    if direction:
        target_waypoint = vehicle_waypoint.get_left_lane()
    else:
        target_waypoint = vehicle_waypoint.get_right_lane()

    if target_waypoint is None:
        return False

    for i in range(1, 10):  # 检查未来20米范围内的换道路径
        next_wp = target_waypoint.next(2.0 * i)[0]
        if next_wp:
            for other_vehicle in world.get_actors().filter('*vehicle*'):
                if other_vehicle.id != vehicle.id:
                    other_vehicle_location = other_vehicle.get_location()
                    other_vehicle_waypoint = world.get_map().get_waypoint(other_vehicle_location)
                    
                    # 检查其他车辆是否在目标车道上，并且是否在换道车辆的前方
                    if other_vehicle_waypoint.road_id == target_waypoint.road_id and other_vehicle_waypoint.lane_id == target_waypoint.lane_id:
                        distance = next_wp.transform.location.distance(other_vehicle_location)
                        relative_position = next_wp.transform.location - vehicle_transform.location
                        if distance < 10.0 and relative_position.x > 0:  # 确保车辆在前方20米以内
                            return False
    return True

def setup_convoy(world, client, ego_vehicle_bp, spawn_points, reserved_points, max_attempts=100000):
    ego_vehicle = None
    for attempt in range(max_attempts):
        try:
            ego_spawn_point = random.choice(reserved_points)
            ego_spawn_point.location.z += 1.0  # 抬高主车辆的 z 坐标
            ego_vehicle = world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)
            if ego_vehicle is not None:
                break
        except RuntimeError as e:
            logging.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
            time.sleep(1)  # 添加一些延迟再重试
    
    if ego_vehicle is None:
        raise RuntimeError(f"Failed to spawn ego vehicle after {max_attempts} attempts")

    traffic_manager = client.get_trafficmanager()
    ego_vehicle.set_autopilot(True, traffic_manager.get_port())
    spawn_points.remove(ego_spawn_point)
    reserved_points.remove(ego_spawn_point)

    follow_vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    follow_vehicle_bp.set_attribute('color', '0, 0, 200')
    follow_vehicle_bp.set_attribute('role_name', 'follow_car')

    follow_vehicle2_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    follow_vehicle2_bp.set_attribute('color', '0, 0, 200')
    follow_vehicle2_bp.set_attribute('role_name', 'follow_car2')

    def try_spawn_follow_vehicle(world, blueprint, ego_transform, distance=3):
        for i in range(10):
            follow_vehicle_spawn_point = carla.Transform(
                carla.Location(
                    x=ego_transform.location.x - (distance + i) * np.cos(np.radians(ego_transform.rotation.yaw)),
                    y=ego_transform.location.y - (distance + i) * np.sin(np.radians(ego_transform.rotation.yaw)),
                    z=ego_transform.location.z + 1.0  # 抬高跟随车辆的 z 坐标
                ),
                ego_transform.rotation
            )
            follow_vehicle = world.try_spawn_actor(blueprint, follow_vehicle_spawn_point)
            if follow_vehicle:
                return follow_vehicle
        return None

    follow_vehicle = try_spawn_follow_vehicle(world, follow_vehicle_bp, ego_spawn_point)
    if follow_vehicle is None:
        raise RuntimeError("无法在任何位置生成跟随车辆")

    follow_vehicle.set_autopilot(False)

    follow_vehicle2 = try_spawn_follow_vehicle(world, follow_vehicle2_bp, ego_spawn_point, distance=6)
    if follow_vehicle2 is None:
        raise RuntimeError("无法在任何位置生成第二辆跟随车辆")

    follow_vehicle2.set_autopilot(False)

    return ego_vehicle, follow_vehicle, follow_vehicle2


def distance_vehicle(transform1, transform2):
    loc1 = transform1.location
    loc2 = transform2.location
    return np.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2 + (loc1.z - loc2.z) ** 2)

def draw_path(world, path, color=carla.Color(0, 255, 0)):
    if not path:
        logging.warning("No path found, cannot draw path.")
        return

    for i in range(len(path) - 1):
        world.debug.draw_line(
            path[i] + carla.Location(z=0.5),
            path[i + 1] + carla.Location(z=0.5),
            thickness=0.1,
            color=color,
            life_time=10.0
        )

def draw_section_boundaries(world, waypoints):
    road_length = sum([wp.transform.location.distance(waypoints[i+1].transform.location) for i, wp in enumerate(waypoints[:-1])])
    section_length = road_length / 6

    current_length = 0
    section_points = []
    
    for i, wp in enumerate(waypoints[:-1]):
        current_length += wp.transform.location.distance(waypoints[i+1].transform.location)
        if current_length >= section_length:
            current_length = 0
            section_points.append(wp.transform.location)
            # Get right vector (perpendicular to the road)
            right_vector = wp.transform.get_right_vector()
            start_point = wp.transform.location + right_vector * 10  # Adjust the offset as needed to cover the road
            end_point = wp.transform.location - right_vector * 10
            world.debug.draw_line(start_point, end_point, thickness=0.2, color=carla.Color(255, 0, 0), life_time=10.0)

def compute_vehicle_metrics(world, ego_vehicle, follow_vehicle, follow_vehicle2, pid_controller, pid_controller2, target_speed1, target_speed2, historical_data):
    """
    统计每个仿真步长三辆车的轨迹、速度、航向角和PID期望速度。

    :param world: CARLA 世界对象
    :param ego_vehicle: 主车辆
    :param follow_vehicle: 跟随车辆1
    :param follow_vehicle2: 跟随车辆2
    :param pid_controller: 跟随车辆1的PID控制器
    :param pid_controller2: 跟随车辆2的PID控制器
    :param target_speed1: 跟随车辆1的目标速度
    :param target_speed2: 跟随车辆2的目标速度
    :param historical_data: 用于存储历史数据的字典
    """
    # 车辆列表
    vehicles = [ego_vehicle, follow_vehicle, follow_vehicle2]
    pids = [pid_controller, pid_controller2, None]  # 主车辆没有PID控制器
    target_speeds = [None, target_speed1, target_speed2]  # Add target speeds for the follow vehicles

    # 初始化当前步长的数据存储
    current_step_data = {
        "trajectories": [],
        "speeds": [],
        "angles": [],
        "target_speeds": []  # Store target speeds here
    }

    for i, vehicle in enumerate(vehicles):
        # 获取车辆的当前变换信息
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        angle = transform.rotation.yaw

        # 记录数据
        current_step_data["trajectories"].append((transform.location.x, transform.location.y, transform.location.z))
        current_step_data["speeds"].append(speed)
        current_step_data["angles"].append(angle)
        current_step_data["target_speeds"].append(target_speeds[i])  # Store the target speed

    # 将当前步长的数据添加到历史数据中
    historical_data["trajectories"].append(current_step_data["trajectories"])
    historical_data["speeds"].append(current_step_data["speeds"])
    historical_data["angles"].append(current_step_data["angles"])
    historical_data["target_speeds"].append(current_step_data["target_speeds"])  # Save target speeds

def save_vehicle_metrics_to_excel(historical_data, filename="vehicle_metricslast.xlsx"):
    """
    将车辆的轨迹、速度、航向角和PID期望速度数据保存到Excel文件中。

    :param historical_data: 用于存储历史数据的字典
    :param filename: 保存的文件名
    """
    # 创建DataFrame
    df_trajectories = pd.DataFrame(historical_data["trajectories"], columns=['Ego Vehicle Trajectory', 'Follow Vehicle 1 Trajectory', 'Follow Vehicle 2 Trajectory'])
    df_speeds = pd.DataFrame(historical_data["speeds"], columns=['Ego Vehicle Speed', 'Follow Vehicle 1 Speed', 'Follow Vehicle 2 Speed'])
    df_angles = pd.DataFrame(historical_data["angles"], columns=['Ego Vehicle Yaw', 'Follow Vehicle 1 Yaw', 'Follow Vehicle 2 Yaw'])
    df_target_speeds = pd.DataFrame(historical_data["target_speeds"], columns=['Ego Vehicle PID Speed', 'Follow Vehicle 1 PID Speed', 'Follow Vehicle 2 PID Speed'])
    
    # 合并数据
    df = pd.concat([df_trajectories, df_speeds, df_angles, df_target_speeds], axis=1)
    
    # 保存到Excel
    df.to_excel(filename, index=False)
    print(f"Vehicle metrics saved to {filename}")

class VehiclePIDController:
    def __init__(self, vehicle, args_lateral, args_longitudinal):
        self.vehicle = vehicle
        self.args_lateral = args_lateral
        self.args_longitudinal = args_longitudinal
        self.max_throttle = 1.0
        self.max_brake = 1.0
        self.max_steering = 0.8  # 将最大方向盘角度减小
        self.last_steer = 0.0  # 记录上一次的转向值

    def run_step(self, target_speed, target_transform):
        control = carla.VehicleControl()
        current_transform = self.vehicle.get_transform()
        current_velocity = self.vehicle.get_velocity()

        # 计算误差
        error_x = target_transform.location.x - current_transform.location.x
        error_y = target_transform.location.y - current_transform.location.y

        # 计算目标角度
        target_yaw = np.arctan2(error_y, error_x)
        current_yaw = np.radians(current_transform.rotation.yaw)
        yaw_error = target_yaw - current_yaw

        # 在环形道路上处理角度误差，确保角度在[-pi, pi]之间
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # 限制方向变化速率
        steer_change = np.clip(yaw_error - self.last_steer, -0.05, 0.05)  # 减小方向变化速率
        control.steer = np.clip(self.last_steer + steer_change, -self.max_steering, self.max_steering)
        self.last_steer = control.steer

        # 计算速度误差
        current_speed = np.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2)
        speed_error = target_speed - current_speed

        # 调整速度
        if speed_error > 0:
            control.throttle = np.clip(speed_error, 0.0, self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-speed_error, 0.0, self.max_brake)

        logging.debug(f"Steer: {control.steer}, Throttle: {control.throttle}, Brake: {control.brake}, Target Speed: {target_speed}")

        return control
def compute_idm_speed(lead_vehicle, following_vehicle, desired_speed=15.0, min_distance=2.0, time_headway=1.5, max_acceleration=1.0, comfortable_deceleration=1.5):
    """
    根据 IDM 模型计算跟随车辆的目标速度。
    
    :param lead_vehicle: 领航车辆
    :param following_vehicle: 跟随车辆
    :param desired_speed: 跟随车辆的期望速度
    :param min_distance: 最小安全距离
    :param time_headway: 期望时间间隔
    :param max_acceleration: 最大加速度
    :param comfortable_deceleration: 舒适减速度
    :return: 跟随车辆的目标速度
    """
    # 获取两辆车的速度
    lead_speed = np.sqrt(lead_vehicle.get_velocity().x**2 + lead_vehicle.get_velocity().y**2 + lead_vehicle.get_velocity().z**2)
    following_speed = np.sqrt(following_vehicle.get_velocity().x**2 + following_vehicle.get_velocity().y**2 + following_vehicle.get_velocity().z**2)
    
    # 获取两辆车的位置
    lead_transform = lead_vehicle.get_transform()
    following_transform = following_vehicle.get_transform()
    
    # 计算两辆车的距离
    distance = np.sqrt((lead_transform.location.x - following_transform.location.x) ** 2 + 
                       (lead_transform.location.y - following_transform.location.y) ** 2 +
                       (lead_transform.location.z - following_transform.location.z) ** 2)
    
    # 计算期望车距
    delta_v = following_speed - lead_speed
    s_star = min_distance + max(0, following_speed * time_headway + (following_speed * delta_v) / (2 * np.sqrt(max_acceleration * comfortable_deceleration)))
    
    # 计算加速度
    acceleration = max_acceleration * (1 - (following_speed / desired_speed) ** 4 - (s_star / distance) ** 2)
    
    # 计算目标速度
    target_speed = following_speed + acceleration
    
    return max(0, min(target_speed, desired_speed))  # 目标速度应在0和期望速度之间


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

    # 加载OpenDrive地图
    xodr_path = str("D:/CODE/carla/CARLA_0.9.15/WindowsNoEditor/PythonAPI/util/opendrive/mycircle03.xodr")
    
    with open(xodr_path, encoding='utf-8') as od_file:
        data = od_file.read()
        vertex_distance = 1.0  # 单位：米
        wall_height = 0.5      # 单位：米
        extra_width = 3.0      # 单位：米
        world = client.generate_opendrive_world(
            data, carla.OpendriveGenerationParameters(
                vertex_distance=vertex_distance,
                wall_height=wall_height,
                additional_width=extra_width,
                smooth_junctions=True,
                enable_mesh_visibility=True))

    # 设置天气
    weather = carla.WeatherParameters(
        cloudiness=20.0,
        precipitation=0.0,
        fog_density=0.0,
        sun_altitude_angle=70.0
    )
    world.set_weather(weather)

    ego_vehicle_bp = world.get_blueprint_library().find('vehicle.mercedes.coupe')
    ego_vehicle_bp.set_attribute('color', '0, 0, 200')
    ego_vehicle_bp.set_attribute('role_name', 'my_car')

    spawn_points = world.get_map().get_spawn_points()
    logging.info(f"Total available spawn points: {len(spawn_points)}")

    waypoints = world.get_map().generate_waypoints(2.0)
    road_width = 5.0
    additional_spawn_points = generate_spawn_points_from_road(world, waypoints, road_width, max_points=200)
    spawn_points.extend(additional_spawn_points)
    logging.info(f"Total spawn points after addition: {len(spawn_points)}")

    reserved_points = random.sample(spawn_points, 3)
    ego_vehicle, follow_vehicle, follow_vehicle2 = setup_convoy(world, client, ego_vehicle_bp, spawn_points, reserved_points)
    other_vehicle_list = setup_vehicles(world, client, spawn_points, reserved_points)
    
    camera = setup_camera(world, ego_vehicle)
    lidar = setup_lidar(world, ego_vehicle)

    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    spectator = world.get_spectator()

    camera.listen(lambda image: process_img(image, display))
    trajectory2 = deque(maxlen=5)
    trajectory3 = deque(maxlen=5)
    trajectory = []
    speeds = []
    angles = []

    monitor_thread = Thread(target=monitor_obstacles_and_lane_change, args=(ego_vehicle, world, client.get_trafficmanager()))
    monitor_thread.start()

    args_lateral = {'K_P': 1.2, 'K_I': 0.2, 'K_D': 0.4}
    args_longitudinal = {'K_P': 1, 'K_I': 0.1, 'K_D': 0.1}
    pid_controller = VehiclePIDController(follow_vehicle, args_lateral, args_longitudinal)
    pid_controller2 = VehiclePIDController(follow_vehicle2, args_lateral, args_longitudinal)

    plt.ion()
    plt.show()

    step_count = 0
    historical_data = {
        "trajectories": [],
        "speeds": [],
        "angles": [],
        "target_speeds": []  # Initialize target_speeds key here
    }


    try:
        while step_count < 5000:  # 假设仿真持续5000步
            world.tick()
            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=100, x=-50, y=-50),
                                        carla.Rotation(pitch=-70, yaw=45, roll=0)))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            velocity = ego_vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            location = transform.location
            rotation = transform.rotation

            speeds.append(speed)
            angles.append(rotation.yaw)
            trajectory.append((location.x, location.y, location.z))
            trajectory2.append(location)
            draw_path(world, list(trajectory2))
        
# Inside your main loop, after computing target speeds for the follow vehicles

# For vehicle 1
            target_speed1 = compute_idm_speed(ego_vehicle, follow_vehicle)

# For vehicle 2
            target_speed2 = compute_idm_speed(follow_vehicle, follow_vehicle2)

# Update metrics
            compute_vehicle_metrics(world, ego_vehicle, follow_vehicle, follow_vehicle2, pid_controller, pid_controller2, target_speed1, target_speed2, historical_data)

            # 使用 IDM 模型计算跟随车辆的目标速度
            if len(trajectory) > 1:
                lead_vehicle = ego_vehicle
                target_speed = compute_idm_speed(lead_vehicle, follow_vehicle)
                target_waypoint = carla.Transform(carla.Location(x=trajectory[-1][0], y=trajectory[-1][1], z=trajectory[-1][2]))
                control = pid_controller.run_step(target_speed, target_waypoint)
                follow_vehicle.apply_control(control)

            # 使用 IDM 模型计算第二辆跟随车辆的目标速度
            trajectory3.append(follow_vehicle.get_location())
            draw_path(world, list(trajectory3), color=carla.Color(255, 0, 0))
            if len(trajectory3) > 1:
                lead_vehicle = follow_vehicle
                last_location = trajectory3[-1]  # 这是 carla.Location 对象
                target_speed2 = compute_idm_speed(follow_vehicle, follow_vehicle2)
                target_waypoint2 = carla.Transform(carla.Location(x=last_location.x, y=last_location.y, z=last_location.z))
                control2 = pid_controller2.run_step(target_speed2, target_waypoint2)
                follow_vehicle2.apply_control(control2)

            step_count += 1
            pygame.display.flip()

    finally:
        # 仿真结束时保存数据到Excel文件
        save_vehicle_metrics_to_excel(historical_data)

        # 销毁车辆和传感器
        for vehicle in world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        for controller in world.get_actors().filter('*controller*'):
            controller.stop()
        if camera:
            camera.destroy()
        if lidar:
            lidar.stop()
            lidar.destroy()        
        pygame.quit()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('结束仿真')
