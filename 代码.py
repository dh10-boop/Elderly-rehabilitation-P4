import csv
import math
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp


# __file__：Python自动提供的当前代码文件路径。
# OUTPUT_DIR：当前代码.py所在的文件夹路径。
OUTPUT_DIR = Path(__file__).parent
# CSV_PATH：标准动作角度数据要保存到的CSV文件路径。
CSV_PATH = OUTPUT_DIR / "standard_template.csv"
# PNG_PATH：角度曲线图要保存到的PNG文件路径。
PNG_PATH = OUTPUT_DIR / "standard_template.png"


def calculate_angle(a, b, c):
    """计算以 b 为顶点的二维夹角，返回 0～180 度。"""
    # a、b、c：三个关键点的像素坐标，其中b是关节角的顶点。
    # direction1：从顶点b指向点a的方向，单位是弧度。
    direction1 = math.atan2(a[1] - b[1], a[0] - b[0])
    # direction2：从顶点b指向点c的方向，单位是弧度。
    direction2 = math.atan2(c[1] - b[1], c[0] - b[0])

    # angle：两个方向之差转换成的角度，先取绝对值。
    angle = abs(math.degrees(direction2 - direction1))

    if angle > 180:
        angle = 360 - angle

    return angle


def get_point(landmark_list, point_id, width, height):
    """按关键点名称从列表取点，再转换成画面像素坐标。"""
    # landmark_list：MediaPipe识别出的33个关键点组成的列表。
    # point_id：要获取的身体关键点名称，例如P.LEFT_ELBOW。
    # width、height：当前摄像头画面的像素宽度和高度。
    # index：point_id对应的数字编号，例如LEFT_ELBOW.value是13。
    index = point_id.value
    # landmark：从33个关键点中取出的指定关键点，保存x、y、z和visibility。
    landmark = landmark_list[index]

    # x、y：把0～1的归一化坐标转换成当前画面中的像素坐标。
    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return (x, y)


def save_template(recorded_data, joint_names):
    """把录制结果保存为 CSV 表格。"""
    # recorded_data：录制期间保存的全部数据，每个子列表代表一帧。
    # joint_names：8个关节角度的名称，用于生成CSV表头。
    if not recorded_data:
        print("没有可保存的数据")
        return

    # file：根据CSV_PATH打开的文件对象，with结束时会自动关闭。
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        # writer：按照CSV格式向file写入数据的CSV写入器对象。
        writer = csv.writer(file)
        writer.writerow(["Time"] + joint_names)
        writer.writerows(recorded_data)

    print(f"标准模板已保存：{CSV_PATH}")


def plot_template(recorded_data, joint_names):
    """把8个关节角度画成4组时间曲线，并保存为 PNG。"""
    # recorded_data：保存了每一帧时间和8个关节角度的二维列表。
    # joint_names：关节名称列表，用来查找每个角度位于row中的哪一列。
    if not recorded_data:
        print("没有可绘制的数据")
        return

    # times：取出每一行的第0项，组成所有采样时刻的列表；row代表其中一帧的数据。
    times = [row[0] for row in recorded_data]

    # plot_groups：4张子图的画图配置；每项依次是标题、左侧名称、右侧名称。
    plot_groups = [
        ("Shoulder", "LeftShoulder", "RightShoulder"),
        ("Elbow", "LeftElbow", "RightElbow"),
        ("Hip", "LeftHip", "RightHip"),
        ("Knee", "LeftKnee", "RightKnee"),
    ]

    # figure：整张大画布；axes：装着4个子图对象的2×2数组。
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    # axis：当前子图对象；title：子图标题；left_name、right_name：左右关节名称。
    for axis, (title, left_name, right_name) in zip(axes.flat, plot_groups):
        # left_index、right_index：左右关节角度在每个row中的列号。
        # 加1是因为row[0]保存时间，而joint_names中没有Time。
        left_index = joint_names.index(left_name) + 1
        right_index = joint_names.index(right_name) + 1

        # left_angles、right_angles：从全部录制帧中取出的左右角度序列。
        left_angles = [row[left_index] for row in recorded_data]
        right_angles = [row[right_index] for row in recorded_data]

        axis.plot(times, left_angles, label=left_name)
        axis.plot(times, right_angles, label=right_name)
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Angle (degree)")
        axis.set_ylim(0, 180)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("Standard Rehabilitation Movement Template")
    figure.tight_layout()
    figure.savefig(PNG_PATH, dpi=150)
    plt.close(figure)

    print(f"角度曲线已保存：{PNG_PATH}")


def main():
    # mp_pose：MediaPipe的人体姿态识别模块。
    mp_pose = mp.solutions.pose
    # mp_drawing：MediaPipe提供的关键点和骨架绘制工具。
    mp_drawing = mp.solutions.drawing_utils
    # P：PoseLandmark枚举类的简写，用于用名称表示33个关键点。
    P = mp_pose.PoseLandmark

    # pose：人体姿态识别对象，后面用pose.process()处理每一帧。
    pose = mp_pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5,
    )

    # cap：连接0号摄像头的视频采集对象；它不是摄像头画面。
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        pose.close()
        raise SystemExit("摄像头打开失败")

    # joint_list：要计算的8个关节角度定义。
    # 每项依次保存：显示名称、点a、作为角顶点的点b、点c。
    joint_list = [
        ("LeftShoulder", P.LEFT_ELBOW, P.LEFT_SHOULDER, P.LEFT_HIP),
        ("RightShoulder", P.RIGHT_ELBOW, P.RIGHT_SHOULDER, P.RIGHT_HIP),
        ("LeftElbow", P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST),
        ("RightElbow", P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST),
        ("LeftHip", P.LEFT_SHOULDER, P.LEFT_HIP, P.LEFT_KNEE),
        ("RightHip", P.RIGHT_SHOULDER, P.RIGHT_HIP, P.RIGHT_KNEE),
        ("LeftKnee", P.LEFT_HIP, P.LEFT_KNEE, P.LEFT_ANKLE),
        ("RightKnee", P.RIGHT_HIP, P.RIGHT_KNEE, P.RIGHT_ANKLE),
    ]

    # joint_names：只取出8个关节名称，用于CSV表头和画图时查找列号。
    # item：遍历joint_list时临时代表其中一项，item[0]就是该关节的显示名称。
    joint_names = [item[0] for item in joint_list]

    # recording：录制状态；True表示正在记录每一帧的角度数据。
    recording = False
    # recorded_data：全部已录制数据；每项row代表一帧的时间和8个角度。
    recorded_data = []
    # start_time：按下r开始录制时的时间戳，用来计算相对经过时间。
    start_time = 0.0

    while True:
        # ret：本次是否成功读到画面；frame：本次读到的一帧BGR图像。
        ret, frame = cap.read()

        if not ret:
            print("读取画面失败")
            break

        # rgb_frame：把frame转换成MediaPipe需要的RGB图像。
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results：MediaPipe对当前帧的识别结果。
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

            # landmark_list：当前帧识别出的33个人体关键点列表。
            landmark_list = results.pose_landmarks.landmark
            # height、width：画面像素高和宽；_接收本节用不到的颜色通道数。
            height, width, _ = frame.shape
            # angle_values：当前这一帧计算出的8个关节角度。
            angle_values = []

            # name：关节名称；a_id、b_id、c_id：计算该关节角的三个关键点名称。
            for name, a_id, b_id, c_id in joint_list:
                # a、b、c：三个关键点转换后的像素坐标，其中b是角的顶点。
                a = get_point(landmark_list, a_id, width, height)
                b = get_point(landmark_list, b_id, width, height)
                c = get_point(landmark_list, c_id, width, height)

                # angle：当前关节在当前帧中的角度，范围是0～180度。
                angle = calculate_angle(a, b, c)
                angle_values.append(angle)

                cv2.putText(
                    frame,
                    f"{name}:{angle:.0f}",
                    b,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            if recording:
                # elapsed_time：从按下r开始录制到当前帧所经过的秒数。
                elapsed_time = time.time() - start_time
                # row：准备保存的一行数据，第0项是时间，后面是8个角度。
                row = [round(elapsed_time, 3)]
                # value：遍历angle_values时临时代表其中一个关节角度。
                row += [round(value, 2) for value in angle_values]
                recorded_data.append(row)

        if recording:
            # status_text：显示在画面上的当前状态和已录制帧数。
            status_text = f"RECORDING  Frames:{len(recorded_data)}"
            # status_color：状态文字的BGR颜色，录制时为红色。
            status_color = (0, 0, 255)
        else:
            status_text = "READY  R:Start  Q:Quit"
            # 未录制时使用绿色状态文字。
            status_color = (0, 255, 0)

        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        cv2.imshow("Standard Template Recording", frame)

        # key：waitKey返回的整数按键编码，用来判断r、s或q。
        key = cv2.waitKey(1)

        if key == ord("r") and not recording:
            recorded_data.clear()
            start_time = time.time()
            recording = True
            print("开始录制标准动作")

        elif key == ord("s") and recording:
            recording = False
            save_template(recorded_data, joint_names)
            plot_template(recorded_data, joint_names)
            print("录制完成，可以按 r 重新录制，或按 q 退出")

        elif key == ord("q"):
            if recording:
                recording = False
                save_template(recorded_data, joint_names)
                plot_template(recorded_data, joint_names)
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
