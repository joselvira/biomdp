"""
Created on Thu Jul 04 09:35:20 2024

@author: Jose L. L. Elvira

INSTALL:
# - pip install OpenCV-python
- conda install -c conda-forge opencv

- pip install mediapipe

More info:
https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md

"""

# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "v.1.1.4"
__date__ = "27/03/2025"


"""
Updates:
    27/03/2025, v1.1.4
        - save_frame_file supports Path objects.
        - Error message if could not save processed image.

    09/03/2025, v1.1.3
        - Adapted to biomdp with translations.

    04/03/2025, v1.1.2
        - Incluida opción de escapar de los vídeos con tecla Esc.

    01/03/2025, v1.1.1
        - Creada función para añadir marcadores centro_caderas y centro_hombros.
        - En discreto, independiente show markers y save_frame_file.

    28/02/2025, v1.1.0
        - Incluida posibilidad de procesar un nº concreto de fotograma
          del vídeo. Guarda la imagen del fotograma procesado.

    26/02/2025, v1.0.2
        - Añadida versión calculo ángulos para numpy.
    
    14/02/2025, v1.0.1
        - Cambiados nombres funciones a procesa_imagen y procesa_video
          en lugar de procesa_imagen_moderno y procesa_video_moderno.
    
    07/02/2025, v1.0.0
        - Iniciado a partir de pruebas anteriores.
        - Añadidas funciones split_dim_side y asigna_subcategorias_xr
        - Perfeccionada función procesar imagen y vídeo.
    
"""

from typing import List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

import time
from pathlib import Path

try:
    import cv2
except:
    raise ImportError(
        "Could not load the “opencv” library. \nInstall with conda install conda-forge::opencv."
    )

try:
    import mediapipe as mp

    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except:
    raise ImportError(
        "Could not load the “mediapipe” library.\nInstall with pip install mediapipe."
    )


# =============================================================================
# %% LOAD FUNCTIONS
# =============================================================================
# Nombres marcadores originales
# n_markers = [marker.name for marker in mp.solutions.pose.PoseLandmark]

# Nombres marcadores adaptados
N_MARKERS = [
    "nose",
    "eye_inner_L",
    "eye_center_L",
    "eye_outer_L",
    "eye_inner_R",
    "eye_center_R",
    "eye_outer_R",
    "ear_L",
    "ear_R",
    "mouth_L",
    "mouth_R",
    "shoulder_L",
    "shoulder_R",
    "elbow_L",
    "elbow_R",
    "wrist_L",
    "wrist_R",
    "pinky_L",
    "pinky_R",
    "index_L",
    "index_R",
    "thumb_L",
    "thumb_R",
    "hip_L",
    "hip_R",
    "knee_L",
    "knee_R",
    "ankle_L",
    "ankle_R",
    "heel_L",
    "heel_R",
    "toe_L",
    "toe_R",
]
N_MARCADORES = [
    "nariz",
    "ojo_int_L",
    "ojo_cent_L",
    "ojo_ext_L",
    "ojo_int_R",
    "ojo_cent_R",
    "ojo_ext_R",
    "oreja_L",
    "oreja_R",
    "boca_L",
    "boca_R",
    "hombro_L",
    "hombro_R",
    "codo_L",
    "codo_R",
    "muñeca_L",
    "muñeca_R",
    "meñique_L",
    "meñique_R",
    "indice_L",
    "indice_R",
    "pulgar_L",
    "pulgar_R",
    "cadera_L",
    "cadera_R",
    "rodilla_L",
    "rodilla_R",
    "tobillo_L",
    "tobillo_R",
    "talon_L",
    "talon_R",
    "toe_L",
    "toe_R",
]


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def extract_markers(data_mark, coords):
    return xr.DataArray(
        data=data_mark,
        dims=coords.keys(),
        coords=coords,
    ).transpose("marker", "axis")


def pose_landmarkers_to_xr(pose_landmarker_result, image):
    landmarks = pose_landmarker_result.pose_landmarks[0]
    data = np.full((len(landmarks), 5), np.nan)
    h, w, c = image.numpy_view().shape
    for i, _landmark in enumerate(landmarks):
        data[i, 0] = _landmark.x * w
        data[i, 1] = _landmark.y * h
        data[i, 2] = _landmark.z * c
        if _landmark.visibility:
            data[i, 3] = _landmark.visibility
        if _landmark.presence:
            data[i, 4] = _landmark.presence

    # Pasa a dataarray
    coords = {
        "marker": N_MARKERS,
        "axis": ["x", "y", "z", "visib", "presence"],
    }

    return xr.DataArray(
        data=data,
        dims=coords.keys(),
        coords=coords,
    )


def assign_subcategories_xr(da, estudio=None) -> xr.DataArray:
    if estudio is None:
        estudio = "X"

    da = da.assign_coords(
        estudio=("ID", [estudio] * len(da.ID)),
        particip=("ID", da.ID.to_series().str.split("_").str[1].to_list()),
        tipo=("ID", da.ID.to_series().str.split("_").str[5].to_list()),
        subtipo=("ID", da.ID.to_series().str.split("_").str[7].to_list()),
        repe=("ID", da.ID.to_series().str.split("_").str[6].to_list()),
    )

    return da


def split_dim_side(daData, n_bilat=None):
    """n_bilat: list, numpy array, dataarray
    list with bilateral variables, to be included repeated in the two coordinates (L and R)

    """
    if n_bilat is not None:
        bilat = daData.sel(marker=n_bilat)
        daData = daData.sel(marker=~daData.marker.isin(n_bilat))

    # Lo subdivide en side L y R
    L = daData.sel(marker=daData.marker.str.endswith("_L"))
    L = L.assign_coords(marker=L.marker.str.rstrip(to_strip="_L"))
    R = daData.sel(marker=daData.marker.str.endswith("_R"))
    R = R.assign_coords(marker=R.marker.str.rstrip(to_strip="_R"))

    # Si hay variables bilaterales, las añade al lado derecho e izquierdo
    if n_bilat is not None:
        L = xr.concat([L, bilat], dim="marker")
        R = xr.concat([R, bilat], dim="marker")

    daData_side = xr.concat([L, R], dim="side").assign_coords(side=["L", "R"])

    # daData_side = daData_side.transpose('marker', 'ID', 'qual', 'test', 'lap', 'side', 'time') #reordena las dimensiones

    return daData_side


def add_shoulder_hip_centers(daData):
    if "cadera" not in daData.marker and "hombro" not in daData.marker:
        raise ValueError("No hay marcadores caderas ni hombros")

    c_cad = (
        daData.sel(marker="cadera").mean("side").assign_coords(marker="centro_caderas")
    )
    c_hom = (
        daData.sel(marker="hombro").mean("side").assign_coords(marker="centro_hombros")
    )
    return xr.concat([daData, c_cad, c_hom], dim="marker")


def calculate_angle(points):
    # TODO: INCLUDE IMAGE DIMENSIONS TO SCALE?
    if len(points) == 3:
        a = np.array([points[0].x, points[0].y])
        b = np.array([points[1].x, points[1].y])
        c = np.array([points[1].x, points[1].y])
        d = np.array([points[2].x, points[2].y])
    elif len(points) == 4:
        a = np.array([points[0].x, points[0].y])
        b = np.array([points[1].x, points[1].y])
        c = np.array([points[2].x, points[2].y])
        d = np.array([points[3].x, points[3].y])

    radians = np.arctan2(np.linalg.norm(np.cross(a - b, d - c)), np.dot(a - b, d - c))
    angle = np.abs(np.rad2deg(radians))

    if angle > 180.0:
        angle = 360 - angle

    return round(angle)


def calculate_angle_np(points):
    # TODO: INCLUDE IMAGE DIMENSIONS TO SCALE?
    if len(points) == 3:
        a = points[0]  # np.array([points[0].x, points[0].y])
        b = points[1]  # np.array([points[1].x, points[1].y])
        c = points[1]  # np.array([points[1].x, points[1].y])
        d = points[2]  # np.array([points[2].x, points[2].y])
    elif len(points) == 4:
        a = points[0]  # np.array([points[0].x, points[0].y])
        b = points[1]  # np.array([points[1].x, points[1].y])
        c = points[2]  # np.array([points[2].x, points[2].y])
        d = points[3]  # np.array([points[3].x, points[3].y])

    radians = np.arctan2(np.linalg.norm(np.cross(a - b, d - c)), np.dot(a - b, d - c))
    angle = np.abs(np.rad2deg(radians))

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_angle_xr(daPoints: xr.DataArray) -> np.ndarray:
    # TODO: INCLUDE IMAGE DIMENSIONS TO SCALE?
    if len(daPoints.marker) == 3:
        a = (
            daPoints.isel(marker=0).sel(axis=["x", "y"]).data
        )  # np.array([puntos[0].x, puntos[0].y])
        b = (
            daPoints.isel(marker=1).sel(axis=["x", "y"]).data
        )  # np.array([puntos[1].x, puntos[1].y])
        c = (
            daPoints.isel(marker=1).sel(axis=["x", "y"]).data
        )  # np.array([puntos[1].x, puntos[1].y])
        d = (
            daPoints.isel(marker=2).sel(axis=["x", "y"]).data
        )  # np.array([puntos[2].x, puntos[2].y])
    elif len(daPoints.marker) == 4:
        a = (
            daPoints.isel(marker=0).sel(axis=["x", "y"]).data
        )  # np.array([puntos[0].x, puntos[0].y])
        b = (
            daPoints.isel(marker=1).sel(axis=["x", "y"]).data
        )  # np.array([puntos[1].x, puntos[1].y])
        c = (
            daPoints.isel(marker=2).sel(axis=["x", "y"]).data
        )  # np.array([puntos[2].x, puntos[2].y])
        d = (
            daPoints.isel(marker=3).sel(axis=["x", "y"]).data
        )  # np.array([puntos[3].x, puntos[3].y])

    radians = np.arctan2(np.linalg.norm(np.cross(a - b, d - c)), np.dot(a - b, d - c))
    angle = np.abs(np.rad2deg(radians))

    if angle > 180.0:
        angle = 360 - angle

    return angle


def process_video_old(file, side, fv=30, show=False):
    print("This function is deprecated. Use process_video instead.")
    return process_video(file, side, fv, show)


def process_video_deprecated(
    file, side: str, fv: int = 30, show: bool = False
) -> xr.DataArray:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(file.as_posix())
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Run MediaPipe Pose and draw pose landmarks.
    pTime = 0
    frame = 0
    data_mark = np.full((num_frames, 33, 3), np.nan)
    while frame < num_frames:
        success, img = cap.read()

        with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
        ) as pose:

            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            markers = []
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    # print(id, lm)
                    cx, cy, cz = (
                        int(lm.x * w),
                        int(lm.y * h),
                        lm.z,
                    )  # la coordenada z está sin escalar
                    markers.append([id, cx, cy, cz])
                markers = np.asarray(markers)
            else:
                markers = np.full((33, 2), np.nan)
            data_mark[frame] = markers[:, 1:]

            # image_hight, image_width, _ = img.shape
            annotated_image = img.copy()

            if results.pose_landmarks:
                if side == "D":
                    hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
                    foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
                    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                else:
                    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
                    foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                if side == "D":
                    shoulder2 = "hombro_R"
                    hip2 = "cadera_R"
                    knee2 = "rodilla_R"
                    ankle2 = "tobillo_R"
                    heel2 = "talon_R"
                    toe2 = "toe_R"
                else:
                    shoulder2 = "hombro_L"
                    hip2 = "cadera_L"
                    knee2 = "rodilla_L"
                    ankle2 = "tobillo_L"
                    heel2 = "talon_L"
                    toe2 = "toe_L"
                da_mark = extract_markers(data_mark[frame])
                ang_cadera = (
                    calculate_angle_xr(
                        marcadores=da_mark.sel(marcador=[knee2, hip2, shoulder2])
                    )
                    .round()
                    .astype(int)
                )
                ang_rodilla = (
                    calculate_angle_xr(
                        marcadores=da_mark.sel(marcador=[ankle2, knee2, hip2])
                    )
                    .round()
                    .astype(int)
                )
                ang_tobillo = (
                    calculate_angle_xr(
                        marcadores=da_mark.sel(marcador=[toe2, heel2, ankle2, knee2])
                    )
                    .round()
                    .astype(int)
                )
                """
                ang_cadera = calcula_angulo([knee, hip, shoulder])
                ang_rodilla = calcula_angulo([ankle, knee, hip])
                ang_tobillo = calcula_angulo([foot, heel, ankle, knee])
                """
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Pinta líneas modelo simplificado
                bMuestraModeloSimplificado = False
                if bMuestraModeloSimplificado and ang_cadera:
                    cv2.line(
                        annotated_image,
                        (int(shoulder.x * w), int(shoulder.y * h)),
                        (int(hip.x * w), int(hip.y * h)),
                        (0, 0, 255),
                        2,
                    )
                    cv2.line(
                        annotated_image,
                        (int(hip.x * w), int(hip.y * h)),
                        (int(knee.x * w), int(knee.y * h)),
                        (0, 0, 255),
                        2,
                    )
                    cv2.line(
                        annotated_image,
                        (int(knee.x * w), int(knee.y * h)),
                        (int(ankle.x * w), int(ankle.y * h)),
                        (0, 0, 255),
                        2,
                    )
                    cv2.line(
                        annotated_image,
                        (int(heel.x * w), int(heel.y * h)),
                        (int(foot.x * w), int(foot.y * h)),
                        (0, 0, 255),
                        2,
                    )

                    # Pinta puntos
                    for p in [hip, knee, ankle, heel, foot, shoulder]:
                        cx, cy = int(p.x * w), int(p.y * h)
                        cv2.circle(
                            annotated_image, (cx, cy), 8, (255, 255, 255), cv2.FILLED
                        )
                        cv2.circle(
                            annotated_image, (cx, cy), 8, (0, 0, 255), 2
                        )  # cv2.FILLED)

                # Escribe texto ángulos a su lado
                for artic, ang in zip(
                    [hip, knee, ankle], [ang_cadera, ang_rodilla, ang_tobillo]
                ):
                    cv2.putText(
                        annotated_image,
                        str(ang),
                        (int(artic.x * w) - 80, int(artic.y * h)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        cv2.LINE_4,
                    )

            else:
                print(f"No hay marcadores en fot: {frame}")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(
            annotated_image,
            str(int(fps)),
            (20, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_image,
            f"Frame {frame}/{num_frames}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_image,
            "q or Esc to exit",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if num_frames == 1:
            cv2.imwrite(
                (file.parent / (file.stem + "_angles.jpg")).as_posix(),
                annotated_image,
            )
            print(
                "\nImage saved",
                (file.parent / (file.stem + "_angles.jpg")).as_posix(),
            )

        cv2.imshow("Image", annotated_image)
        frame += 1

        if cv2.waitKey(1) in [ord("q"), 27]:
            break
    cv2.destroyAllWindows()

    # Pasa los marcadores a xarary
    coords = {
        "time": np.arange(0, num_frames) / fv,
        "marker": N_MARKERS,
        "axis": ["x", "y", "z"],
    }
    da = xr.DataArray(
        data=data_mark,
        dims=coords.keys(),
        coords=coords,
    ).transpose("marker", "axis", "time")
    if len(da.time) > 1:  # si es vídeo filtra
        da_filt = da  # filtrar_Butter(da, fr=fv, fc=6)
        da.sel(marcador=["tobillo_L", "tobillo_R"]).plot.line(
            x="time", col="axis", sharey=False
        )
        da_filt.sel(marcador=["tobillo_L", "tobillo_R"]).plot.line(
            x="time", col="axis", sharey=False
        )
        plt.show()

    else:
        da_filt = da

    # Calcula ángulos
    if side == "R":
        shoulder = "hombro_R"
        hip = "cadera_R"
        knee = "rodilla_R"
        ankle = "tobillo_R"
        heel = "talon_R"
        toe = "toe_R"
    else:
        shoulder = "hombro_L"
        hip = "cadera_L"
        knee = "rodilla_L"
        ankle = "tobillo_L"
        heel = "talon_L"
        toe = "toe_L"

    cadera = []
    rodilla = []
    tobillo = []
    for i in range(num_frames):
        cadera.append(
            calculate_angle_xr(
                marcadores=da_filt.isel(time=i).sel(marcador=[knee, hip, shoulder])
            )
        )
        rodilla.append(
            calculate_angle_xr(
                marcadores=da_filt.isel(time=i).sel(marcador=[ankle, knee, hip])
            )
        )
        tobillo.append(
            calculate_angle_xr(
                marcadores=da_filt.isel(time=i).sel(marcador=[toe, heel, ankle, knee])
            )
        )

    # Pasa los ángulos a xarary
    coords = {
        "angle": ["obj_cadera", "obj_rodilla", "obj_tobillo"],
        "time": np.arange(0, num_frames) / fv,
    }
    da_ang = xr.DataArray(
        data=np.array([cadera, rodilla, tobillo]),
        dims=coords.keys(),
        coords=coords,
    )
    if show:
        da_ang.plot.line(x="time", marker="o", size=3)
        sns.move_legend(plt.gca(), loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    # Escribe el resultado
    if len(da.time) > 1:  # si es vídeo se queda con el fotograma de mínimo áng rodilla
        # idxmin_ang_rodilla = da_ang.sel(angulo='obj_rodilla').idxmin('time')
        idxmin_ang_rodilla = (
            da_ang.sel(angulo="obj_rodilla")
            .isel(time=slice(None, int(len(da_ang.time) * 0.6)))
            .argmin("time")
        )  # busca en la mitad del salto para no coger la caída
        # da_ang.sel(angulo='obj_rodilla').isel(time=slice(None, int(len(da_ang.time)*0.6))).plot.line(x='time', marker='o')
        da_ang_result = da_ang.isel(time=idxmin_ang_rodilla.data)
    else:
        da_ang_result = da_ang

    dfResumen = (
        da_ang_result.round()
        .astype(int)
        .to_dataframe(name="obj_ang")
        .reset_index("angulo")
    )
    dfResumen = dfResumen.replace(
        {
            "obj_cadera": "obj_cadera=",
            "obj_rodilla": "obj_rodilla=",
            "obj_tobillo": "obj_tobillo=",
        }
    )
    dfResumen["obj_ang"] = dfResumen["obj_ang"].astype(int)

    dfResumen[["angle", "obj_ang"]].to_csv(
        (file.parent / (file.stem + "_Objetivo_ang")).with_suffix(".txt"),
        sep=" ",
        index=False,
        header=False,
    )
    print(
        f"File saved {(file.parent / (file.stem+'_Objetivo_ang')).with_suffix('.txt')}"
    )

    return da_ang_result


def process_image_old(file=None, image=None, model_path=None, show=False):
    """
    show = False, 'markers' o 'mask'
    """
    # STEP 2: Create a PoseLandmarker object.
    base_options = python.BaseOptions(
        model_asset_path=Path(r"pose_landmarker_heavy.task")
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        min_pose_detection_confidence=0.9,
        min_pose_presence_confidence=0.9,
        min_tracking_confidence=0.9,
        output_segmentation_masks=True,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    if image is None:
        image = mp.Image.create_from_file((file).as_posix())
    else:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    if show:
        if show == "markers":
            cv2.imshow("window_name", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)

            # closing all open windows
            cv2.destroyAllWindows()

        elif show == "mask":
            # Ejemplo de máscara
            segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
            visualized_mask = (
                np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
            )

            cv2.imshow("window_name", visualized_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return detection_result


def process_image_modern(
    file=None,
    image=None,
    mpdc=0.8,
    mppc=0.8,
    mtc=0.8,
    model_path=None,
    show=False,
    format="xr",
):
    print("This function is deprecated. Use process_image instead.")
    return process_image(
        file,
        image,
        mpdc=mpdc,
        mppc=mppc,
        mtc=mtc,
        model_path=model_path,
        show=show,
        format=format,
    )


def process_image(
    file: str | Path | None = None,
    image=None,
    mpdc: float = 0.8,
    mppc: float = 0.8,
    mtc: float = 0.8,
    model_path: str | Path | None = None,
    show: bool | str = False,
    format: str = "xr",
):
    """
    Process a single image using the PoseLandmarker model.

    Parameters
    ----------
    file : str or Path
        Path to the image file.
    image : numpy array
        Image data.
    mpdc : float
        Minimum pose detection confidence.
    mppc : float
        Minimum pose presence confidence.
    mtc : float
        Minimum tracking confidence.
    model_path : str or Path
        Path to the PoseLandmarker model.
    show : bool or str
        If True, show the image with pose landmarks.
        If 'colab' open in Google Colab
        If 'mask', show the segmentation mask.
    format : str
        Format of the output.
        'raw' returns the PoseLandmarkerResult object,
        'xr' returns an xarray object.

    Returns
    -------
    detection_result : PoseLandmarkerResult or xarray.DataArray
        Object containing the pose landmarks and segmentation mask.
    """

    if isinstance(file, Path):
        file = file.as_posix()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=mpdc,
        min_pose_presence_confidence=mppc,
        min_tracking_confidence=mtc,
        output_segmentation_masks=False,
    )

    # LOAD IMAGE
    if image is None:
        image = mp.Image.create_from_file(file)
        # image = cv2.imread(file)
    elif isinstance(image, np.ndarray):  # does it work for cv2 image?
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Load the input image from a numpy array.
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Perform pose landmarking on the provided single image.
        # The pose landmarker must be created with the image mode.
        pose_landmarker_result = landmarker.detect(image)

    if show is not False:
        annotated_image = draw_landmarks_on_image(
            image.numpy_view(), pose_landmarker_result
        )

        if show == "colab":
            from google.colab.patches import cv2_imshow

            cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)

            # closing all open windows
            cv2.destroyAllWindows()

        elif show == True:
            cv2.imshow(
                "Marker detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            )
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)

            # closing all open windows
            cv2.destroyAllWindows()

        elif show == "mask":
            segmentation_mask = pose_landmarker_result.segmentation_masks[
                0
            ].numpy_view()
            visualized_mask = (
                np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
            )

            cv2.imshow("Mask results", visualized_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if format == "raw":
        return pose_landmarker_result
    elif format == "xr":
        return pose_landmarkers_to_xr(pose_landmarker_result)

    else:
        raise ValueError(f"Format {format} not recognized. Must be 'raw' or 'xr'")


def procesS_video_modern(
    file,
    fv=30,
    n_vars_load=None,
    mpdc=0.5,
    mppc=0.5,
    mtc=0.5,
    model_path=None,
    show=False,
):
    print("This function is deprecated. Use procesa_video instead.")
    return process_video(
        file,
        fv,
        n_vars_load,
        mpdc,
        mppc,
        mtc,
        model_path,
        show,
    )


def process_video(
    file: str | Path,
    fv: int = 30,
    n_vars_load: List[str] | None = None,
    mpdc: float = 0.5,
    mppc: float = 0.5,
    mtc: float = 0.5,
    model_path: str | Path | None = None,
    num_frame: int | None = None,
    save_frame_file: bool | Path | None = None,
    show: bool | str = False,
    show_every_frames: int = 10,
) -> xr.DataArray:
    """
    Processes a video file to extract pose landmarks using MediaPipe Pose.

    Parameters
    ----------
    file : str or Path
        Path to the video file to be processed.
    fv : int, optional
        Frame rate of the video, defaults to 30.
    n_vars_load : list os str, optional
        List of variables to load, defaults to None.
    mpdc : float, optional
        Minimum pose detection confidence, defaults to 0.5.
    mppc : float, optional
        Minimum pose presence confidence, defaults to 0.5.
    mtc : float, optional
        Minimum tracking confidence, defaults to 0.5.
    model_path : str or Path, optional
        Path to the model file, defaults to None, which uses "pose_landmarker_heavy.task".
    num_frame : int, optional
        Number of frames to process, if None, all frames are processed.
    save_frame_file : bool or Path, optional
        Defaults to None
        True: save to the same folder
        Path: save to the proposed folderto save frames to file.
    show : bool or str, optional
        If False, no display is shown. If True, it displays the frames with markers in a local environment.
        If 'colab', it displays the frames in Google Colab.
    show_every_frames : int, optional
        Frame skip number to display.

    Returns
    -------
    xarray.DataArray
        A DataArray containing the pose landmarks and additional metadata.
    """

    if not isinstance(file, Path):
        file = Path(file)

    t_ini = time.perf_counter()
    # print(f"Processing video {file.name}...")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    if model_path is None:
        model_path = "pose_landmarker_heavy.task"
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=Path(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=mpdc,
        min_pose_presence_confidence=mppc,
        min_tracking_confidence=mtc,
        output_segmentation_masks=False,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        cap = cv2.VideoCapture(file.as_posix())
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f"Video fps:{vid_fps}")
        if num_frame is not None and (num_frame > num_frames or num_frame < 0):
            raise ValueError(
                f"num_frame {num_frame} is out of range of video frames (0-{num_frames})"
            )

        pTime = 0
        frame = 0
        # data_mark = np.full((num_frames, 33, 3), np.nan)

        # Precreate Dataarray
        coords = {
            "time": np.arange(0, num_frames),  # / fv,
            "marker": N_MARKERS,
            "axis": ["x", "y", "z", "visib", "presence"],
        }
        daMarkers = (
            xr.DataArray(
                data=np.full((num_frames, 33, 5), np.nan),
                dims=coords.keys(),
                coords=coords,
            ).expand_dims({"ID": [file.stem]})
            # .assign_coords(visibiility=("time", np.full(num_frames, np.nan)))
            .copy()
        )  # .transpose("marker", "axis", "time")

        # Process frame by frame
        while frame < num_frames:  # cap.isOpened()
            if num_frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)  # restar 1?????
                if show not in [True, "colab"]:
                    show = True

            success, img = cap.read()
            if not success:
                break

            # Reset colors. It is not necessary but it does not seem to slow down
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Test readjustment
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

            # Perform pose landmarking on the provided single image.
            # The pose landmarker must be created with the video mode.
            pose_landmarker_result = landmarker.detect_for_video(
                mp_image, int(frame * 1 / fv * 1000)
            )

            # Loop through the detected poses to visualize.
            if pose_landmarker_result.pose_landmarks:
                pose_landmarks = pose_landmarker_result.pose_landmarks[0]

                """pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z,
                            visibility=landmark.visibility,
                        )
                        for landmark in pose_landmarks
                    ]
                )"""

                # if pose_landmarks is not None:

                dat = []
                [
                    dat.append(
                        [
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility,
                            landmark.presence,
                        ]
                    )
                    for landmark in pose_landmarks
                ]

            else:
                dat = np.full((33, 5), np.nan)

            daMarkers.loc[dict(ID=file.stem, time=frame)] = np.asanyarray(dat)

            # Calcula fps
            cTime = time.time()
            fps = 1 / (cTime - pTime) if cTime != pTime else 0

            ############################

            # Anota en las imágenes
            if show in [True, "colab"]:
                annotated_image = draw_landmarks_on_image(img, pose_landmarker_result)
                cv2.putText(
                    annotated_image,
                    "q or Esc to exit",
                    (30, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                show_frame = frame if num_frame is None else frame + num_frame
                cv2.putText(
                    annotated_image,
                    f"Frame {show_frame}/{num_frames} fps: {fps:.2f}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                if show == True:
                    cv2.imshow(
                        r"Marker detection",
                        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
                    )

                elif show == "colab":
                    from google.colab.patches import cv2_imshow

                    if frame % show_every_frames == 0:
                        cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                        # waits for user to press any key
                        # (this is necessary to avoid Python kernel form crashing)
                        cv2.waitKey(0)

                        # closing all open windows
                        cv2.destroyAllWindows()

                # Save the selected num_frame
                if num_frame is not None and save_frame_file not in [None, False]:
                    if save_frame_file == True:
                        save_frame_file = file.parent

                    if isinstance(save_frame_file, Path):
                        if not save_frame_file.is_dir():
                            save_frame_file.mkdir(parents=True)
                        # save_frame_file = save_frame_file.as_posix()

                    saved = cv2.imwrite(
                        (
                            (
                                save_frame_file / f"{file.stem}_fot{num_frame}"
                            ).with_suffix(".jpg")
                        ).as_posix(),
                        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
                    )
                    if saved:
                        print(f"Saved frame {num_frame}")
                    else:
                        print(f"Error saving file {file} frame {num_frame}")

            else:  # if show== False
                if frame % show_every_frames == 0:
                    print(f"Frame {frame}/{num_frames} fps: {fps:.2f}")

            # Waits for user to press any key
            if cv2.waitKey(1) in [ord("q"), 27] or num_frame is not None:
                break
            # cv2.waitKey(0)

            pTime = cTime
            frame += 1

    # closing all open windows
    cv2.destroyAllWindows()

    print(f"Video processed in {time.perf_counter() - t_ini:.2f} s")

    # Adjust time coordinate
    if num_frame is None:
        daMarkers = daMarkers.assign_coords(time=np.arange(0, num_frames) / fv)
    else:  # con un solo fotograma
        # daMarkers = daMarkers.isel(time=slice(frame - 3, frame + 3)).assign_coords(
        #     time=np.arange(frame - 3, frame + 3) / fv
        # )
        daMarkers = daMarkers.isel(time=frame).assign_coords(time=num_frame / fv)

    if n_vars_load is not None:
        daMarkers = daMarkers.sel(marker=n_vars_load)

    # Invert y coordinates
    daMarkers.loc[dict(axis="y")] = -daMarkers.loc[dict(axis="y")]

    return daMarkers


def process_video_mixed(file, fv=30, show=False):

    t_ini = time.perf_counter()
    print(f"Processing vídeo{file.name}...")

    cap = cv2.VideoCapture(file.as_posix())
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video fps:{vid_fps}")

    pTime = 0
    frame = 0
    data_mark = np.full((num_frames, 33, 3), np.nan)

    while frame < num_frames:  # cap.isOpened()
        success, img = cap.read()
        if not success:
            break
        # frame += 1

        # Reajusta colores. No es necesario pero parece que no retrasa
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Probar reajuste
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        # Procesa imagen
        detection_result = process_image(file=None, image=img)  # img)

        # detection_result.pose_landmarks[0][0].x
        if detection_result.pose_landmarks:
            # data_mark = np.full((num_frames, 33, 3), np.nan)
            markers = []
            # landmarks = detection_result.pose_landmarks
            h, w, c = img.shape
            for id, lm in enumerate(detection_result.pose_landmarks[0]):
                # print(id, lm)
                cx, cy, cz = (
                    int(lm.x * w),
                    int(lm.y * h),
                    int(lm.z * 1000),
                )  # la coordenada z está sin escalar
                markers.append([id, cx, cy, cz])
            markers = np.asarray(markers)
        else:  # si no ha detectado marcadores
            markers = np.full((33, 2), np.nan)
        data_mark[frame] = markers[:, 1:]

        # Calcula fps
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0

        # Muestra imágenes
        if show == "markers":
            annotated_image = draw_landmarks_on_image(img, detection_result)
            cv2.putText(
                annotated_image,
                "q or Esc to exit",
                (30, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated_image,
                f"Frame {frame}/{num_frames} fps: {fps:.2f}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.imshow(
                r"Marker detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            )

        elif show == "mask":
            # Ejemplo de máscara
            segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
            visualized_mask = (
                np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
            )

            cv2.imshow("window_name", visualized_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print(f"Frame {frame}/{num_frames} fps: {fps:.2f}")

        # print(frame, fps)

        pTime = cTime
        frame += 1

        # waits for user to press any key
        if cv2.waitKey(1) in [ord("q"), 27]:
            break
        # cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
    print(f"Processing finished in {time.perf_counter() - t_ini:.2f} s")

    # Pasa los marcadores a xarary
    coords = {
        "time": np.arange(0, num_frames) / fv,
        "marker": N_MARKERS,
        "axis": ["x", "y", "z"],
    }
    da = xr.DataArray(
        data=data_mark,
        dims=coords.keys(),
        coords=coords,
    )  # .transpose("marcador", "eje", "time")

    return da


def plot_pose_2D(
    daData,
    dim_col: str = "ID",
    x: str = "x",
    y: str = "y",
    frames: int | List[int] = [None, None],
):
    """
    Creates a 2D scatter plot of the given xr Dataaray, `daData`, for specified frames.
    Probably not necessary function, but to remenber how to do it.

    Parameters:
    daData : xarray.DataArray
        The data array containing pose data to be plotted.
    dim_col : str
        The dimension along which to create separate subplots.
    x : str
        The coordinate to use for the x-axis in the scatter plot.
    y : str
        The coordinate to use for the y-axis in the scatter plot.
    frames : int or list of int
        If a single integer, it specifies the frame index to be plotted.
        If a tuple of two integers, it specifies the start and end frame indices to be plotted.

    Returns:
    None
    """
    if isinstance(frames, int):
        frames = [frames, frames + 1]
    if "time" in daData.dims:
        daData.isel(time=slice(frames[0], frames[1])).to_dataset("axis").plot.scatter(
            x=x, y=y, col=dim_col, col_wrap=3, linewidths=0.5
        )
    else:
        daData.to_dataset("axis").plot.scatter(
            x=x, y=y, col=dim_col, col_wrap=3, linewidths=0.5
        )


# =============================================================================
# %% TESTS
# =============================================================================
if __name__ == "__main__":

    # import biomdp.image_pose as impo

    work_path = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisAaron\Registros\CambioDireccion\Videos\Cortados"
    )
    # Download model if not already downloaded
    import urllib.request

    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        work_path / "pose_landmarker_heavy.task",
    )

    file = work_path / "COD_S01_G1_A_90_I_1_FRONT.mov"

    # Process the complete video
    daMarkers = process_video(
        file,
        fv=240,
        mpdc=0.9,
        mppc=0.95,
        mtc=0.95,
        show=True,
        # show_every_frames=20,
        model_path=work_path / "pose_landmarker_heavy.task",
    )
    daMarkers = split_dim_side(daMarkers)
    daMarkers.isel(ID=0).sel(
        axis="x", marker=["hip", "knee", "ankle", "heel", "toe"]
    ).plot.line(x="time", col="side")
    plot_pose_2D(daMarkers, frames=[100, None])

    # Process and saves one single frame
    daMarkers_frame = process_video(
        file,
        fv=240,
        mpdc=0.9,
        mppc=0.95,
        mtc=0.95,
        num_frame=178,
        show=False,
        model_path=work_path / "pose_landmarker_heavy.task",
    )
    daMarkers_frame = split_dim_side(daMarkers_frame)
    plot_pose_2D(daMarkers_frame)
