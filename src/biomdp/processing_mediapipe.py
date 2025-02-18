"""
Created on Thu Jul 04 09:35:20 2024

@author: Jose L. L. Elvira

Carga y procesa vídeo con cambios de dirección.

INSTALAR:
# - pip install OpenCV-python
- conda install -c conda-forge opencv

- pip install mediapipe

"""

# =============================================================================
# %% IMPORTA LIBRERIAS
# =============================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

import time
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


__author__ = "Jose Luis Lopez Elvira"
__version__ = "v.1.0.1"
__date__ = "14/02/2025"


"""
Modificaciones:
    14/02/2025, v1.0.1
        - Cambiados nombres funciones a procesa_imagen y procesa_video
          en lugar de procesa_imagen_moderno y procesa_video_moderno.
    
    07/02/2025, v1.0.0
        - Iniciado a partir de pruebas anteriores.
        - Añadidas funciones separa_dim_lado y asigna_subcategorias_xr
        - Perfeccionada función procesar imagen y vídeo.
    
"""


# =============================================================================
# %% CARGA FUNCIONES
# =============================================================================
# Nombres marcadores originales
# n_markers = [marker.name for marker in mp.solutions.pose.PoseLandmark]

# Nombres marcadores adaptados
N_MARKERS = [
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


def extrae_marcadores(data_mark):
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


def asigna_subcategorias_xr(da, estudio=None) -> xr.DataArray:
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


def separa_dim_lado(daDatos, n_bilat=None):
    """n_bilat: lista, numpy array, dataarray
    lista con variables bilaterales, a incluir repetida en las dos coordenadas (L y R)

    """
    if n_bilat is not None:
        bilat = daDatos.sel(marker=n_bilat)
        daDatos = daDatos.sel(marker=~daDatos.marker.isin(n_bilat))

    # Lo subdivide en side L y R
    L = daDatos.sel(marker=daDatos.marker.str.endswith("_L"))
    L = L.assign_coords(marker=L.marker.str.rstrip(to_strip="_L"))
    R = daDatos.sel(marker=daDatos.marker.str.endswith("_R"))
    R = R.assign_coords(marker=R.marker.str.rstrip(to_strip="_R"))

    # Si hay variables bilaterales, las añade al lado derecho e izquierdo
    if n_bilat is not None:
        L = xr.concat([L, bilat], dim="marker")
        R = xr.concat([R, bilat], dim="marker")

    daDatos_side = xr.concat([L, R], dim="side").assign_coords(side=["L", "R"])

    # daDatos_side = daDatos_side.transpose('marker', 'ID', 'qual', 'test', 'lap', 'side', 'time') #reordena las dimensiones

    return daDatos_side


def calcula_angulo(puntos):
    if len(puntos) == 3:
        a = np.array([puntos[0].x, puntos[0].y])
        b = np.array([puntos[1].x, puntos[1].y])
        c = np.array([puntos[1].x, puntos[1].y])
        d = np.array([puntos[2].x, puntos[2].y])
    elif len(puntos) == 4:
        a = np.array([puntos[0].x, puntos[0].y])
        b = np.array([puntos[1].x, puntos[1].y])
        c = np.array([puntos[2].x, puntos[2].y])
        d = np.array([puntos[3].x, puntos[3].y])

    radians = np.arctan2(np.linalg.norm(np.cross(a - b, d - c)), np.dot(a - b, d - c))
    angle = np.abs(np.rad2deg(radians))

    if angle > 180.0:
        angle = 360 - angle

    return round(angle)


def calcula_angulo_xr(markers: xr.DataArray) -> np.ndarray:
    if len(markers) == 3:
        a = (
            markers.isel(marker=0).sel(axis=["x", "y"]).data
        )  # np.array([puntos[0].x, puntos[0].y])
        b = (
            markers.isel(marker=1).sel(axis=["x", "y"]).data
        )  # np.array([puntos[1].x, puntos[1].y])
        c = (
            markers.isel(marker=1).sel(axis=["x", "y"]).data
        )  # np.array([puntos[1].x, puntos[1].y])
        d = (
            markers.isel(marker=2).sel(axis=["x", "y"]).data
        )  # np.array([puntos[2].x, puntos[2].y])
    elif len(markers) == 4:
        a = (
            markers.isel(marker=0).sel(axis=["x", "y"]).data
        )  # np.array([puntos[0].x, puntos[0].y])
        b = (
            markers.isel(marker=1).sel(axis=["x", "y"]).data
        )  # np.array([puntos[1].x, puntos[1].y])
        c = (
            markers.isel(marker=2).sel(axis=["x", "y"]).data
        )  # np.array([puntos[2].x, puntos[2].y])
        d = (
            markers.isel(marker=3).sel(axis=["x", "y"]).data
        )  # np.array([puntos[3].x, puntos[3].y])

    radians = np.arctan2(np.linalg.norm(np.cross(a - b, d - c)), np.dot(a - b, d - c))
    angle = np.abs(np.rad2deg(radians))

    if angle > 180.0:
        angle = 360 - angle

    return angle


def procesa_video_antiguo(file, lado, fv=30, show=False):
    print("This function is deprecated. Use procesa_video instead.")
    return procesa_video_antiguo(file, lado, fv, show)


def procesa_video_deprecated(file, lado, fv=30, show=False):
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
                if lado == "D":
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
                if lado == "D":
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
                da_mark = extrae_marcadores(data_mark[frame])
                ang_cadera = (
                    calcula_angulo_xr(
                        marcadores=da_mark.sel(marcador=[knee2, hip2, shoulder2])
                    )
                    .round()
                    .astype(int)
                )
                ang_rodilla = (
                    calcula_angulo_xr(
                        marcadores=da_mark.sel(marcador=[ankle2, knee2, hip2])
                    )
                    .round()
                    .astype(int)
                )
                ang_tobillo = (
                    calcula_angulo_xr(
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
            "pulsa q para salir",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if num_frames == 1:
            cv2.imwrite(
                (file.parent / (file.stem + "_angulos.jpg")).as_posix(),
                annotated_image,
            )
            print(
                "\nGuardada la imagen",
                (file.parent / (file.stem + "_angulos.jpg")).as_posix(),
            )

        cv2.imshow("Imagen", annotated_image)
        frame += 1

        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()

    # Pasa los marcadores a xarary
    coords = {
        "time": np.arange(0, num_frames) / fv,
        "marcador": N_MARKERS,
        "eje": ["x", "y", "z"],
    }
    da = xr.DataArray(
        data=data_mark,
        dims=coords.keys(),
        coords=coords,
    ).transpose("marcador", "eje", "time")
    if len(da.time) > 1:  # si es vídeo filtra
        da_filt = da  # filtrar_Butter(da, fr=fv, fc=6)
        da.sel(marcador=["tobillo_L", "tobillo_R"]).plot.line(
            x="time", col="eje", sharey=False
        )
        da_filt.sel(marcador=["tobillo_L", "tobillo_R"]).plot.line(
            x="time", col="eje", sharey=False
        )
        plt.show()

    else:
        da_filt = da

    # Calcula ángulos
    if lado == "D":
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
            calcula_angulo_xr(
                marcadores=da_filt.isel(time=i).sel(marcador=[knee, hip, shoulder])
            )
        )
        rodilla.append(
            calcula_angulo_xr(
                marcadores=da_filt.isel(time=i).sel(marcador=[ankle, knee, hip])
            )
        )
        tobillo.append(
            calcula_angulo_xr(
                marcadores=da_filt.isel(time=i).sel(marcador=[toe, heel, ankle, knee])
            )
        )

    # Pasa los ángulos a xarary
    coords = {
        "angulo": ["obj_cadera", "obj_rodilla", "obj_tobillo"],
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

    dfResumen[["angulo", "obj_ang"]].to_csv(
        (file.parent / (file.stem + "_Objetivo_ang")).with_suffix(".txt"),
        sep=" ",
        index=False,
        header=False,
    )
    print(
        f"Guardado el archivo {(file.parent / (file.stem+'_Objetivo_ang')).with_suffix('.txt')}"
    )

    return da_ang_result


def procesa_imagen_antiguo(file=None, image=None, model_path=None, show=False):
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


def procesa_imagen_moderno(
    file=None,
    image=None,
    mpdc=0.8,
    mppc=0.8,
    mtc=0.8,
    model_path=None,
    show=False,
    format="xr",
):
    print("This function is deprecated. Use procesa_imagen instead.")
    return procesa_imagen(
        file,
        image,
        mpdc=mpdc,
        mppc=mppc,
        mtc=mtc,
        model_path=model_path,
        show=show,
        format=format,
    )


def procesa_imagen(
    file=None,
    image=None,
    mpdc=0.8,
    mppc=0.8,
    mtc=0.8,
    model_path=None,
    show=False,
    format="xr",
):
    """
    show: 'markers'
          'mask'
    format: 'raw', as PoseLandmarkerResult
            'xr', as xarray
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

    # CARGA IMAGEN
    # EJEMPLO https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
    # img = cv2.imread("image.jpg")
    if image is None:
        image = mp.Image.create_from_file(file)
    elif isinstance(image, np.ndarray):  # funciona para cuando es imagen cv2?
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Load the input image from a numpy array.
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Perform pose landmarking on the provided single image.
        # The pose landmarker must be created with the image mode.
        pose_landmarker_result = landmarker.detect(image)

    if show:
        annotated_image = draw_landmarks_on_image(
            image.numpy_view(), pose_landmarker_result
        )

        if show == "markers":
            cv2.imshow("window_name", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)

            # closing all open windows
            cv2.destroyAllWindows()

        elif show == "mask":
            # Ejemplo de máscara
            segmentation_mask = pose_landmarker_result.segmentation_masks[
                0
            ].numpy_view()
            visualized_mask = (
                np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
            )

            cv2.imshow("window_name", visualized_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if format == "raw":
        return pose_landmarker_result
    elif format == "xr":
        return pose_landmarkers_to_xr(pose_landmarker_result)

    else:
        raise ValueError(f"Format {format} not recognized. Must be 'raw' or 'xr'")


def procesa_video_moderno(
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
    return procesa_video(
        file,
        fv,
        n_vars_load,
        mpdc,
        mppc,
        mtc,
        model_path,
        show,
    )


def procesa_video(
    file,
    fv=30,
    n_vars_load=None,
    mpdc=0.5,
    mppc=0.5,
    mtc=0.5,
    model_path=None,
    show=False,
):
    """
    mpdc = min_pose_detection_confidence
    mppc = min_pose_presence_confidence
    mtc = min_tracking_confidence
    show = False, 'markers' o 'mask'
    """
    t_ini = time.perf_counter()
    # print(f"Procesando vídeo{file.name}...")

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
        print(f"Video fps:{vid_fps}")

        pTime = 0
        frame = 0
        # data_mark = np.full((num_frames, 33, 3), np.nan)

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

        # Procesa fotograma a fotograma
        while frame < num_frames:  # cap.isOpened()
            success, img = cap.read()
            if not success:
                break

            # Reajusta colores. No es necesario pero parece que no retrasa
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Probar reajuste
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
            # frame+=1
            ############################

            # Muestra imágenes
            if show == "markers":
                annotated_image = draw_landmarks_on_image(img, pose_landmarker_result)
                cv2.putText(
                    annotated_image,
                    "q para salir",
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
                    r"Marker detection",
                    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
                )

            elif show == "mask":
                # Ejemplo de máscara
                segmentation_mask = pose_landmarker_result.segmentation_masks[
                    0
                ].numpy_view()
                visualized_mask = (
                    np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
                )

                cv2.imshow("window_name", visualized_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            else:
                if frame % 10 == 0:
                    print(f"Frame {frame}/{num_frames} fps: {fps:.2f}")

            pTime = cTime
            frame += 1

            # waits for user to press any key
            if cv2.waitKey(1) == ord("q"):
                break
            # cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

    print(f"Terminado el procesado en {time.perf_counter() - t_ini:.2f} s")

    # Corrige la coordenada time
    daMarkers = daMarkers.assign_coords(time=np.arange(0, num_frames) / fv)

    if n_vars_load is not None:
        daMarkers = daMarkers.sel(marker=n_vars_load)
    return daMarkers


def procesa_video_mixto(file, fv=30, show=False):

    t_ini = time.perf_counter()
    print(f"Procesando vídeo{file.name}...")

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
        detection_result = procesa_imagen(file=None, image=img)  # img)

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
                "q para salir",
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
        if cv2.waitKey(1) == ord("q"):
            break
        # cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
    print(f"Terminado el procesado en {time.perf_counter() - t_ini:.2f} s")

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



# =============================================================================
# %% PRUEBAS
# =============================================================================
if __name__ == "__main__":

    
