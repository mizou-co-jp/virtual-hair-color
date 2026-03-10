"""
髪色を黒に変更する仮想カメラアプリ

使い方:
  python hair_to_black.py

操作:
  q: 終了
  s: 髪色変更のON/OFF切り替え
  +/-: ブレンド強度の調整
"""

import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ImageSegmenter,
    ImageSegmenterOptions,
    RunningMode,
)

try:
    import pyvirtualcam
    HAS_VCAM = True
except ImportError:
    HAS_VCAM = False

MODEL_PATH = "hair_segmenter.tflite"

# 仮想カメラの解像度
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30

# 髪を黒くする設定
# HSVで彩度を下げ、明度を下げることで黒髪にする
TARGET_SATURATION = 30   # 低彩度（0-255）
TARGET_VALUE = 40        # 低明度（0-255）で黒に近づける
BLEND_ALPHA = 0.85       # ブレンド強度（0.0-1.0）


def create_segmenter():
    """MediaPipe髪セグメンターを作成"""
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        output_category_mask=True,
    )
    return ImageSegmenter.create_from_options(options)


def make_hair_black(frame_bgr, hair_mask, alpha):
    """髪領域を黒色に変換"""
    mask_bool = hair_mask > 0

    if not np.any(mask_bool):
        return frame_bgr

    # マスクをぼかしてエッジを滑らかにする
    mask_float = hair_mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (15, 15), 5)
    mask_3ch = np.stack([mask_float] * 3, axis=-1)

    # HSVに変換して彩度と明度を操作
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    black_hsv = hsv.copy()
    black_hsv[:, :, 1] = TARGET_SATURATION  # 低彩度
    black_hsv[:, :, 2] = np.minimum(hsv[:, :, 2], TARGET_VALUE)  # 明度を下げる

    black_bgr = cv2.cvtColor(black_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # マスク領域のみブレンド
    blended = (
        frame_bgr.astype(np.float32) * (1 - mask_3ch * alpha)
        + black_bgr.astype(np.float32) * (mask_3ch * alpha)
    )

    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    # カメラを開く
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"カメラ解像度: {actual_w}x{actual_h}")

    # セグメンター初期化
    segmenter = create_segmenter()
    print("髪セグメンテーションモデル読み込み完了")

    # 仮想カメラ初期化
    vcam = None
    if HAS_VCAM:
        try:
            vcam = pyvirtualcam.Camera(
                width=actual_w,
                height=actual_h,
                fps=CAM_FPS,
                fmt=pyvirtualcam.PixelFormat.RGB,
            )
            print(f"仮想カメラ開始: {vcam.device}")
        except Exception as e:
            print(f"仮想カメラの初期化に失敗: {e}")
            print("OBS Studioをインストールし、仮想カメラを一度起動してください")
            vcam = None
    else:
        print("pyvirtualcamが見つかりません。プレビューのみで動作します。")

    enabled = True
    alpha = BLEND_ALPHA
    frame_count = 0
    start_time = time.time()

    print("\n--- 操作方法 ---")
    print("q: 終了")
    print("s: 髪色変更 ON/OFF")
    print("+/-: ブレンド強度調整")
    print("----------------\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレーム取得に失敗")
                break

            # 必要に応じてリサイズ
            if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
                frame = cv2.resize(frame, (actual_w, actual_h))

            output = frame.copy()

            if enabled:
                # BGR→RGBに変換してMediaPipeに渡す
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame
                )

                timestamp_ms = int(time.time() * 1000)
                result = segmenter.segment_for_video(mp_image, timestamp_ms)

                if result.category_mask is not None:
                    hair_mask = result.category_mask.numpy_view()
                    # マスクを0-255にスケール
                    hair_mask_scaled = (hair_mask * 255).astype(np.uint8)
                    output = make_hair_black(frame, hair_mask_scaled, alpha)

            # 仮想カメラに送信
            if vcam is not None:
                rgb_output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                vcam.send(rgb_output)
                vcam.sleep_until_next_frame()

            # プレビュー表示
            # ステータス表示
            status = "ON" if enabled else "OFF"
            cv2.putText(
                output,
                f"Hair->Black: {status} | Alpha: {alpha:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # FPS表示
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(
                    output,
                    f"FPS: {fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Hair to Black - Virtual Camera", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                enabled = not enabled
                print(f"髪色変更: {'ON' if enabled else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                alpha = min(1.0, alpha + 0.05)
                print(f"ブレンド強度: {alpha:.2f}")
            elif key == ord("-"):
                alpha = max(0.0, alpha - 0.05)
                print(f"ブレンド強度: {alpha:.2f}")

    except KeyboardInterrupt:
        print("\n中断されました")
    finally:
        cap.release()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
        segmenter.close()
        print("終了しました")


if __name__ == "__main__":
    main()
