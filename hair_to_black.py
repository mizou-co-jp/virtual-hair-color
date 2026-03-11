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
DARKENING_FACTOR = 0.35  # 明度をどれだけ暗くするか（小さいほど暗い）
DESAT_FACTOR = 0.15      # 彩度をどれだけ残すか（小さいほど無彩色）
BLEND_ALPHA = 0.95       # ブレンド強度（0.0-1.0）

# マスク精製設定
GUIDED_FILTER_RADIUS = 8     # ガイドフィルタ半径（大きいほど滑らか）
GUIDED_FILTER_EPS = 0.01     # ガイドフィルタ正則化（小さいほどエッジに忠実）
MORPH_ERODE_SIZE = 3         # 収縮カーネルサイズ（マスク内側のノイズ除去）
MORPH_DILATE_SIZE = 7        # 膨張カーネルサイズ（マスクを広げて取りこぼし防止）
REFINE_ITERATIONS = 3        # マスク精製の反復回数
MASK_THRESHOLD = 0.1         # マスク閾値（低いほど広く検出、0.0-1.0）
TEMPORAL_SMOOTH = 0.6        # 時間平滑化（前フレームのマスクをどれだけ混ぜるか）


def create_segmenter():
    """MediaPipe髪セグメンターを作成"""
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        output_category_mask=True,
    )
    return ImageSegmenter.create_from_options(options)


def guided_filter(guide, src, radius, eps):
    """エッジ保持ガイドフィルタ（生え際を画像のエッジに沿わせる）"""
    guide_f = guide.astype(np.float32) / 255.0
    src_f = src.astype(np.float32)

    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_g = cv2.blur(guide_f, ksize)
    mean_s = cv2.blur(src_f, ksize)
    mean_gs = cv2.blur(guide_f * src_f, ksize)
    mean_gg = cv2.blur(guide_f * guide_f, ksize)

    cov_gs = mean_gs - mean_g * mean_s
    var_g = mean_gg - mean_g * mean_g

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.blur(a, ksize)
    mean_b = cv2.blur(b, ksize)

    return mean_a * guide_f + mean_b


def refine_hair_mask(frame_bgr, raw_mask):
    """生え際を精密にするマスク精製パイプライン"""
    h, w = raw_mask.shape[:2]

    # 1. マスクをフレーム解像度にリサイズ（バイキュービック補間で高品質に）
    if raw_mask.shape[:2] != frame_bgr.shape[:2]:
        mask = cv2.resize(
            raw_mask, (w, h), interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)
    else:
        mask = raw_mask.astype(np.float32)

    # 正規化
    if mask.max() > 1.0:
        mask = mask / 255.0

    # 2. 低い閾値で二値化（取りこぼしを減らす）
    mask = np.where(mask > MASK_THRESHOLD, mask, 0.0)

    mask_u8 = (mask * 255).astype(np.uint8)

    # 膨張で髪領域を広げる（生え際の取りこぼし防止）
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_DILATE_SIZE, MORPH_DILATE_SIZE)
    )
    mask_u8 = cv2.dilate(mask_u8, kernel_dilate, iterations=1)

    # 小さな穴を埋める（close）
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)

    # 小さなノイズを除去（open）
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_ERODE_SIZE, MORPH_ERODE_SIZE)
    )
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open)

    # 3. ガイドフィルタを複数回適用（画像のエッジに沿ってマスクを整列）
    #    グレースケール画像をガイドにすることで、髪と肌の境界に忠実なマスクになる
    guide_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    refined = mask_u8.astype(np.float32) / 255.0

    for _ in range(REFINE_ITERATIONS):
        refined = guided_filter(guide_gray, refined, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPS)

    # 4. エッジ付近でさらに精密なトリミング
    #    Cannyエッジ検出で画像の輪郭を取得し、マスク境界をエッジに吸着
    edges = cv2.Canny(guide_gray, 50, 150)
    edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edge_band = edge_dilated.astype(np.float32) / 255.0

    # エッジ帯域ではマスクをよりシャープに（ガイドフィルタの結果を優先）
    # エッジ外ではよりスムーズに
    smooth = cv2.GaussianBlur(refined, (3, 3), 1)
    refined = refined * edge_band + smooth * (1 - edge_band)

    # 5. 値域をクリップして最終マスク
    refined = np.clip(refined, 0.0, 1.0)

    return refined


def make_hair_black(frame_bgr, hair_mask, alpha):
    """髪領域を自然な黒髪に変換（精密マスク + テクスチャ保持）"""
    if not np.any(hair_mask > 0):
        return frame_bgr

    # 精密マスク精製
    mask_refined = refine_hair_mask(frame_bgr, hair_mask)
    mask_3ch = np.stack([mask_refined] * 3, axis=-1)

    # HSVに変換：元のテクスチャ（明暗の濃淡）を保持しつつ暗くする
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    black_hsv = hsv.copy()
    # 彩度を大幅に落とす（でもゼロにはしない→微妙な色味が残りリアル）
    black_hsv[:, :, 1] = hsv[:, :, 1] * DESAT_FACTOR
    # 明度を比率で下げる→元の明暗差（ツヤ、影）がそのまま残る
    black_hsv[:, :, 2] = hsv[:, :, 2] * DARKENING_FACTOR

    black_bgr = cv2.cvtColor(
        np.clip(black_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
    )

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
    prev_mask = None  # 前フレームのマスク（時間平滑化用）

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
                    hair_mask_f = hair_mask.astype(np.float32)

                    # 時間平滑化：前フレームのマスクと混合してちらつき防止
                    if prev_mask is not None and prev_mask.shape == hair_mask_f.shape:
                        hair_mask_f = (
                            TEMPORAL_SMOOTH * prev_mask
                            + (1 - TEMPORAL_SMOOTH) * hair_mask_f
                        )
                        # 前フレームで検出されていた領域は簡単に消えないようにする
                        hair_mask_f = np.maximum(hair_mask_f, prev_mask * 0.5)
                    prev_mask = hair_mask_f.copy()

                    # 0-255にスケール
                    hair_mask_scaled = (hair_mask_f * 255).astype(np.uint8)
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
