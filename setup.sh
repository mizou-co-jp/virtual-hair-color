#!/bin/bash
set -e

echo "=== 髪色変更 仮想カメラ セットアップ ==="

# Python仮想環境の作成
if [ ! -d "venv" ]; then
    echo "Python仮想環境を作成中..."
    python3 -m venv venv
fi

source venv/bin/activate

# パッケージインストール
echo "パッケージをインストール中..."
pip install -r requirements.txt

# モデルダウンロード
if [ ! -f "hair_segmenter.tflite" ]; then
    echo "髪セグメンテーションモデルをダウンロード中..."
    curl -o hair_segmenter.tflite \
        https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite
fi

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "【重要】OBS Studioが必要です:"
echo "  1. brew install --cask obs でインストール"
echo "  2. OBSを開いて「仮想カメラ開始」→「仮想カメラ停止」を1回実行（初回のみ）"
echo "  3. OBSを閉じる"
echo ""
echo "実行方法:"
echo "  source venv/bin/activate"
echo "  python hair_to_black.py"
