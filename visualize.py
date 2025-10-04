# visualize.py

import argparse
import itertools
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm import tqdm

# ===================================================================
# ✨ 視覺化工具 ✨
# ===================================================================

# 建立一個顏色循環器，讓每個 BBox 的顏色都不同，方便在擁擠時區分
COLORS = [
    "#FF3838",
    "#FF9D97",
    "#FF7566",
    "#FFA459",
    "#FFB243",
    "#FFD700",
    "#A8E4A0",
    "#32CD32",
    "#00A550",
    "#00BFFF",
    "#1E90FF",
    "#87CEEB",
    "#9370DB",
    "#C71585",
    "#FF1493",
]


def parse_prediction_string(pred_str: str) -> list[dict]:
    """解析 Kaggle 格式的預測字串。"""
    preds = []
    if pd.isna(pred_str) or not isinstance(pred_str, str) or pred_str.strip() == "":
        return preds

    parts = pred_str.strip().split()
    if len(parts) % 6 != 0:
        print(f"警告：預測字串格式不完整，將忽略結尾部分: {pred_str}")
        parts = parts[: (len(parts) // 6) * 6]

    for i in range(0, len(parts), 6):
        try:
            conf, x, y, w, h, cls_id = map(float, parts[i : i + 6])
            preds.append({"conf": conf, "x": x, "y": y, "w": w, "h": h, "cls": int(cls_id)})
        except (ValueError, IndexError):
            print(f"警告：無法解析預測字串片段: {' '.join(parts[i : i + 6])}")
            continue
    return preds


def draw_detections(
    image_path: Path, detections: list[dict], conf_threshold: float, class_names: dict, title: str
) -> plt.Figure:
    """在單張圖片上繪製所有偵測框。"""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"無法讀取圖片: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- 智慧調整畫布尺寸以符合圖片原始比例 ---
    dpi = 100
    height, width, _ = img_rgb.shape
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img_rgb)
    ax.axis("off")

    filtered_dets = [d for d in detections if d["conf"] >= conf_threshold]
    color_cycle = itertools.cycle(COLORS)

    for det in filtered_dets:
        x, y, w, h, cls_id, conf = det["x"], det["y"], det["w"], det["h"], det["cls"], det["conf"]
        color = next(color_cycle)

        rect = Rectangle((x, y), w, h, edgecolor=color, facecolor="none", linewidth=2.5)
        ax.add_patch(rect)

        label = f"{class_names.get(cls_id, f'Class {cls_id}')}: {conf:.2f}"
        ax.text(
            x, y - 5, label, fontsize=12, color="white", bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1)
        )

    ax.set_title(title, fontsize=14)
    plt.tight_layout(pad=0)
    return fig


def get_image_ids_to_visualize(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """根據命令行參數，決定要視覺化哪些圖片 ID。"""
    df["detection_count"] = df["PredictionString"].apply(lambda x: len(parse_prediction_string(x)))

    if args.fixed_ids:
        print(f"🔍 使用固定的 Image IDs: {args.fixed_ids}")
        return args.fixed_ids
    if args.random_k:
        print(f"🎲 隨機選取 {args.random_k} 張圖片...")
        return df["Image_ID"].sample(n=min(args.random_k, len(df)), random_state=args.seed).tolist()
    if args.top_k_most:
        print(f"📈 選取偵測框最多的 {args.top_k_most} 張圖片...")
        return df.nlargest(args.top_k_most, "detection_count")["Image_ID"].tolist()
    if args.top_k_least:
        df_with_dets = df[df["detection_count"] > 0]
        print(f"📉 選取偵測框最少 (但 > 0) 的 {args.top_k_least} 張圖片...")
        return df_with_dets.nsmallest(args.top_k_least, "detection_count")["Image_ID"].tolist()
    if args.no_detections:
        df_no_dets = df[df["detection_count"] == 0]
        print(f"👻 選取 {args.no_detections} 張完全沒有偵測結果的圖片...")
        return df_no_dets.sample(n=min(args.no_detections, len(df_no_dets)), random_state=args.seed)[
            "Image_ID"
        ].tolist()

    # 預設行為：使用一組固定的 ID
    print(" defaulting to fixed IDs")
    return [1000, 1859, 832, 664, 1, 817, 1360, 1258]


def main(args):
    """主執行函數"""
    csv_path = Path(args.csv_path)
    test_dir = Path(args.test_dir)

    # --- 1. 建立輸出資料夾 ---
    output_dir_local = Path(f"viz_{csv_path.stem}")
    output_dir_local.mkdir(exist_ok=True)
    print(f"✅ 設定完成，本地視覺化結果將儲存至: {output_dir_local.resolve()}")

    # --- 2. 讀取並準備資料 ---
    try:
        df = pd.read_csv(csv_path)
        df["Image_ID"] = df["Image_ID"].astype(str)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到 submission 檔案 '{csv_path}'。")
        return

    # --- 3. 篩選要處理的圖片 ---
    image_ids = get_image_ids_to_visualize(df, args)

    # --- 4. 迴圈處理每張圖片 ---
    print(f"\n🚀 開始分析 '{csv_path.name}'...")
    for image_id in tqdm(image_ids, desc="Visualizing"):
        image_id_str = str(image_id)
        row = df[df["Image_ID"] == image_id_str]

        if row.empty:
            print(f"⚠️ 警告：在 CSV 中找不到 Image_ID: {image_id_str}")
            continue

        pred_str = row["PredictionString"].iloc[0]
        preds = parse_prediction_string(pred_str)

        img_path = test_dir / f"{int(image_id_str):08d}.jpg"
        if not img_path.exists():
            print(f"⚠️ 警告：找不到圖片檔案: {img_path}")
            continue

        filtered_count = sum(1 for p in preds if p["conf"] >= args.conf_threshold)
        title = (
            f"Image ID: {image_id_str}\n"
            f"Source: {csv_path.name}\n"
            f"Detections (conf ≥ {args.conf_threshold}): {filtered_count}"
        )

        try:
            fig = draw_detections(img_path, preds, args.conf_threshold, {0: "pig"}, title)
            output_path = output_dir_local / f"{image_id_str}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"❌ 處理 Image ID {image_id_str} 時發生錯誤: {e}")
            plt.close("all")

    print(f"\n🎉 完成！本地視覺化結果已儲存於 '{output_dir_local}'。")

    # --- 5. 複製到 Google Drive (如果指定路徑) ---
    if args.gdrive_path:
        gdrive_project_path = Path(args.gdrive_path)
        if gdrive_project_path.is_dir():
            destination_path = gdrive_project_path / output_dir_local.name
            print("\n🚀 正在將結果複製到 Google Drive...")
            if destination_path.exists():
                shutil.rmtree(destination_path)
            shutil.copytree(output_dir_local, destination_path)
            print(f"✅ 成功！結果已複製到您的 Google Drive: {destination_path}")
        else:
            print(f"\n❌ 複製失敗：Google Drive 路徑不存在或不是資料夾: {gdrive_project_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="視覺化物件偵測結果")

    parser.add_argument("--csv_path", type=str, required=True, help="要分析的 submission.csv 檔案路徑")
    parser.add_argument("--test_dir", type=str, default="/content/data/test/img", help="測試圖片所在的資料夾")
    parser.add_argument("--gdrive_path", type=str, default=None, help="Google Drive 專案路徑，結果將複製到此處")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="視覺化的信心度門檻")
    parser.add_argument("--seed", type=int, default=42, help="用於隨機抽樣的種子")

    # --- 建立一個互斥的參數組，使用者只能選一種圖片篩選模式 ---
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fixed_ids", type=int, nargs="+", help="指定一組固定的圖片 ID 進行分析")
    group.add_argument("--random_k", type=int, help="隨機選取 K 張圖片進行分析")
    group.add_argument("--top_k_most", type=int, help="分析偵測框最多的 K 張圖片")
    group.add_argument("--top_k_least", type=int, help="分析偵測框最少 (但 > 0) 的 K 張圖片")
    group.add_argument("--no_detections", type=int, help="分析 K 張完全沒有偵測結果的圖片")

    args = main(parser.parse_args())
