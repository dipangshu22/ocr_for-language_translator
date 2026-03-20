from flask import Flask, request, jsonify
import os
import traceback
import uuid
import cv2
import numpy as np
import easyocr
import torch
from spellchecker import SpellChecker
from difflib import SequenceMatcher

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Startup ──────────────────────────────────────────────────────────────────

print("🖥 CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🖥 GPU:", torch.cuda.get_device_name(0))

USE_GPU = torch.cuda.is_available()

reader_standard = easyocr.Reader(
    ['en'],
    gpu=USE_GPU,
    model_storage_directory="models"
)
reader_beam = easyocr.Reader(
    ['en'],
    gpu=USE_GPU,
    model_storage_directory="models",
    recog_network='english_g2'
)

spell = SpellChecker()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Super Resolution ─────────────────────────────────────────────────────────

def super_resolve(img: np.ndarray) -> np.ndarray:
    h, w   = img.shape[:2]
    target = 1200
    if max(h, w) >= target:
        return img
    result = img.copy()
    while max(result.shape[:2]) < target:
        result = cv2.resize(
            result, None, fx=1.5, fy=1.5,
            interpolation=cv2.INTER_LANCZOS4
        )
        result = unsharp_mask(
            result if len(result.shape) == 2
            else cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            if len(result.shape) == 3 else result
        )
    print(f"  🔬 Super resolved: {w}x{h} → {result.shape[1]}x{result.shape[0]}")
    return result


# ─── Perspective Correction ───────────────────────────────────────────────────

def correct_perspective(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
        if len(img.shape) == 3 else img.copy()
    edges     = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (img.shape[0] * img.shape[1] * 0.1):
        return img
    angle = cv2.minAreaRect(largest)[-1]
    if abs(angle) < 1 or abs(angle) > 45:
        return img
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    print(f"  📐 Perspective corrected: {angle:.2f}°")
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# ─── Image Helpers ────────────────────────────────────────────────────────────

def unsharp_mask(image: np.ndarray,
                 sigma: float = 1.0,
                 strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)


def remove_noise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(
        image, h=10, templateWindowSize=7, searchWindowSize=21
    )


def add_border(image: np.ndarray, size: int = 40) -> np.ndarray:
    return cv2.copyMakeBorder(
        image, size, size, size, size,
        cv2.BORDER_CONSTANT, value=255
    )


def smart_resize(img: np.ndarray) -> np.ndarray:
    h, w          = img.shape[:2]
    target_height = max(128, 64 * max(1, round(h / 30)))
    scale         = target_height / h
    return cv2.resize(
        img, (int(w * scale), target_height),
        interpolation=cv2.INTER_LANCZOS4
    )


def deskew(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(img: np.ndarray) -> list:
    versions = []
    h, w     = img.shape[:2]
    print(f"  📐 Input: {w}x{h}")

    corrected = correct_perspective(img)
    resized   = smart_resize(corrected)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) \
        if len(resized.shape) == 3 else resized
    gray = super_resolve(gray)

    print(f"  📐 After pipeline: {gray.shape[1]}x{gray.shape[0]}")

    # V1: Raw
    versions.append(("raw",         add_border(gray)))

    # V2: Denoised
    versions.append(("denoised",    add_border(remove_noise(gray))))

    # V3: Unsharp mask
    versions.append(("unsharp",     add_border(unsharp_mask(gray))))

    # V4: CLAHE + Otsu
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(remove_noise(gray))
    _, v4    = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    versions.append(("clahe_otsu",  add_border(deskew(v4))))

    # V5: Adaptive threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    v5      = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )
    versions.append(("adaptive",    add_border(deskew(v5))))

    # V6: Inverted Otsu
    _, v6 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    versions.append(("inverted",    add_border(deskew(v6))))

    # V7: Bilateral + unsharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    versions.append(("bilateral",   add_border(unsharp_mask(bilateral))))

    # V8: Morphological closing
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    v8     = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    versions.append(("morphological", add_border(deskew(v8))))

    return versions


# ─── Spell Correction ─────────────────────────────────────────────────────────

def correct_word(word: str) -> str:
    if (word.isdigit() or len(word) <= 2
            or word.isupper() or not word.isalpha()):
        return word
    corrected = spell.correction(word)
    if corrected is None:
        return word
    sim = SequenceMatcher(None, word.lower(), corrected.lower()).ratio()
    if sim >= 0.8:
        return corrected.capitalize() if word[0].isupper() else corrected
    return word


def correct_text(text: str) -> str:
    return "\n".join(
        " ".join(correct_word(w) for w in line.split())
        for line in text.split("\n")
    )


# ─── Voting ───────────────────────────────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def vote_best_text(all_results: list) -> str:
    if not all_results:
        return ""

    candidates = []
    for result_list, label in all_results:
        if not result_list:
            continue
        text       = group_lines(result_list)
        avg_conf   = sum(r[2] for r in result_list) / len(result_list)
        char_count = sum(len(r[1]) for r in result_list)
        score      = avg_conf * (1 + 0.05 * char_count)
        candidates.append((text, score, avg_conf, label))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[1], reverse=True)

    for text, score, conf, label in candidates:
        print(f"  🗳 {label}: score={score:.3f}, conf={conf:.3f}, "
              f"text='{text[:50]}'")

    top = candidates[:3]
    if len(top) == 1:
        return top[0][0]

    top_words  = [t[0].split() for t in top]
    top_scores = [t[1] for t in top]
    base       = max(top_words, key=len)
    if not base:
        return top[0][0]

    final_words = []
    for i, base_word in enumerate(base):
        best_word       = base_word
        best_word_score = 0
        for words, score in zip(top_words, top_scores):
            if i < len(words):
                sim        = similarity(base_word, words[i])
                word_score = score * sim
                if word_score > best_word_score:
                    best_word_score = word_score
                    best_word       = words[i]
        final_words.append(best_word)

    return " ".join(final_words)


# ─── Line Grouping ────────────────────────────────────────────────────────────

def group_lines(result: list, line_threshold: int = 20) -> str:
    if not result:
        return ""
    result       = sorted(result, key=lambda x: x[0][0][1])
    lines        = []
    current_line = [result[0]]
    current_y    = result[0][0][0][1]
    for item in result[1:]:
        item_y = item[0][0][1]
        if abs(item_y - current_y) <= line_threshold:
            current_line.append(item)
        else:
            current_line.sort(key=lambda x: x[0][0][0])
            lines.append(" ".join(w[1] for w in current_line))
            current_line = [item]
            current_y    = item_y
    current_line.sort(key=lambda x: x[0][0][0])
    lines.append(" ".join(w[1] for w in current_line))
    return "\n".join(lines)


# ─── OCR Pipeline ─────────────────────────────────────────────────────────────

OCR_PARAMS_AGGRESSIVE = {
    "paragraph":      False,
    "contrast_ths":   0.05,
    "adjust_contrast": 0.7,
    "text_threshold": 0.4,
    "low_text":       0.25,
    "link_threshold": 0.2,
    "mag_ratio":      2.5,
    "slope_ths":      0.3,
    "ycenter_ths":    0.7,
    "height_ths":     0.7,
    "width_ths":      0.7,
    "add_margin":     0.25,
}

OCR_PARAMS_CONSERVATIVE = {
    "paragraph":      False,
    "contrast_ths":   0.1,
    "adjust_contrast": 0.5,
    "text_threshold": 0.6,
    "low_text":       0.3,
    "link_threshold": 0.3,
    "mag_ratio":      1.5,
    "add_margin":     0.1,
}


def run_ocr(image: np.ndarray, reader, params: dict) -> list:
    raw = reader.readtext(image, detail=1, **params)
    return [r for r in raw if r[2] >= 0.15]


def full_ocr_pipeline(img: np.ndarray) -> tuple:
    """
    Full pipeline — no debug images saved.
    1. Preprocess into 8 versions
    2. Run 2 readers × 2 param sets on each
    3. Vote best text word-by-word
    4. Spell correct
    """
    versions    = preprocess_image(img)
    all_results = []

    for i, (name, version) in enumerate(versions):
        # ✅ No cv2.imwrite — debug images completely disabled

        attempts = [
            (reader_standard, OCR_PARAMS_AGGRESSIVE,   f"v{i+1}_{name}_std_aggr"),
            (reader_standard, OCR_PARAMS_CONSERVATIVE,  f"v{i+1}_{name}_std_cons"),
            (reader_beam,     OCR_PARAMS_AGGRESSIVE,    f"v{i+1}_{name}_beam_aggr"),
        ]

        for reader_obj, params, label in attempts:
            try:
                filtered = run_ocr(version, reader_obj, params)
                if not filtered:
                    continue
                avg_conf = sum(r[2] for r in filtered) / len(filtered)
                print(f"  {label}: {len(filtered)} regions, conf={avg_conf:.3f}")
                for r in filtered:
                    print(f"    → '{r[1]}' ({r[2]:.2f})")
                all_results.append((filtered, label))
            except Exception as e:
                print(f"  {label} failed: {e}")
                continue

    voted_text = vote_best_text(all_results)

    if voted_text:
        corrected = correct_text(voted_text)
        if corrected != voted_text:
            print(f"  ✏️ Spell corrected: '{voted_text}' → '{corrected}'")
        voted_text = corrected

    best_result = max(
        all_results,
        key=lambda x: sum(r[2] for r in x[0]) / len(x[0]) if x[0] else 0
    )[0] if all_results else []

    avg_conf = (
        round(sum(r[2] for r in best_result) / len(best_result), 3)
        if best_result else 0
    )

    return best_result, voted_text, avg_conf


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return "OCR API is running"


@app.route("/ocr", methods=["GET", "POST"])
def ocr():
    if request.method == "GET":
        return "OCR endpoint is alive (use POST with an image file)"

    filepath = None

    try:
        if "image" not in request.files:
            return jsonify({"success": False,
                            "error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"success": False,
                            "error": "Empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Unsupported type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        ext           = file.filename.rsplit(".", 1)[1].lower()
        safe_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath      = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)

        print(f"\n📥 Received: {file.filename}")

        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"success": False,
                            "error": "Could not decode image"}), 400

        best_result, text, avg_conf = full_ocr_pipeline(img)

        print(f"\n🧠 Final: '{text}' (conf={avg_conf})")

        if not text.strip():
            return jsonify({
                "success": False,
                "error": "No text detected — select a larger or clearer area"
            }), 200

        return jsonify({
            "success":        True,
            "text":           text.strip(),
            "region_count":   len(best_result),
            "avg_confidence": avg_conf,
        })

    except Exception as e:
        print("❌ ERROR:\n", traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            print(f"🗑 Cleaned up: {filepath}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)