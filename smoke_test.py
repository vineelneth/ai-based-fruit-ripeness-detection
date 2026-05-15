"""
Production smoke test -- run after both Render and Vercel are deployed.

Usage:
  python smoke_test.py --backend https://your-app.onrender.com
  python smoke_test.py --backend https://your-app.onrender.com --frontend https://your-app.vercel.app
"""
import argparse
import sys
import io
import requests
from PIL import Image, ImageDraw


def make_synthetic_image() -> bytes:
    """Solid-color ellipse -- will be rejected by the fruit gate (correct behavior)."""
    img = Image.new("RGB", (200, 200), color=(255, 165, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 170, 170], fill=(220, 100, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_fruit_image() -> bytes:
    """
    Download a public-domain fruit photo for end-to-end prediction testing.
    Falls back to None if download fails.
    """
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Clementine_in_hand.jpg/320px-Clementine_in_hand.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/240px-Red_Apple.jpg",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
        except Exception:
            continue
    return None


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" -- {detail}"
    print(line)
    return condition


def test_backend(base: str):
    print(f"\n=== Backend: {base} ===")
    passed = []

    # Health
    try:
        r = requests.get(f"{base}/health", timeout=15)
        data = r.json()
        passed.append(check("Health status 200", r.status_code == 200))
        passed.append(check("model_loaded=true", data.get("model_loaded") is True, str(data)))
    except Exception as e:
        passed.append(check("Health reachable", False, str(e)))
        print("  Cannot reach backend -- skipping remaining tests")
        return False

    # Download fruit image once; reuse for predict + rate-limit tests
    fruit_bytes = make_fruit_image()
    if fruit_bytes:
        print(f"  [INFO] Downloaded real fruit image ({len(fruit_bytes):,} bytes)")
    else:
        print("  [WARN] Could not download fruit image -- skipping predict and rate-limit tests")

    # Predict -- real fruit image -> expect 200 + valid fields
    if fruit_bytes:
        try:
            r = requests.post(
                f"{base}/api/v1/predict",
                files={"file": ("fruit.jpg", fruit_bytes, "image/jpeg")},
                timeout=60,
            )
            data = r.json()
            passed.append(check("Predict status 200", r.status_code == 200, str(r.status_code)))
            passed.append(check("ripeness_pct in response", "ripeness_pct" in data))
            passed.append(check("ripeness_pct in 0-100", 0 <= data.get("ripeness_pct", -1) <= 100))
            passed.append(check("days_to_ripe >= 0", data.get("days_to_ripe", -1) >= 0))
            passed.append(check("status field present", "status" in data, data.get("status")))
            print(f"         prediction: {data}")
        except Exception as e:
            passed.append(check("Predict request", False, str(e)))

    # Gate -- non-fruit image -> expect 422 UNSUPPORTED_IMAGE
    try:
        synthetic = make_synthetic_image()
        r = requests.post(
            f"{base}/api/v1/predict",
            files={"file": ("synthetic.jpg", synthetic, "image/jpeg")},
            timeout=30,
        )
        data = r.json()
        is_422 = r.status_code == 422
        detail = data.get("detail", {})
        has_code = isinstance(detail, dict) and detail.get("error_code") == "UNSUPPORTED_IMAGE"
        passed.append(check("Gate rejects non-fruit -> 422", is_422, str(r.status_code)))
        passed.append(check("Gate returns UNSUPPORTED_IMAGE code", has_code, str(detail)))
        if is_422 and has_code:
            print(f"         gate message: {detail.get('message')}")
    except Exception as e:
        passed.append(check("Gate rejection test", False, str(e)))

    # Predict -- wrong content type -> expect 415
    try:
        r = requests.post(
            f"{base}/api/v1/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            timeout=10,
        )
        passed.append(check("Wrong MIME type -> 415", r.status_code == 415, str(r.status_code)))
    except Exception as e:
        passed.append(check("Wrong MIME type test", False, str(e)))

    # Rate limit -- 12 rapid requests with real fruit image -> expect at least one 429
    # Uses real fruit so requests actually reach the slowapi counter (gate-rejected
    # requests return 422 before the counter increments).
    if fruit_bytes:
        try:
            statuses = []
            for _ in range(12):
                r = requests.post(
                    f"{base}/api/v1/predict",
                    files={"file": ("t.jpg", fruit_bytes, "image/jpeg")},
                    timeout=60,
                )
                statuses.append(r.status_code)
            got_429 = 429 in statuses
            passed.append(check("Rate limit fires (429)", got_429, f"statuses: {statuses}"))
        except Exception as e:
            passed.append(check("Rate limit test", False, str(e)))

    return all(passed)


def test_frontend(base: str, backend: str):
    print(f"\n=== Frontend via Vercel rewrite: {base} ===")
    passed = []

    # Frontend loads
    try:
        r = requests.get(base, timeout=10)
        passed.append(check("Frontend loads (200)", r.status_code == 200))
        passed.append(check("Has drop-zone element", "drop-zone" in r.text))
        passed.append(check("Has FreshScan title", "FreshScan" in r.text))
    except Exception as e:
        passed.append(check("Frontend reachable", False, str(e)))
        return False

    # Vercel rewrite -- predict through frontend domain using real fruit image
    fruit_bytes = make_fruit_image()
    if fruit_bytes:
        try:
            r = requests.post(
                f"{base}/api/v1/predict",
                files={"file": ("test.jpg", fruit_bytes, "image/jpeg")},
                timeout=60,
            )
            passed.append(check("Vercel->Render rewrite works (200)", r.status_code == 200, str(r.status_code)))
            if r.status_code == 200:
                data = r.json()
                passed.append(check("Full end-to-end prediction", "ripeness_pct" in data, str(data)))
        except Exception as e:
            passed.append(check("Vercel rewrite test", False, str(e)))
    else:
        print("  [WARN] Skipping rewrite predict test -- no fruit image available")

    return all(passed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, help="Render backend URL, e.g. https://app.onrender.com")
    parser.add_argument("--frontend", help="Vercel frontend URL, e.g. https://app.vercel.app")
    args = parser.parse_args()

    backend_ok = test_backend(args.backend.rstrip("/"))
    frontend_ok = True
    if args.frontend:
        frontend_ok = test_frontend(args.frontend.rstrip("/"), args.backend.rstrip("/"))

    print("\n" + "=" * 50)
    print("Backend:", "PASS" if backend_ok else "FAIL")
    if args.frontend:
        print("Frontend:", "PASS" if frontend_ok else "FAIL")

    if not backend_ok or not frontend_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
