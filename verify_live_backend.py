import requests
import os

server_url = 'http://127.0.0.1:5000'
workspace = r'c:\Users\abhin\Downloads\testing'
img_path = os.path.join(workspace, 'input_img.jpg')
video_path = os.path.join(workspace, 'input.mp4')

def test_live_demo():
    print("--- Verifying Live Analysis Demo Backend ---")
    
    # 1. Test /stream page
    try:
        r = requests.get(f"{server_url}/stream", timeout=10)
        if r.status_code == 200 and 'Live Streaming Analysis' in r.text:
            print("[SUCCESS] /stream page is accessible and valid.")
        else:
            print(f"[FAILED] /stream page. Status: {r.status_code}")
    except Exception as e:
        print(f"[ERROR] /stream: {e}")

    # 2. Test /serve-video/input.mp4
    try:
        r = requests.get(f"{server_url}/serve-video/input.mp4", stream=True, timeout=10)
        if r.status_code == 200:
            print("[SUCCESS] /serve-video/input.mp4 is serving content.")
        else:
            print(f"[FAILED] /serve-video. Status: {r.status_code}")
    except Exception as e:
        print(f"[ERROR] /serve-video: {e}")

    # 3. Test /api/analyze-frame
    if os.path.exists(img_path):
        try:
            with open(img_path, 'rb') as f:
                files = {'image': f}
                r = requests.post(f"{server_url}/api/analyze-frame", files=files, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    print("[SUCCESS] /api/analyze-frame.")
                    print(f"   Vehicles detected: {data.get('vehicles', {}).get('total', 0)}")
                    print(f"   ANPR plates detected: {len(data.get('anpr', {}).get('plates', []))}")
                    print(f"   Faces detected: {data.get('faces', {}).get('total_faces', 0)}")
                    print(f"   People detected: {data.get('people', {}).get('total_people', 0)}")
                else:
                    print(f"[FAILED] /api/analyze-frame. Status: {r.status_code}, Body: {r.text}")
        except Exception as e:
            print(f"[ERROR] /api/analyze-frame: {e}")
    else:
        print("[WARNING] input_img.jpg not found for frame analysis test.")

if __name__ == "__main__":
    test_live_demo()
