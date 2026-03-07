import requests
import os
import time

server_url = 'http://127.0.0.1:5000'
workspace = r'c:\Users\abhin\Downloads\testing'
img_path = os.path.join(workspace, 'input_img.jpg')

def test_upgraded_models():
    print("--- Verifying High-Performance Model Upgrades ---")
    print("NOTE: This may take several minutes as models (~300MB+) are downloaded on first run.")
    
    # Increase timeout significantly for first-run model downloads + CPU inference
    TIMEOUT = 600 

    if os.path.exists(img_path):
        try:
            with open(img_path, 'rb') as f:
                files = {'image': f}
                print("Sending parallel analysis request (All 4 modules)...")
                start_time = time.time()
                r = requests.post(f"{server_url}/api/analyze-frame", files=files, timeout=TIMEOUT)
                end_time = time.time()
                
                if r.status_code == 200:
                    data = r.json()
                    print(f"[SUCCESS] Request completed in {end_time - start_time:.1f}s")
                    print(f"Results:")
                    print(f"   Vehicles (Large): {data.get('vehicles', {}).get('total', 0)}")
                    print(f"   ANPR (Large): {len(data.get('anpr', {}).get('plates', []))}")
                    print(f"   Faces (RetinaFace): {data.get('faces', {}).get('total_faces', 0)}")
                    print(f"   People (Large): {data.get('people', {}).get('total_people', 0)}")
                else:
                    print(f"[FAILED] Status: {r.status_code}, Body: {r.text}")
        except Exception as e:
            print(f"[ERROR]: {e}")
    else:
        print("[WARNING] input_img.jpg missing for verification.")

if __name__ == "__main__":
    test_upgraded_models()
