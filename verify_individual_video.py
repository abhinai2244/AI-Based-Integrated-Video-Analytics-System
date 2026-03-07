import requests
import os
import sys

# Set paths
workspace = r'c:\Users\abhin\Downloads\testing'
video_path = os.path.join(workspace, 'input.mp4')
server_url = 'http://127.0.0.1:5000'

def test_endpoint(endpoint, file_key, file_path):
    print(f"\nTesting {endpoint} with {os.path.basename(file_path)}...")
    files = {file_key: open(file_path, 'rb')}
    try:
        r = requests.post(f"{server_url}{endpoint}", files=files, timeout=60)
        if r.status_code == 200:
            data = r.json()
            print(f"  SUCCESS: Status {r.status_code}")
            if 'total' in data: print(f"  Total Vehicles: {data['total']}")
            if 'total_plates' in data: print(f"  Total Plates: {data['total_plates']}")
            if 'total_faces' in data: print(f"  Total Faces: {data['total_faces']}")
            if 'total_people' in data: print(f"  Total People: {data['total_people']}")
            return True
        else:
            print(f"  FAILED: Status {r.status_code}")
            print(f"  Response: {r.text}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        files[file_key].close()

if __name__ == "__main__":
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        sys.exit(1)
        
    endpoints = [
        ('/api/detect-vehicles', 'file'),
        ('/api/anpr', 'file'),
        # ('/api/recognize-face', 'file'), # Skip face as it's slow/needs GPU/large model docs
        ('/api/count-people', 'file')
    ]
    
    overall_success = True
    for endpoint, key in endpoints:
        if not test_endpoint(endpoint, key, video_path):
            overall_success = False
            
    if overall_success:
        print("\nALL VERIFIED ENDPOINTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
