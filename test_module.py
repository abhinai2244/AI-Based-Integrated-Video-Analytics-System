import cv2
import sys
import os

# Set paths
workspace = r'c:\Users\abhin\Downloads\testing'
img_path = os.path.join(workspace, 'input_img.jpg')
sys.path.insert(0, workspace)

try:
    from modules.vehicle_counter import detect_vehicles
    print('Module imported OK')
    
    img = cv2.imread(img_path)
    if img is None:
        print(f'FAILED to read image: {img_path}')
        sys.exit(1)
    
    print(f'Image shape: {img.shape}')
    _, buf = cv2.imencode('.jpg', img)
    img_bytes = buf.tobytes()
    print(f'Encoded image size: {len(img_bytes)} bytes')
    
    print('Starting detection...')
    result = detect_vehicles(img_bytes)
    print('Detection finished')
    
    print(f"Total vehicles: {result.get('total', 0)}")
    print(f"Counts: {result.get('counts', {})}")
    print(f"Detections count: {len(result.get('detections', []))}")
    print(f"Has annotated image: {len(result.get('annotated_image', '')) > 0}")
    print('TEST SUCCESS')
    
except Exception as e:
    import traceback
    print('TEST FAILED')
    traceback.print_exc()
    sys.exit(1)
