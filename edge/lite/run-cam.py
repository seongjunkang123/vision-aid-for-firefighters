import cv2

def main():
    # For Raspberry Pi camera module
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Could not open camera.")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame.")
                break
            
            frame = cv2.flip(frame, 1)
            
            cv2.imshow('Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")

if __name__ == "__main__":
    main()