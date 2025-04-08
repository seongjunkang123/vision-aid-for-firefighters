import cv2
import time
import math

LENGTH = 256

def main():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LENGTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LENGTH)
    
    if not cap.isOpened():
        print("Could not open camera.")
        return
    
    count = 0
    
    try:
        while True:
            start = time.time() 

            ret, frame = cap.read() # read frame
            
            if not ret:
                print("Error: Could not read frame.")
                break
            
            end = time.time()
            fps = math.ceil(1/(end - start))
            count += 1

            if count == 1:
                average_fps = fps
            else:
                average_fps = math.ceil((average_fps * count + fps) / (count + 1))
            
            frame = cv2.flip(frame, 1) # mirror frame

            # print(f"fps: {fps} \r")
            # print(f"average fps: {average_fps}\r")

            
            cv2.putText(frame, f"fps: {str(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
            cv2.putText(frame, f"average fps: {str(average_fps)}", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

            cv2.imshow('Camera', frame) # show frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")

if __name__ == "__main__":
    main()