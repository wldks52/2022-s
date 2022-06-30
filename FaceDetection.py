import cv2

# Cascades 디렉토리의 haarcascade_frontalface_default.xml 파일을 Classifier로 사용
faceCascade = cv2.CascadeClassifier('C:/Users/openin/AppData/Local/Programs/Python/Python39/Lib/site-packages/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while True:
    ret, img = cap.read()
    img = cv2.flip(img,1) # -1 : 상하반전
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #흑백으로 변환
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video',img) # video라는 이름으로 출력

    k = cv2.waitKey(30) & 0xff  #30초 동안 대기, esc는 27 리턴, 0 : 키 입력시까지 대기
    if k == 27: # ESC를 누르면 종료
        break
cap.release()
cv2.destroyAllWindows()
