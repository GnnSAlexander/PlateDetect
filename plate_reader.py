import cv2
import pytesseract

# Cargar la imagen
image = cv2.imread('license_plate.jpg')

# Preprocesar la imagen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# Encontrar contornos y localizar la placa (simplificado)
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plate_contour = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
    if len(approx) == 4:
        plate_contour = approx
        break

if plate_contour is not None:
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate = gray[y:y+h, x:x+w]
    
    # Aplicar OCR
    text = pytesseract.image_to_string(plate, config='--psm 11')
    print("Número de placa:", text.strip())
else:
    print("No se pudo detectar la placa.")