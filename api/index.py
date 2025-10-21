import cv2
import re
import imutils
import numpy as np
import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
from collections import Counter
import os # Para verificar o caminho do XML

# --- Configuração da API ---
app = FastAPI()

# --- Configuração do Detector de Placas ---

# O 'vercel.json' garante que este arquivo estará disponível
cascade_path = 'haarcascade_russian_plate_number.xml'

# Verificação de segurança
if not os.path.exists(cascade_path):
    print(f"Erro: Não foi possível encontrar '{cascade_path}'.")
    # Em um app real, isso impediria o boot.
    
plate_cascade = cv2.CascadeClassifier(cascade_path)
if plate_cascade.empty():
    print(f"Erro: Não foi possível carregar o classificador de placas em '{cascade_path}'")

# --- Configuração do EasyOCR ---
# Carrega o modelo de OCR em português. Isso pode levar alguns segundos no "cold start".
# gpu=False é essencial para rodar na Vercel (que não tem GPU).
try:
    reader = easyocr.Reader(['pt'], gpu=False)
except Exception as e:
    print(f"Erro ao carregar EasyOCR: {e}")
    reader = None

# --- Suas Funções de Correção (Mantidas) ---
PLATE_REGEX = r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$|^[A-Z]{3}[0-9]{4}$'
MAPA_LETRA_NUM = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
MAPA_NUM_LETRA = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
FRAME_WIDTH = 640

def corrigir_placa(texto_ocr):
    texto_limpo = re.sub(r'[\W_]+', '', texto_ocr).upper()
    if len(texto_limpo) != 7:
        return None

    chars = list(texto_limpo)
    
    for i in [0, 1, 2]:
        if chars[i] in MAPA_NUM_LETRA:
            chars[i] = MAPA_NUM_LETRA[chars[i]]
    
    is_mercosul = 'A' <= chars[4] <= 'Z'
    is_antiga = '0' <= chars[4] <= '9'

    if is_mercosul:
        if chars[3] in MAPA_LETRA_NUM: chars[3] = MAPA_LETRA_NUM[chars[3]]
        if chars[4] in MAPA_NUM_LETRA: chars[4] = MAPA_NUM_LETRA[chars[4]]
        if chars[5] in MAPA_LETRA_NUM: chars[5] = MAPA_LETRA_NUM[chars[5]]
        if chars[6] in MAPA_LETRA_NUM: chars[6] = MAPA_LETRA_NUM[chars[6]]
    
    elif is_antiga:
        for i in [3, 4, 5, 6]:
            if chars[i] in MAPA_LETRA_NUM:
                chars[i] = MAPA_LETRA_NUM[chars[i]]
    else:
        return None

    placa_corrigida = "".join(chars)
    if re.match(PLATE_REGEX, placa_corrigida):
        return placa_corrigida
    
    return None

# --- Função Principal de Processamento da API ---
def processar_imagem(frame):
    """Detecta placas e extrai o texto de UM ÚNICO frame."""
    
    if plate_cascade.empty():
        raise HTTPException(status_code=500, detail="Classificador Haar Cascade não carregado.")
        
    if reader is None:
        raise HTTPException(status_code=500, detail="Modelo EasyOCR não carregado.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 20))
    
    placas_encontradas = []

    for (x, y, w, h) in plates:
        plate_roi = gray[y:y+h, x:x+w]
        
        # --- Pré-processamento (Opcional, mas recomendado) ---
        plate_roi_contrast = cv2.equalizeHist(plate_roi)
        _, plate_thresh = cv2.threshold(plate_roi_contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        try:
            # Roda o EasyOCR na imagem processada
            # 'detail=0' retorna apenas o texto, 'paragraph=True' tenta juntar
            results = reader.readtext(plate_thresh, detail=0, paragraph=True)
            
            for text in results:
                placa_corrigida = corrigir_placa(text)
                if placa_corrigida:
                    placas_encontradas.append(placa_corrigida)
                    
        except Exception as e:
            print(f"Erro no OCR: {e}")
            pass 

    if placas_encontradas:
        # Lógica de votação simples: retorna a placa mais comum
        (placa_mais_comum, _) = Counter(placas_encontradas).most_common(1)[0]
        return placa_mais_comum
        
    return None

# --- Endpoints da API ---

@app.get("/api")
def root():
    return {"message": "API de Detecção de Placas. Envie um POST para /api/detectar"}

@app.post("/api/detectar")
async def detectar_placa(file: UploadFile = File(...)):
    """
    Endpoint principal. Recebe um arquivo de imagem (upload)
    e retorna a placa detectada.
    """
    try:
        # 1. Ler os bytes da imagem
        image_bytes = await file.read()
        
        # 2. Converter bytes para uma imagem OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Arquivo de imagem inválido.")
            
        # 3. Redimensionar (como no seu script original)
        frame = imutils.resize(frame, width=FRAME_WIDTH)
        
        # 4. Processar a imagem
        placa_lida = processar_imagem(frame)
        
        # 5. Retornar o resultado
        if placa_lida:
            return {"placa_detectada": placa_lida}
        else:
            return {"placa_detectada": None, "message": "Nenhuma placa válida encontrada."}

    except Exception as e:
        # Captura erros gerais
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")