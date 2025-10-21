import cv2
import pytesseract
import re
from collections import Counter
import imutils # Nova biblioteca, instale com: pip install imutils

# --- Configuração Inicial ---

# 1. Caminho para o Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Carregar o classificador Haar Cascade
cascade_path = 'haarcascade_russian_plate_number.xml'
plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    print(f"Erro: Não foi possível carregar o classificador de placas em '{cascade_path}'")
    exit()

# 3. Configurações do Tesseract
TESS_CONFIG = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# 4. Filtro regex para limpar a placa (Padrão Mercosul/Antigo)
PLATE_REGEX = r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$|^[A-Z]{3}[0-9]{4}$'

# 5. Mapas de Correção
MAPA_LETRA_NUM = {
    'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 
    'Z': '2', 'S': '5', 'G': '6', 'B': '8'
}
MAPA_NUM_LETRA = {
    '0': 'D', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'
}

# --- Configurações de Otimização e Estabilização ---
historico_deteccoes = []
frames_sem_deteccao = 0
placa_estavel = "Aguardando..."

QTD_FRAMES_PARA_RESET = 30
QTD_VOTOS_PARA_CONFIRMAR = 5
MAX_HISTORICO = 20

FRAME_WIDTH = 640         # NOVO: Redimensiona a imagem para processar mais rápido
FRAME_SKIP_OCR = 5        # NOVO: Só roda o OCR a cada 5 frames


def corrigir_placa(texto_ocr):
    """Tenta corrigir erros comuns de OCR (ex: D vs 0) usando a posição."""
    
    texto_limpo = re.sub(r'[\W_]+', '', texto_ocr).upper()
    if len(texto_limpo) != 7:
        return None

    chars = list(texto_limpo)
    
    # Posições 0, 1, 2 (Sempre Letra)
    for i in [0, 1, 2]:
        if chars[i] in MAPA_NUM_LETRA:
            chars[i] = MAPA_NUM_LETRA[chars[i]]
    
    is_mercosul = 'A' <= chars[4] <= 'Z'
    is_antiga = '0' <= chars[4] <= '9'

    if is_mercosul:
        # LLLNLNN
        if chars[3] in MAPA_LETRA_NUM: chars[3] = MAPA_LETRA_NUM[chars[3]]
        if chars[4] in MAPA_NUM_LETRA: chars[4] = MAPA_NUM_LETRA[chars[4]]
        if chars[5] in MAPA_LETRA_NUM: chars[5] = MAPA_LETRA_NUM[chars[5]]
        if chars[6] in MAPA_LETRA_NUM: chars[6] = MAPA_LETRA_NUM[chars[6]]
    
    elif is_antiga:
        # LLLNNNN
        for i in [3, 4, 5, 6]:
            if chars[i] in MAPA_LETRA_NUM:
                chars[i] = MAPA_LETRA_NUM[chars[i]]
    else:
        return None

    placa_corrigida = "".join(chars)
    
    if re.match(PLATE_REGEX, placa_corrigida):
        return placa_corrigida
    
    return None


def processar_frame(frame, rodar_ocr: bool):
    """
    Detecta placas e, se 'rodar_ocr' for True, extrai o texto.
    """
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # MUDANÇA: 'minNeighbors' reduzido para 4.
    # Fica mais sensível para detectar placas apagadas (pode dar mais alarme falso)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 20))
    
    placa_lida_frame = None # Inicia como None
    
    for (x, y, w, h) in plates:
        # Desenhar o retângulo verde (feedback imediato)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # MUDANÇA: Só roda o OCR se for o frame correto
        if rodar_ocr:
            plate_roi = gray[y:y+h, x:x+w]
            
            # --- NOVO PRÉ-PROCESSAMENTO PARA PLACAS APAGADAS ---
            # 1. Aumenta o contraste
            plate_roi_contrast = cv2.equalizeHist(plate_roi) 
            
            # 2. Aplica o threshold (binarização) na imagem COM contraste
            _, plate_thresh = cv2.threshold(plate_roi_contrast, 0, 255, 
                                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # (Opcional: descomente para ver o que o OCR está lendo)
            # cv2.imshow("Pre-processamento OCR", plate_thresh) 
            # --- FIM DO NOVO PRÉ-PROCESSAMENTO ---
            
            try:
                # Usa a imagem processada (plate_thresh)
                text = pytesseract.image_to_string(plate_thresh, config=TESS_CONFIG)
                
                placa_corrigida = corrigir_placa(text)
                
                if placa_corrigida:
                    placa_lida_frame = placa_corrigida # Salva a placa lida
                    # Escreve a leitura "corrigida" (apenas no frame de OCR)
                    cv2.putText(frame, placa_corrigida, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Se já achou uma, pare de processar outras placas neste frame
                    break 

            except Exception:
                pass 

    # Retorna o frame (com os retângulos) e a placa lida (se o OCR rodou)
    return frame, placa_lida_frame


def iniciar_camera():
    """Inicia a webcam e aplica a lógica de estabilização."""
    
    global historico_deteccoes, frames_sem_deteccao, placa_estavel
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    print("Câmera ligada. Pressione 'q' para sair.")
    
    frame_count = 0 # NOVO: Contador de frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1 # NOVO
        
        # --- Otimização de Velocidade ---
        # NOVO: Redimensiona o frame para processar mais rápido
        frame = imutils.resize(frame, width=FRAME_WIDTH)
        
        # NOVO: Decide se este frame deve rodar o OCR
        rodar_ocr_neste_frame = (frame_count % FRAME_SKIP_OCR == 0)
        
        # 1. Processa o frame
        frame_processado, placa_lida_frame = processar_frame(frame, rodar_ocr_neste_frame)
        
        # --- 2. Lógica de Estabilização (Votação) ---
        
        # SÓ ATUALIZA O HISTÓRICO SE O OCR RODOU E ACHOU ALGO
        if placa_lida_frame: 
            historico_deteccoes.append(placa_lida_frame)
            frames_sem_deteccao = 0
            
            if len(historico_deteccoes) > MAX_HISTORICO:
                historico_deteccoes.pop(0)
        
        # Se o OCR *não* rodou, não contamos como "frame sem detecção"
        elif rodar_ocr_neste_frame: 
            frames_sem_deteccao += 1

        # 3. Resetar se não houver detecção por um tempo
        if frames_sem_deteccao > (QTD_FRAMES_PARA_RESET / FRAME_SKIP_OCR): # Ajuste no reset
            historico_deteccoes.clear()
            placa_estavel = "Aguardando..."

        # 4. Fazer a Votação
        if historico_deteccoes:
            (placa_mais_comum, contagem) = Counter(historico_deteccoes).most_common(1)[0]
            
            if contagem >= QTD_VOTOS_PARA_CONFIRMAR:
                placa_estavel = placa_mais_comum

        # --- 5. Lógica de Exibição (Marcar o valor) ---
        
        cv2.rectangle(frame_processado, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame_processado, "Placa Lida:", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # NOVO: Muda a cor da placa estável para feedback
        cor_placa = (0, 255, 0) # Verde se estável
        if placa_estavel == "Aguardando...":
            cor_placa = (0, 255, 255) # Amarelo se aguardando
        
        cv2.putText(frame_processado, placa_estavel, (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_placa, 3)
        
        # Exibir o resultado
        cv2.imshow('Detector de Placas - Pressione Q para sair', frame_processado)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Câmera desligada.")

# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    # --- NOVO: Instalação da biblioteca 'imutils' ---
    try:
        import imutils
    except ImportError:
        print("Instalando 'imutils'...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imutils"])
        
    iniciar_camera()