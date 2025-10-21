import cv2
import pytesseract
import re
from collections import Counter
import imutils 
from ultralytics import YOLO 
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import numpy as np
from PIL import Image

# --- Configuração Inicial ---

# 1. Caminho para o Tesseract-OCR
# !!! ATENÇÃO: Este caminho deve existir no computador ONDE o Streamlit está rodando !!!
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Tenta rodar uma vez para verificar se o Tesseract está OK
    pytesseract.get_tesseract_version()
except Exception as e:
    st.error(f"Erro ao localizar o Tesseract-OCR. Verifique se ele está instalado em 'C:\\Program Files\\Tesseract-OCR'.")
    st.error(f"Detalhe: {e}")
    st.stop()

# 2. Carregar o Detector YOLO (usando @st.cache_resource)
@st.cache_resource
def carregar_modelo_yolo():
    try:
        print("Carregando detector de placas (best.pt)...")
        model = YOLO('best.pt')
        print("Detector de placas (YOLO) carregado com sucesso.")
        return model
    except Exception as e:
        st.error(f"Erro: Não foi possível carregar o modelo 'best.pt' do YOLO: {e}")
        st.error("Certifique-se que o 'best.pt' (de 6.387 KB) está na pasta.")
        st.stop()

plate_detector = carregar_modelo_yolo()

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
QTD_FRAMES_PARA_RESET = 30
QTD_VOTOS_PARA_CONFIRMAR = 5
MAX_HISTORICO = 20
FRAME_WIDTH = 640       
FRAME_SKIP_OCR = 5      
YOLO_CONFIDENCE = 0.4   

# --- Funções Puras (não precisam de mudança) ---

def corrigir_placa(texto_ocr):
    """Tenta corrigir erros comuns de OCR (ex: D vs 0) usando a posição."""
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

def processar_ocr(plate_roi_gray):
    """Função de OCR otimizada."""
    try:
        scale = 3
        w, h = plate_roi_gray.shape[1], plate_roi_gray.shape[0]
        plate_roi_resized = cv2.resize(plate_roi_gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        plate_roi_blurred = cv2.GaussianBlur(plate_roi_resized, (5, 5), 0)
        plate_roi_contrast = cv2.equalizeHist(plate_roi_blurred) 
        _, plate_thresh = cv2.threshold(plate_roi_contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        text = pytesseract.image_to_string(plate_thresh, config=TESS_CONFIG)
        
        if text.strip():
            print(f"Texto Bruto do Tesseract: '{text.strip()}'")
            
        return corrigir_placa(text)
    except Exception:
        return None

# --- Classe do Streamlit para Processamento de Vídeo ---

class PlateTransformer(VideoTransformerBase):
    def __init__(self):
        # Inicializa o estado da sessão (substitui variáveis globais)
        if "historico" not in st.session_state:
            st.session_state.historico = []
        if "frames_sem" not in st.session_state:
            st.session_state.frames_sem = 0
        if "placa_estavel" not in st.session_state:
            st.session_state.placa_estavel = "Aguardando..."
        if "frame_count" not in st.session_state:
            st.session_state.frame_count = 0

    def recv(self, frame):
        # Converte o frame do Streamlit (av.VideoFrame) para OpenCV (numpy array)
        img = frame.to_ndarray(format="bgr24")
        
        # Lógica de processamento principal (do seu 'iniciar_camera')
        st.session_state.frame_count += 1
        
        frame_processado = imutils.resize(img, width=FRAME_WIDTH)
        gray = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2GRAY)
        
        rodar_ocr_neste_frame = (st.session_state.frame_count % FRAME_SKIP_OCR == 0)
        
        placa_lida_frame = None
        
        # Detecção YOLO
        plate_results = plate_detector(frame_processado, verbose=False)

        for result in plate_results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                
                if score > YOLO_CONFIDENCE:
                    x, y = int(x1), int(y1)
                    w, h = int(x2) - x, int(y2) - y
                    
                    # Desenha a caixa verde
                    cv2.rectangle(frame_processado, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    if rodar_ocr_neste_frame:
                        plate_roi_gray = gray[y:y+h, x:x+w]
                        if plate_roi_gray.size > 0:
                            placa_lida_frame = processar_ocr(plate_roi_gray)
                            if placa_lida_frame:
                                break # Para de procurar se achar uma
            if placa_lida_frame: break

        # --- Lógica de Estabilização (Votação) ---
        if placa_lida_frame: 
            st.session_state.historico.append(placa_lida_frame)
            st.session_state.frames_sem = 0
            if len(st.session_state.historico) > MAX_HISTORICO:
                st.session_state.historico.pop(0)
        elif rodar_ocr_neste_frame: 
            st.session_state.frames_sem += 1

        if st.session_state.frames_sem > (QTD_FRAMES_PARA_RESET / FRAME_SKIP_OCR): 
            st.session_state.historico.clear()
            st.session_state.placa_estavel = "Aguardando..."

        if st.session_state.historico:
            (placa_mais_comum, contagem) = Counter(st.session_state.historico).most_common(1)[0]
            if contagem >= QTD_VOTOS_PARA_CONFIRMAR:
                st.session_state.placa_estavel = placa_mais_comum

        # --- Desenha o resultado estável na tela ---
        cor_placa = (0, 255, 0) if st.session_state.placa_estavel != "Aguardando..." else (0, 255, 255)
        cv2.rectangle(frame_processado, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.putText(frame_processado, "Placa Lida:", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_processado, st.session_state.placa_estavel, (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_placa, 3)

        # Retorna o frame processado para o Streamlit
        return frame_processado

# --- Interface Principal do Streamlit ---

def main():
    st.set_page_config(page_title="Detector de Placas", page_icon="🚗")
    st.title("Detector de Placas em Tempo Real (YOLO + Tesseract)")
    st.write("Aponte a câmera do seu celular para uma placa. O sistema tentará ler.")

    # O componente que liga a câmera
    webrtc_streamer(
        key="detector",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=PlateTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Mostra o status na barra lateral
    st.sidebar.title("Status")
    st.sidebar.write("Placa Estável Atual:")
    
    # Um "placeholder" que atualiza o texto
    placa_placeholder = st.sidebar.empty()
    placa_placeholder.code(st.session_state.get("placa_estavel", "Aguardando..."))
    
    st.sidebar.write("Histórico de Votos:")
    st.sidebar.write(st.session_state.get("historico", []))
    
    # Hack para forçar a barra lateral a atualizar
    if "frame_count" in st.session_state and st.session_state.frame_count % FRAME_SKIP_OCR == 0:
        placa_placeholder.code(st.session_state.placa_estavel)


if __name__ == "__main__":
    main()