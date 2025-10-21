import cv2
import pytesseract
import re
from collections import Counter
import imutils
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Configuração Inicial ---

# 1. Caminho para o Tesseract-OCR
# !!! ATENÇÃO: Verifique se este caminho está correto no seu sistema !!!
try:
    # A linha abaixo foi REMOVIDA pois só funciona no Windows local.
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Esta verificação agora vai procurar o Tesseract instalado pelo packages.txt
    pytesseract.get_tesseract_version()
    print("Tesseract-OCR localizado com sucesso.")
except Exception as e:
    st.error(f"Erro ao localizar o Tesseract-OCR. O Tesseract-OCR não está instalado ou não está no PATH do sistema.")
    st.error(f"Se estiver fazendo deploy no Streamlit Cloud, verifique se 'tesseract-ocr' está no seu arquivo 'packages.txt'.")
    st.error(f"Detalhe do Erro: {e}")
    st.stop()
# 2. Carregar o Detector Haar Cascade (substituindo o YOLO)
@st.cache_resource
def carregar_cascade():
    cascade_path = 'haarcascade_russian_plate_number.xml'
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    if plate_cascade.empty():
        st.error(f"Erro: Não foi possível carregar o classificador Haar Cascade em '{cascade_path}'")
        st.error("Certifique-se de que o arquivo .xml está na mesma pasta que o script.")
        st.stop()
    print("Detector Haar Cascade carregado com sucesso.")
    return plate_cascade

plate_cascade = carregar_cascade()

# 3. Configurações do Tesseract
TESS_CONFIG = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# 4. Filtro regex para limpar a placa (Padrão Mercosul/Antigo)
PLATE_REGEX = r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$|^[A-Z]{3}[0-9]{4}$'

# 5. Mapas de Correção de caracteres
MAPA_LETRA_NUM = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
MAPA_NUM_LETRA = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}

# --- Configurações de Otimização e Estabilização ---
QTD_FRAMES_PARA_RESET = 30
QTD_VOTOS_PARA_CONFIRMAR = 5
MAX_HISTORICO = 20
FRAME_WIDTH = 640
FRAME_SKIP_OCR = 5

# --- Funções de Processamento (Puras) ---

def corrigir_placa(texto_ocr):
    """Tenta corrigir erros comuns de OCR usando a posição dos caracteres."""
    texto_limpo = re.sub(r'[\W_]+', '', texto_ocr).upper()
    if len(texto_limpo) != 7:
        return None
    
    chars = list(texto_limpo)
    
    # Posições 0, 1, 2 devem ser letras
    for i in [0, 1, 2]:
        if chars[i] in MAPA_NUM_LETRA:
            chars[i] = MAPA_NUM_LETRA[chars[i]]
            
    is_mercosul = 'A' <= chars[4] <= 'Z'
    is_antiga = '0' <= chars[4] <= '9'
    
    if is_mercosul: # Padrão LLLNLNN
        if chars[3] in MAPA_LETRA_NUM: chars[3] = MAPA_LETRA_NUM[chars[3]]
        if chars[4] in MAPA_NUM_LETRA: chars[4] = MAPA_NUM_LETRA[chars[4]]
        if chars[5] in MAPA_LETRA_NUM: chars[5] = MAPA_LETRA_NUM[chars[5]]
        if chars[6] in MAPA_LETRA_NUM: chars[6] = MAPA_LETRA_NUM[chars[6]]
    elif is_antiga: # Padrão LLLNNNN
        for i in [3, 4, 5, 6]:
            if chars[i] in MAPA_LETRA_NUM:
                chars[i] = MAPA_LETRA_NUM[chars[i]]
    else:
        return None # Formato desconhecido
        
    placa_corrigida = "".join(chars)
    if re.match(PLATE_REGEX, placa_corrigida):
        return placa_corrigida
        
    return None

def processar_ocr(plate_roi_gray):
    """Aplica pré-processamento e extrai texto da imagem da placa."""
    try:
        # Aumenta a escala para melhorar a leitura
        scale = 3
        w, h = plate_roi_gray.shape[1], plate_roi_gray.shape[0]
        plate_roi_resized = cv2.resize(plate_roi_gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        
        # Aumenta o contraste
        plate_roi_contrast = cv2.equalizeHist(plate_roi_resized)
        
        # Binarização (preto e branco)
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
        # Inicializa o estado da sessão para guardar as variáveis
        if "historico" not in st.session_state:
            st.session_state.historico = []
        if "frames_sem" not in st.session_state:
            st.session_state.frames_sem = 0
        if "placa_estavel" not in st.session_state:
            st.session_state.placa_estavel = "Aguardando..."
        if "frame_count" not in st.session_state:
            st.session_state.frame_count = 0

    def recv(self, frame):
        # Converte o frame para o formato do OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        st.session_state.frame_count += 1
        
        frame_processado = imutils.resize(img, width=FRAME_WIDTH)
        gray = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2GRAY)
        
        rodar_ocr_neste_frame = (st.session_state.frame_count % FRAME_SKIP_OCR == 0)
        placa_lida_frame = None
        
        # --- DETECÇÃO COM HAAR CASCADE (MAIS SIMPLES) ---
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 20))
        
        for (x, y, w, h) in plates:
            # Desenha a caixa verde de detecção
            cv2.rectangle(frame_processado, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if rodar_ocr_neste_frame:
                plate_roi_gray = gray[y:y+h, x:x+w]
                if plate_roi_gray.size > 0:
                    placa_lida_frame = processar_ocr(plate_roi_gray)
                    if placa_lida_frame:
                        break # Para de procurar se já achou uma placa válida

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
        cv2.putText(frame_processado, "Placa Lida:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_processado, st.session_state.placa_estavel, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_placa, 3)

        return frame_processado

# --- Interface Principal do Streamlit ---

def main():
    st.set_page_config(page_title="Detector de Placas (Simples)", page_icon="🚗")
    st.title("Detector de Placas em Tempo Real")
    st.write("Versão Simplificada (Haar Cascade + Tesseract)")
    st.write("Aponte a câmera do seu celular para uma placa.")

    webrtc_streamer(
        key="detector-simples",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=PlateTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Mostra o status na barra lateral
    st.sidebar.title("Status")
    st.sidebar.write("Placa Estável:")
    placa_placeholder = st.sidebar.empty()
    placa_placeholder.code(st.session_state.get("placa_estavel", "Aguardando..."))
    
    st.sidebar.write("Histórico de Leituras:")
    st.sidebar.write(st.session_state.get("historico", []))
    
    # Hack para forçar a barra lateral a atualizar
    if "frame_count" in st.session_state and st.session_state.frame_count % FRAME_SKIP_OCR == 0:
        placa_placeholder.code(st.session_state.placa_estavel)

if __name__ == "__main__":
    main()

