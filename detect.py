import cv2
from ultralytics import YOLO
import numpy as np
import time # Para calcular o FPS

# --- Configurações Globais e Variáveis ---
SOURCE_IS_VIDEO = True # Mude para False se quiser testar com uma imagem estática como antes
VIDEO_SOURCE = 0 # 0 para webcam, ou 'caminho/para/seu/video.mp4'
# VIDEO_SOURCE = 'seu_video.mp4' # Exemplo para ficheiro de vídeo
IMAGE_SOURCE = 'minha_imagem.jpg' # Usado se SOURCE_IS_VIDEO for False

CONFIDENCE_THRESHOLD = 0.4 # Limiar de confiança mínimo para considerar uma deteção

# --- Variáveis de Interface (resetadas por frame no caso de vídeo) ---
img_display_processed = None # Imagem com as deteções para exibir
detections_this_frame = [] # Lista para armazenar informações das deteções do frame atual
selected_box_index = -1

COLOR_DEFAULT = (0, 255, 0) # Verde
COLOR_SELECTED = (0, 0, 255) # Vermelho
TEXT_COLOR_BOX = (255, 255, 255) # Branco para texto na caixa
TEXT_COLOR_INFO = (0, 0, 0) # Preto para informações no canto
INFO_BG_COLOR = (200, 200, 200, 180) # Cinza claro semi-transparente
FPS_COLOR = (0, 0, 255) # Vermelho para o texto do FPS

# --- Função para Desenhar a Interface (Caixas, Informações, FPS) ---
def draw_interface_video(original_frame):
    global img_display_processed, detections_this_frame, selected_box_index

    img_display_processed = original_frame.copy()

    # Desenha todas as caixas delimitadoras válidas do frame atual
    for i, det in enumerate(detections_this_frame):
        xyxy = det['xyxy']
        label = det['label']
        conf = det['confidence']
        color = COLOR_SELECTED if i == selected_box_index else COLOR_DEFAULT

        cv2.rectangle(img_display_processed, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
        text_on_box = f"{label}: {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text_on_box, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Fonte menor
        cv2.rectangle(img_display_processed, (int(xyxy[0]), int(xyxy[1]) - text_height - baseline), (int(xyxy[0]) + text_width, int(xyxy[1])), color, -1)
        cv2.putText(img_display_processed, text_on_box, (int(xyxy[0]), int(xyxy[1]) - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR_BOX, 1, cv2.LINE_AA)

    # Desenha informações do item selecionado
    info_text_lines = []
    if selected_box_index != -1 and selected_box_index < len(detections_this_frame):
        selected_det = detections_this_frame[selected_box_index]
        info_text_lines.append(f"Selecionado: {selected_det['label']}")
        info_text_lines.append(f"Confianca: {selected_det['confidence']:.2f}")
        # info_text_lines.append(f"Coords: {[int(c) for c in selected_det['xyxy']]}") # Opcional, pode poluir
    else:
        info_text_lines.append("Nenhum item selecionado")

    # Desenha o fundo para as informações
    info_y_start = 10
    info_x_start = 10
    info_line_height = 25
    # Altura do fundo baseada no número de linhas + um pouco de padding
    bg_height = info_y_start + (len(info_text_lines) * info_line_height)
    # Largura do fundo (ajuste conforme necessário)
    cv2.rectangle(img_display_processed, (info_x_start -5 , info_y_start -5 ), (350, bg_height), INFO_BG_COLOR, -1)

    for i, line in enumerate(info_text_lines):
        y_pos = info_y_start + (i * info_line_height) + (info_line_height // 2) # Centraliza verticalmente
        cv2.putText(img_display_processed, line, (info_x_start, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR_INFO, 1, cv2.LINE_AA)


    cv2.imshow("Deteccoes YOLOv8 - Video Interativo", img_display_processed)

# --- Função de Callback do Rato ---
def mouse_callback_video(event, x, y, flags, param):
    global selected_box_index, detections_this_frame, img_display_processed # Precisa de img_display_processed para redesenhar
    # Passamos o frame original como param para redesenhar corretamente
    original_frame_for_redraw = param

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_on_box_this_time = False
        # Verifica se há deteções neste frame para evitar erro se detections_this_frame estiver vazio
        if detections_this_frame:
            for i, det in enumerate(detections_this_frame):
                xmin, ymin, xmax, ymax = det['xyxy']
                if xmin < x < xmax and ymin < y < ymax:
                    selected_box_index = i
                    print_selected_info_video()
                    clicked_on_box_this_time = True
                    break
        if not clicked_on_box_this_time:
            selected_box_index = -1
            print_selected_info_video()
        # Importante: redesenha a interface COM O FRAME ORIGINAL para não acumular desenhos
        if original_frame_for_redraw is not None:
            draw_interface_video(original_frame_for_redraw)


# --- Função para imprimir informações do selecionado no console ---
def print_selected_info_video():
    global selected_box_index, detections_this_frame
    if selected_box_index != -1 and selected_box_index < len(detections_this_frame):
        det = detections_this_frame[selected_box_index]
        print(f"\n--- Item Destacado (Frame Atual, Índice: {selected_box_index}) ---")
        print(f"  Classe: {det['label']}")
        print(f"  Confiança: {det['confidence']:.2f}")
    else:
        print("\n--- Nenhum item destacado (Frame Atual) ---")


# --- Script Principal ---
if __name__ == "__main__":
    model = YOLO('yolov8n.pt') # Pode usar 'yolov8n-seg.pt' para segmentação, por exemplo

    if SOURCE_IS_VIDEO:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
            exit()
        # Definir o callback do rato antes do loop, passando None inicialmente para param
        # O 'param' será atualizado com o frame atual dentro do loop
        cv2.namedWindow("Deteccoes YOLOv8 - Video Interativo")
        # Para passar o frame atual para o callback, podemos usar uma lambda ou um wrapper,
        # mas uma abordagem mais simples é chamar draw_interface_video diretamente após o clique
        # e garantir que mouse_callback_video tenha acesso ao frame correto para redesenhar.
        # O ideal seria que o param do setMouseCallback pudesse ser atualizado dinamicamente,
        # mas não é tão direto com a API padrão do OpenCV.
        # Por agora, vamos garantir que draw_interface_video use o frame correto.

    else: # Processamento de imagem única (código anterior adaptado)
        frame = cv2.imread(IMAGE_SOURCE)
        if frame is None:
            print(f"Erro: Não foi possível carregar a imagem: {IMAGE_SOURCE}")
            exit()
        cv2.namedWindow("Deteccoes YOLOv8 - Video Interativo") # Mesmo nome de janela

        # Processa a imagem única
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections_this_frame.clear()
        for box in results[0].boxes:
            detections_this_frame.append({
                'xyxy': box.xyxy[0].tolist(),
                'label': model.names[int(box.cls[0].item())],
                'confidence': box.conf[0].item()
            })
        cv2.setMouseCallback("Deteccoes YOLOv8 - Video Interativo", mouse_callback_video, frame)
        draw_interface_video(frame)


    # Loop principal (para vídeo ou para manter a imagem estática aberta)
    prev_time = 0 # Para cálculo do FPS
    while True:
        if SOURCE_IS_VIDEO:
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro na leitura do frame.")
                break

            # Atualiza o 'param' do callback do rato a cada novo frame.
            # Isto é uma forma de dar ao callback acesso ao frame atual.
            # No entanto, a função mouse_callback_video agora usa o frame que lhe é passado
            # diretamente na sua chamada de draw_interface_video.
            cv2.setMouseCallback("Deteccoes YOLOv8 - Video Interativo", mouse_callback_video, frame)


            # Processa o frame atual com YOLO
            # Adicionamos 'conf=CONFIDENCE_THRESHOLD' diretamente na chamada do modelo
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD) # stream=True pode ser mais eficiente para vídeo
            current_detections = results[0].boxes

            # Limpa as deteções do frame anterior e preenche com as novas
            detections_this_frame.clear()
            selected_box_index = -1 # Resetar seleção a cada novo frame

            for box in current_detections: # Já filtrado pela confiança no 'model()'
                detections_this_frame.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'label': model.names[int(box.cls[0].item())],
                    'confidence': box.conf[0].item()
                })

            # Calcular e mostrar FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30), # Canto superior direito
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, FPS_COLOR, 2, cv2.LINE_AA)

            # Desenha a interface (caixas, infos) no frame processado
            draw_interface_video(frame) # Passa o frame original para ser a base do desenho

        # Lógica de Teclado (comum para imagem e vídeo)
        key = cv2.waitKey(1) & 0xFF # Espera por 1ms

        if key == ord('q') or key == 27:
            break
        elif key == ord('n'):
            if detections_this_frame:
                selected_box_index = (selected_box_index + 1) % len(detections_this_frame)
                print_selected_info_video()
                if SOURCE_IS_VIDEO: draw_interface_video(frame) # Redesenha o frame atual
                else: draw_interface_video(cv2.imread(IMAGE_SOURCE)) # Redesenha a imagem estática
        elif key == ord('p'):
            if detections_this_frame:
                selected_box_index = (selected_box_index - 1 + len(detections_this_frame)) % len(detections_this_frame)
                print_selected_info_video()
                if SOURCE_IS_VIDEO: draw_interface_video(frame)
                else: draw_interface_video(cv2.imread(IMAGE_SOURCE))

        # Se não for vídeo, e uma tecla for pressionada (exceto q/esc/n/p),
        # o loop continua, mas nada muda visualmente até uma tecla de navegação.
        # Se for imagem única e nenhuma tecla for pressionada, waitKey(1) simplesmente retorna.
        # Se estivermos no modo de imagem e não for uma das teclas de controle, o loop continua.
        # Para sair do modo de imagem estática sem ser 'q' ou 'esc', pode precisar de outra condição.
        # No entanto, o 'q' ou 'esc' deve funcionar para ambos.
        if not SOURCE_IS_VIDEO and key != 255 and key not in [ord('n'), ord('p'), ord('q'), 27]:
             pass # Permite que a janela da imagem estática permaneça responsiva


    # Libera os recursos
    if SOURCE_IS_VIDEO:
        cap.release()
    cv2.destroyAllWindows()