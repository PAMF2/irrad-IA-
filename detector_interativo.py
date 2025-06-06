import cv2
from ultralytics import YOLO
import numpy as np

# --- Variáveis Globais para a Interface ---
img_display = None
img_original_for_drawing = None # Usaremos uma cópia limpa da imagem original para desenhar a cada vez
detections_list = []
selected_box_index = -1

COLOR_DEFAULT = (0, 255, 0) # Verde
COLOR_SELECTED = (0, 0, 255) # Vermelho
TEXT_COLOR_BOX = (255, 255, 255) # Branco para texto na caixa
TEXT_COLOR_INFO = (0, 0, 0) # Preto para informações no canto (melhor contraste em fundo claro)
INFO_BG_COLOR = (200, 200, 200, 100) # Cinza claro semi-transparente para fundo da info

# --- Função para Desenhar as Deteções e Informações ---
def draw_interface():
    global img_display, detections_list, selected_box_index, img_original_for_drawing

    if img_original_for_drawing is None or not detections_list:
        if img_original_for_drawing is not None: # Se só não há deteções, mostra a imagem original
            cv2.imshow("Deteccoes YOLOv8 - Interativo", img_original_for_drawing)
        return

    # Começa sempre com uma cópia limpa da imagem original
    current_img_to_show = img_original_for_drawing.copy()

    # Desenha todas as caixas delimitadoras
    for i, det in enumerate(detections_list):
        xyxy = det['xyxy']
        label = det['label']
        conf = det['confidence']
        color = COLOR_SELECTED if i == selected_box_index else COLOR_DEFAULT

        cv2.rectangle(current_img_to_show, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
        text_on_box = f"{label}: {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text_on_box, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(current_img_to_show, (int(xyxy[0]), int(xyxy[1]) - text_height - baseline), (int(xyxy[0]) + text_width, int(xyxy[1])), color, -1)
        cv2.putText(current_img_to_show, text_on_box, (int(xyxy[0]), int(xyxy[1]) - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR_BOX, 1, cv2.LINE_AA)

    # Desenha informações do item selecionado no canto superior esquerdo
    if selected_box_index != -1 and selected_box_index < len(detections_list):
        selected_det = detections_list[selected_box_index]
        info_text_l1 = f"Selecionado: {selected_det['label']}"
        info_text_l2 = f"Confianca: {selected_det['confidence']:.2f}"
        info_text_l3 = f"Coords (xyxy): {[int(c) for c in selected_det['xyxy']]}"

        # Adiciona um fundo para melhor legibilidade
        y_offset = 30
        cv2.rectangle(current_img_to_show, (5, 5), (350, y_offset * 3 + 10), INFO_BG_COLOR, -1) # Fundo
        cv2.putText(current_img_to_show, info_text_l1, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR_INFO, 2, cv2.LINE_AA)
        cv2.putText(current_img_to_show, info_text_l2, (10, y_offset * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR_INFO, 2, cv2.LINE_AA)
        cv2.putText(current_img_to_show, info_text_l3, (10, y_offset * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR_INFO, 2, cv2.LINE_AA)
    else:
        cv2.rectangle(current_img_to_show, (5, 5), (250, 35), INFO_BG_COLOR, -1) # Fundo
        cv2.putText(current_img_to_show, "Nenhum item selecionado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR_INFO, 2, cv2.LINE_AA)


    cv2.imshow("Deteccoes YOLOv8 - Interativo", current_img_to_show)

# --- Função de Callback do Rato ---
def mouse_callback(event, x, y, flags, param):
    global selected_box_index, detections_list

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_on_box_this_time = False
        for i, det in enumerate(detections_list):
            xmin, ymin, xmax, ymax = det['xyxy']
            if xmin < x < xmax and ymin < y < ymax:
                selected_box_index = i
                print_selected_info() # Função auxiliar para imprimir no console
                clicked_on_box_this_time = True
                break
        if not clicked_on_box_this_time:
            selected_box_index = -1
            print_selected_info()
        draw_interface()

# --- Função para imprimir informações do selecionado no console ---
def print_selected_info():
    global selected_box_index, detections_list
    if selected_box_index != -1 and selected_box_index < len(detections_list):
        det = detections_list[selected_box_index]
        print(f"\n--- Item Destacado (Índice: {selected_box_index}) ---")
        print(f"  Classe: {det['label']}")
        print(f"  Confiança: {det['confidence']:.2f}")
        print(f"  Coordenadas (xyxy): {[int(c) for c in det['xyxy']]}")
    else:
        print("\n--- Nenhum item destacado ---")

# --- Script Principal ---
if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    caminho_imagem = 'minha_imagem.jpg'

    img_bgr = cv2.imread(caminho_imagem)
    if img_bgr is None:
        print(f"Erro: Não foi possível carregar a imagem de '{caminho_imagem}'")
        exit()
    img_original_for_drawing = img_bgr.copy() # Mantém uma cópia original limpa

    print("Processando imagem com YOLO...")
    results = model(caminho_imagem, verbose=False)
    result = results[0]

    for box in result.boxes:
        detections_list.append({
            'xyxy': box.xyxy[0].tolist(),
            'label': model.names[int(box.cls[0].item())],
            'confidence': box.conf[0].item()
        })

    print(f"Número de objetos detetados: {len(detections_list)}")
    if not detections_list:
        print("Nenhum objeto detetado.")
    else:
        print("Pressione 'N' para próximo, 'P' para anterior. Clique para selecionar.")
        print("Pressione 'Q' ou ESC para sair.")

    cv2.namedWindow("Deteccoes YOLOv8 - Interativo")
    cv2.setMouseCallback("Deteccoes YOLOv8 - Interativo", mouse_callback)
    draw_interface() # Desenha pela primeira vez

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27: # 'q' ou tecla ESC para sair
            break
        elif key == ord('n'): # Tecla 'N' para próximo
            if detections_list: # Só faz algo se houver deteções
                selected_box_index = (selected_box_index + 1) % len(detections_list)
                print_selected_info()
                draw_interface()
        elif key == ord('p'): # Tecla 'P' para anterior
            if detections_list:
                selected_box_index = (selected_box_index - 1 + len(detections_list)) % len(detections_list)
                # A adição de len(detections_list) antes do módulo garante que o resultado seja sempre positivo
                # Ex: (-1 + 5) % 5 = 4 % 5 = 4. Se fosse só -1 % 5, poderia dar -1 em algumas implementações de módulo.
                print_selected_info()
                draw_interface()

    cv2.destroyAllWindows()