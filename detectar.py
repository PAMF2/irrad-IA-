from ultralytics import YOLO
from PIL import Image # Pillow para manipulação de imagens

# 1. Carregar um modelo YOLOv8 pré-treinado
# Existem diferentes tamanhos de modelo: yolov8n.pt (nano, mais rápido), yolov8s.pt (small),
# yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (extra-large, mais preciso)
# Vamos usar o 'yolov8n.pt' que é pequeno e rápido para começar.
model = YOLO('yolov8n.pt')

# 2. Definir o caminho para a sua imagem
caminho_imagem = 'minha_imagem.jpg' # SUBSTITUA PELO NOME DA SUA IMAGEM SE FOR DIFERENTE

# 3. Fazer a deteção na imagem
# O modelo retorna uma lista de objetos 'Results'. Como estamos a processar uma única imagem, teremos uma lista com um elemento.
results = model(caminho_imagem)

# 4. Processar e mostrar os resultados
# O primeiro (e único) elemento da lista 'results' contém as deteções para a nossa imagem.
result = results[0]

# Quantos objetos foram detetados?
print(f"Número de objetos detetados: {len(result.boxes)}")

# Iterar sobre cada objeto detetado
for i, box in enumerate(result.boxes):
    # Obter as coordenadas da caixa delimitadora (bounding box)
    # no formato (xmin, ymin, xmax, ymax)
    coordenadas = box.xyxy[0].tolist()
    # Obter a classe do objeto (como um número/índice)
    classe_id = int(box.cls[0].item())
    # Obter o nome da classe a partir do ID
    nome_classe = model.names[classe_id]
    # Obter a confiança da deteção
    confianca = box.conf[0].item()

    print(f"  Objeto {i+1}:")
    print(f"    Classe: {nome_classe}")
    print(f"    Confiança: {confianca:.2f}") # Formata para 2 casas decimais
    print(f"    Coordenadas (xyxy): {coordenadas}")
    print("-" * 20)

# 5. Mostrar a imagem com as deteções desenhadas (opcional, mas útil)
# O objeto 'result' tem um método 'plot()' que retorna a imagem com as caixas desenhadas (requer OpenCV instalado pela ultralytics).
imagem_com_deteccoes_array = result.plot() # Retorna um array NumPy (BGR)

# Converter o array NumPy (BGR) para uma imagem PIL (RGB) para fácil visualização
imagem_com_deteccoes_pil = Image.fromarray(imagem_com_deteccoes_array[..., ::-1]) # Converte BGR para RGB

# Mostrar a imagem
imagem_com_deteccoes_pil.show()

# Opcionalmente, pode guardar a imagem:
# imagem_com_deteccoes_pil.save('resultado_deteccao.jpg')
# print("Imagem com deteções guardada como resultado_deteccao.jpg")