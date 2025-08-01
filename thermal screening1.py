import numpy as np
import cv2

base_dir = 'sample videos/'
output_dir = 'outputs/'
threshold = 200
area_of_box = 700  # 3000 для ввода изображения
min_temp = 38  # по Цельсию
font_scale_caution = 1  # 2 для ввода изображения
font_scale_temp = 0.7  # 1 для ввода изображения


def convert_to_temperature(pixel_avg):
    """
    Преобразует значение пикселя (среднее) в температуру (Цельсий) в зависимости от аппаратного обеспечения камеры
    """
    fahrenheit = pixel_avg / 2.25

    return (fahrenheit - 32) / 1.8


def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

    # Двоичный порог
    _, binary_thresh = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)

    # Открытие изображения: Эрозия с последующим расширением
    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

    # Получение контуров из изображения, полученного в результате операции открытия
    contours, _ = cv2.findContours(image_opening, 1, 2)

    image_with_rectangles = np.copy(heatmap)

    temperatures = [0]
    for contour in contours:
        # прямоугольник над каждым контуром
        x, y, w, h = cv2.boundingRect(contour)

        # Пропуск, если площадь прямоугольника недостаточно велика
        if w * h < area_of_box:
            continue

        # Маска - это булевский тип матрицы.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)

        # Среднее значение только тех пикселей, которые находятся в блоках, а не во всем выделенном прямоугольнике
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        # Цвета для прямоугольников и textmin_area
        temperature = round(mean, 2)
        temperatures.append(temperature)
        color = (0, 255, 0) if temperature < min_temp else (
            255, 255, 127)

        # Функция обратного вызова при выполнении следующего условия
        if temperature >= min_temp:
            # Вызовите функцию обратного вызова здесь
            cv2.putText(image_with_rectangles, "Обнаружина высокая температура!", (35, 40),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale_caution, color, 2, cv2.LINE_AA)
        # Рисует прямоугольники для визуализации
        image_with_rectangles = cv2.rectangle(
            image_with_rectangles, (x, y), (x + w, y + h), color, 2)

        # Пишет температуру для каждого прямоугольника
        cv2.putText(image_with_rectangles, "{} C".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_temp, color, 2, cv2.LINE_AA)

    cv2.putText(image_with_rectangles, "Максимальная температура: {} C".format(max(temperatures)), (0, 700),
                cv2.FONT_HERSHEY_COMPLEX, font_scale_caution, (0, 255, 0), 1, cv2.LINE_AA)

    return image_with_rectangles


def main():
    """
    Основная функция
    """
    # Для ввода видео
    name = 'Позиция_1 Тепловизор 2024-09-26 17-28-21.mp4'
    video = cv2.VideoCapture(str(base_dir + name))
    video_frames = []
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = int(round(video.get(5)))
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Обработка каждого кадра
        frame = process_frame(frame)
        height, width, _ = frame.shape
        video_frames.append(frame)

        # Показ видео в окне в процессе обработки
        cv2.imshow('frame', video_frames[-1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Сохранение видео на выходе
    out = cv2.VideoWriter(output_dir + name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for i in range(len(video_frames)):
        out.write(video_frames[i])
    out.release()


if __name__ == "__main__":
    main()
