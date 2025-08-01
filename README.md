# Мини-Реферат по Программе «thermal screening average»

Программа "Thermal_Screening_Temperature_Detection" предназначена для автоматического анализа тепловизионных видеозаписей с целью обнаружения объектов и сущностей с повышенной температурой, и записи средней температуры в текстовый файл. Программа написана на языке Python и использует библиотеки OpenCV, NumPy и другие вспомогательные библиотеки для обработки изображений и видео.
Функциональность:
Программа принимает на вход путь к папке с видеофайлами (или изображениями), выполняет обработку каждого кадра, выделяя области, соответствующие объектам (или другим лицам) с тепловым излучением. Для этого используется пороговая обработка изображения и анализ контуров. Затем программа вычисляет среднюю температуру в каждой выделенной области, преобразуя среднее значение пикселей в градусы Цельсия с учетом калибровки камеры. Если средняя температура превышает заданный порог (min_temp), программа помечает соответствующую область на изображении и выводит предупреждающее сообщение. Результаты анализа (средняя температура на каждой секунде, время обработки) сохраняются в текстовый файл и в виде последовательности обработанных изображений.
Основные алгоритмы и методы:
•	Преобразование цветовой модели: Преобразование изображения из BGR в RGB и затем в градации серого для упрощения обработки.
•	Пороговая обработка: Бинаризация изображения для выделения областей с высокой интенсивностью (тепловыми аномалиями).
•	Морфологическая обработка: Применение операций эрозии и дилатации для улучшения качества бинарного изображения и удаления шума.
•	Анализ контуров: Обнаружение контуров на бинарном изображении для локализации объектов.
•	Вычисление средней температуры: Вычисление среднего значения пикселей в выделенной области и преобразование его в температуру.
•	Визуализация результатов: Наложение прямоугольников и текстовых меток на исходное изображение для отображения результатов анализа.
Преимущества:
•	Автоматизация процесса анализа тепловизионных видеозаписей.
•	Возможность обнаружения объектов с повышенной температурой.
•	Визуализация результатов анализа на видеокадрах.
•	Сохранение результатов в текстовом файле.
Недостатки:
•	Точность обнаружения и измерения температуры зависит от качества тепловизионного изображения, калибровки камеры и параметров обработки.
•	Не учитывает различные факторы, влияющие на температуру тела (одежда, окружающая среда).
•	Требует предварительной настройки параметров, таких как порог температуры и минимальная площадь области.
Пояснения к листингу

Импортируем библиотеки необходимые для работы с изображением, вычислением и датой-временем:
import numpy as np
import cv2
import os
import math
import datetime
Определяем пути к папкам и константные переменные
video_path = "video_files/"
output_path = "output_path/"
threshold = 200
area_of_box = 700  # 3000 для ввода изображения
min_temp = 38  # по Цельсию
font_scale_caution = 1  # 2 для ввода изображения
font_scale_temp = 0.7  # 1 для ввода изображения
Основная функция «main»:
def main():
    """
    Основная функция
Цикл загрузки всех видеофайлов в папке на обработку
    for file_name in os.listdir(video_path):
        video = cv2.VideoCapture(str(video_path + file_name))
        video_frames = []
        avr_temperatures = []
        fps = int(round(video.get(5)))
Обработка каждого кадра
        while video.isOpened():
            frame_id = video.get(1)
            ret, frame = video.read()
            if not ret:
                break

Обработка каждого кадра
            if frame_id % math.floor(fps) == 0:
                # Обработка каждого кадра
                frame, avr_temp = process_frame(frame)
                video_frames.append(frame)
                avr_temperatures.append(avr_temp)

Показ видео в окне в процессе обработки

                cv2.imshow('frame', video_frames[-1])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

Сохранение данных в текстовый файл

        os.makedirs(output_path + os.path.splitext(file_name)[0], exist_ok=True)
        base_path = os.path.join(output_path, os.path.splitext(file_name)[0], os.path.splitext(file_name)[0])
        digit = len(str(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))

                txt_path = os.path.join(base_path + ".txt")
        with open(txt_path, "w+") as file:
            
Сохранение видео на выходе
            for i in range(len(video_frames)):
                file.seek(0, os.SEEK_END)
                
проверяет есть в txt-файле текст
                check_file = file.tell()
                if check_file == 0:
                    file.write(f"Дата/Время    \t  | Кадр | Ср. Значения\n\n")
                else:
                    cv2.imwrite('{}_{}.{}'.format(base_path, str(i).zfill(digit), 'jpg'), video_frames[i])
                    file.write(f"{datetime.datetime.now().replace(microsecond=0)} | {str(i).zfill(digit)} | "
                               f"{round(avr_temperatures[i], 2)} C \n\n")

Запуск программы
if __name__ == "__main__":
    main()
Функция обработки кадра. 
def process_frame(frame):
    height, width, channels = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)
Двоичный порог
_, binary_thresh = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)
Открытие изображения: Эрозия с последующим расширением
kernel = np.ones((3, 3), np.uint8) image_erosion = cv2.erode(binary_thresh, kernel, iterations=1) image_opening = cv2.dilate(image_erosion, kernel, iterations=1)
Получение контуров из изображения, полученного в результате операции открытия
contours, _ = cv2.findContours(image_opening, 1, 2)

image_with_rectangles = np.copy(heatmap)

temperatures = [0] for contour in contours:
прямоугольник над каждым контуром
x, y, w, h = cv2.boundingRect(contour)
Пропуск, если площадь прямоугольника недостаточно велика
if w * h < area_of_box: continue
Пропуск, если площадь прямоугольника недостаточно велика.
mask = np.zeros_like(heatmap_gray) cv2.drawContours(mask, contour, -1, 255, -1)
Среднее значение только тех пикселей, которые находятся в блоках, а не во всем выделенном прямоугольнике
mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])
Цвета для прямоугольников и textmin_area
temperature = round(mean, 2) temperatures.append(temperature) color = (0, 255, 0) if temperature < min_temp else ( 255, 255, 127)
Функция обратного вызова при выполнении следующего условия
if temperature >= min_temp:
Вызовите функцию обратного вызова здесь
cv2.putText(image_with_rectangles, "Обнаружина высокая температура!", (35, 40), cv2.FONT_HERSHEY_COMPLEX, font_scale_temp, color, 1, cv2.LINE_AA)
Рисует прямоугольники для визуализации
image_with_rectangles = cv2.rectangle( image_with_rectangles, (x, y), (x + w, y + h), color, 2)
Пишет температуру для каждого прямоугольника
        cv2.putText(image_with_rectangles, "{} C".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_temp, color, 2, cv2.LINE_AA)

 Вычисляем и пишем среднею температуру на кадре
    avr_temperature = sum(temperatures) / len(temperatures)
    cv2.putText(image_with_rectangles, "Средняя температура: {} C".format(round(avr_temperature, ndigits=1)),
                (0, height-10),
                cv2.FONT_HERSHEY_COMPLEX, font_scale_temp, (0, 255, 0), 1, cv2.LINE_AA)

Возвращаем в основную функцию обработанное кадр и средние значение температуры. 

    return image_with_rectangles, avr_temperature
Функция для преобразования температуры. В зависимости от аппаратного обеспечения камеры преобразует значение пикселя (среднее) в температуру Фаренгейт и затем переводит её в градусы Цельсии.
def convert_to_temperature(pixel_avg):
    
    fahrenheit = pixel_avg / 2.25

    return (fahrenheit - 32) / 1.8
