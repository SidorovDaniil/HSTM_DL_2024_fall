## HW 2

Это задание посвящено работе с сегментацией на датасате изображений изделий из стали (Severstal: Steel Defect Detection).
Северсталь использует машинное обучение для автоматического обнаружения дефектов на поверхности стали. Это помогает инженерам быстро и точно находить дефекты и принимать решения об их исправлении.

### Данные

Данные для этого задания можно скачать на [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection/data).
Изображения находятся в папке `train_images`, а разметка в файле `train.csv`.
Изображения черно-белые, размером 256x1600 пикселей.

train.csv содержит следующие колонки:
- `ImageId` - идентификатор изображения
- `ClassId` - идентификатор класса дефекта. Всего 4 класса дефектов: 1, 2, 3, 4;
- `EncodedPixels` - позиция дефекта на изображении

EncodedPixels - это строка вида `1 3 10 5 20 2`, которая кодирует маску одного класаа, где каждая пара чисел описывает позицию дефекта на изображении:
- Первое число - начальная позиция дефекта
- Второе число - длина дефекта в пикселях

в hw2_template.ipynb есть пример кода загрузки и визуализации изображений и разметки.

### Задание

1. Подготовить датасет для обучения модели сегментации дефектов на стали. Для этого нужно:
    - Заполнить класс SeverstalSteelDataset в hw2_template.ipynb:
        - Реализовать метод __len__ для возвращения размера датасета
        - Реализовать метод __getitem__ для возвращения изображения и маски дефекта по индексу в виде тензоров 
    - Реализовать функцию для разделения датасета на train и val
    - Создать DataLoader для train, val, test датасетов
2. Реализовать класс модели сегментации SegModel в hw2_template.ipynb. 
    - Он может использовать как предобученные модели из torchvision, segmentation_models_pytorch, так и свою архитектуру.
3. Обучить модель на датасете Severstal: Steel Defect Detection.
4. Предсказать дефекты на тестовом датасете и сохранить результат в виде масок изображений перобразовнных в формат EncodedPixels. 
5. Задание сдается в виде Jupyter Notebook/Google Colab ноутбука с кодом и submission.csv файла.

В качестве метрики качества будем использовать Dice score (IoU) для каждого класса дефекта и среднее значение по всем классам.





