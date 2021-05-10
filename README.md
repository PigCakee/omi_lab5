# Лабораторная работа #5
## 2. С использованием примера, техники обучения Transfer Learning, оптимальной политики изменения темпа обучения, аугментации данных с оптимальными настройками обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101

По результатам прошлой рабораторной работы (№4) наилучшие результаты были достигнуты в следюущем случае:
Фиксированный темп обучения 0.001
```
optimizer=tf.optimizers.Adam(lr=0.001)
```
Поворот на случайный угол
```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.1, fill_mode='nearest', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```
При использовании вышеперечисленных техник удалось достичь максимальной точности в 66.9%

Transfer Learning results:

tensorboard.dev/experiment/vvCqHN97RXqbAVmNiZK0Og/#scalars&runSelectionState=eyJmMTAxLTE2MTk2MzM4MDAuMzk4ODY3NC90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYzMzgwMC4zOTg4Njc0L3ZhbGlkYXRpb24iOmZhbHNlLCJmMTAxLTE2MTk2MzUzODIuMDI3NjEzNi90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYzNTM4Mi4wMjc2MTM2L3ZhbGlkYXRpb24iOmZhbHNlLCJmMTAxLTE2MTk2MzY4NDQuMDA4MzkyOC90cmFpbiI6dHJ1ZSwiZjEwMS0xNjE5NjM2ODQ0LjAwODM5MjgvdmFsaWRhdGlvbiI6dHJ1ZX0%3D

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_fill_mode_nearest.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_fill_mode_nearest.svg">

## 3. С использованием техники обучения Fine Tuning дополнительно обучить нейронную сеть EfficientNet-B0 предварительно обученную в пункте 2

Разблокируем веса
```
def unfreeze_model(model):
  for layer in model.layers:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
      layer.trainable = True
```
Используем Fine Tuning
```
unfreeze_model(model)

  log_dir='{}/f101-{}'.format(LOG_DIR, time.time())
  model.compile(
    optimizer=tf.optimizers.Adam(lr=1e-6),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
  )
  model.fit(
    train_dataset,
    epochs=20,
    validation_data=validation_dataset,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir),
    ]
  )
```

В ходе исследований было протестированно 4 различных значения параметра темпа обучения:

## 1. lr=1e-6

Fine Tuning results
https://tensorboard.dev/experiment/VjcnjWc7Q0yGbipzds2DIw/#scalars

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab5/main/epoch_categorical_accuracy_1e-6.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab5/main/epoch_loss_1e-6.svg">

График точности с первой же эпохи пошел резко вниз, ровно как график потерь пошел вверх. Делаем вывод, что параметр подобран неверно и понижаем его. 

## 2. lr=1e-9

Fine Tuning results
https://tensorboard.dev/experiment/Spm5xe1DRVyDA0WqbeqRwA/#scalars&runSelectionState=eyJmMTAxLTE2MjA1NjM4NzEuMDYzMTU4OC92YWxpZGF0aW9uIjpmYWxzZSwiZjEwMS0xNjIwNTYzODcxLjA2MzE1ODgvdHJhaW4iOmZhbHNlfQ%3D%3D

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab5/main/epoch_categorical_accuracy_1e-9.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab5/main/epoch_loss_1e-9.svg">

Графики практически не двигаются, а значит значение параметра слишком сильно занижено. Повышаем в 10 раз.

## 3. lr=1e-8

Fine Tuning results
https://tensorboard.dev/experiment/PRLR0NfGTru4CQdMPP358w/#scalars&runSelectionState=eyJmMTAxLTE2MjA1Njc1OTQuMjM1MjIyNi92YWxpZGF0aW9uIjpmYWxzZSwiZjEwMS0xNjIwNTY3NTk0LjIzNTIyMjYvdHJhaW4iOmZhbHNlfQ%3D%3D

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab5/main/epoch_categorical_accuracy_1e-8.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab5/main/epoch_loss_1e-8.svg">

Графики практически не двигаются, но уже видно тенденцию на улучшение. Повышаем еще в 10 раз.

## 4. lr=1e-7

Fine Tuning results
https://tensorboard.dev/experiment/AACxk4ErTD2Y83lv0UOiJw/#scalars&runSelectionState=eyJmMTAxLTE2MjA1OTU3NTEuMzYzMDkyNC90cmFpbiI6dHJ1ZSwiZjEwMS0xNjIwNTk1NzUxLjM2MzA5MjQvdmFsaWRhdGlvbiI6dHJ1ZSwiZjEwMS0xNjIwNTkyNzc1Ljc1NTE3NjMvdmFsaWRhdGlvbiI6ZmFsc2UsImYxMDEtMTYyMDU5Mjc3NS43NTUxNzYzL3RyYWluIjpmYWxzZX0%3D

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab5/main/epoch_categorical_accuracy_1e-7.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab5/main/epoch_loss_1e-7.svg">

Наивысшая точность, которую удалось достичь в этом случае - 67.5%, что улучшило результаты Transfer Learning на 0.6% (важно отметить, что сеть также сразу начала переобучаться, о чем свидетельствует растущий график потерь)

## Анализ результатов
В ходе экспериментов удалось лишь едва (0.6%) улучшить результаты Transfer Learning, используя метод Fine Tuning. Можно предположить, что идеальный параметр темпа обучения не был найден, однако определенного успеха добиться получилось :)
