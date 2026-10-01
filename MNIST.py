import tensorflow as tf
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import itertools
import pandas as pd


GLOBAL_SEED = 42

os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28*28).astype("float32") / 255.0

histories = {}

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256_no_es = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_256_no_es.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256_no_es = model_256_no_es.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=2
)
histories['256_no_es'] = histories_256_no_es

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=5, restore_best_weights=True, verbose=1),
]

model_256.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256 = model_256.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)
histories['256'] = histories_256

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

ids = np.random.choice(len(x_train), 3000, replace=False)
x_train_short = x_train[ids]
y_train_short = y_train[ids]

histories = {}
histories['256'] = histories_256

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256_short = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=5, restore_best_weights=True, verbose=1),
]

model_256_short.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256_short = model_256_short.fit(
    x_train_short, y_train_short,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)
histories['256_short'] = histories_256_short

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

histories = {}
histories['256'] = histories_256

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_2048 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=5, restore_best_weights=True, verbose=1),
]

model_2048.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_2048 = model_2048.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)
histories['2048'] = histories_2048

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

histories = {}
histories['256'] = histories_256

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256_deep = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=5, restore_best_weights=True, verbose=1),
]

model_256_deep.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256_deep = model_256_deep.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)
histories['256_deep'] = histories_256_deep

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

histories = {}
histories['2048'] = histories_2048

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_2048_deep = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=5, restore_best_weights=True, verbose=1),
]

model_2048_deep.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_2048_deep = model_2048_deep.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)
histories['2048_deep'] = histories_2048_deep

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

histories = {}
histories['256'] = histories_256

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256_l1 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1(0.0001)
      ),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1),
]

model_256_l1.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256_l1 = model_256_l1.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)

histories['256_l1'] = histories_256_l1

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256_l2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001)
      ),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1),
]

model_256_l2.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy",
                       tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256_l2 = model_256_l2.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)

histories['256_l2'] = histories_256_l2

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

histories = {}
histories['256'] = histories_256

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256_drop_02 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=5, restore_best_weights=True, verbose=1),
]

model_256_drop_02.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy",
                       tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256_drop_02 = model_256_drop_02.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)

histories['256_drop_02'] = histories_256_drop_02

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256_drop_05 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1),
]

model_256_drop_05.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy",
                       tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256_drop_05 = model_256_drop_05.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)

histories['256_drop_05'] = histories_256_drop_05

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

histories = {}
histories['256'] = histories_256

tf.keras.backend.clear_session()
tf.random.set_seed(GLOBAL_SEED)

model_256_bn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),

    tf.keras.layers.Dense(10, activation="softmax")
])

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1),
]

model_256_bn.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy",
                       tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)])

histories_256_bn = model_256_bn.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)

histories['256_bn'] = histories_256_bn

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

histories = {}
histories['256'] = histories_256
histories['256_l1'] = histories_256_l1
histories['256_l2'] = histories_256_l2
histories['256_drop_02'] = histories_256_drop_02
histories['256_drop_05'] = histories_256_drop_05
histories['256_bn'] = histories_256_bn

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)

histories_comb_all = {}

# Definiujemy dostępne warianty technik regularyzacyjnych
reg_options = {
    'none': None,
    'l1': tf.keras.regularizers.l1(0.0001),
    'l2': tf.keras.regularizers.l2(0.0001),
    'l1_l2': tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)
}

drop_options = {
    'none': 0.0,
    'drop_02': 0.2,
    'drop_05': 0.5
}

bn_options = {
    'none': False,
    'bn': True
}

# Generujemy wszystkie 24 kombinacje
combinations = list(itertools.product(reg_options.items(), drop_options.items(), bn_options.items()))

print(f"Rozpoczynanie eksperymentu dla {len(combinations)} kombinacji modeli...")

for (reg_name, reg_val), (drop_name, drop_val), (bn_name, bn_val) in combinations:

    tf.keras.backend.clear_session()
    tf.random.set_seed(GLOBAL_SEED)

    # Budujemy czytelną i unikalną nazwę dla słownika historii
    # Jeśli technika jest wyłączona ('none'), pomijamy ją w sufixie, żeby nazwy były zwięzłe
    suffix_parts = []
    if reg_name != 'none': suffix_parts.append(reg_name)
    if drop_name != 'none': suffix_parts.append(drop_name)
    if bn_name != 'none': suffix_parts.append(bn_name)

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    history_key = f"256_deep{suffix}"

    # 1. Definiowanie warstw sieci dynamicznie w zależności od kombinacji
    layers = [tf.keras.layers.Input(shape=(784,))]
    hidden_units = [256, 128, 64, 16]

    for units in hidden_units:
        # Gęsta warstwa (z biasem lub bez, w zależności od Batch Normalization)
        layers.append(tf.keras.layers.Dense(
            units,
            use_bias=not bn_val,
            kernel_regularizer=reg_val
        ))

        # Opcjonalna warstwa Batch Normalization przed aktywacją
        if bn_val:
            layers.append(tf.keras.layers.BatchNormalization())

        # Aktywacja
        layers.append(tf.keras.layers.Activation("relu"))

        # Opcjonalna warstwa Dropout po aktywacji
        if drop_val > 0.0:
            layers.append(tf.keras.layers.Dropout(drop_val))

    # Warstwa wyjściowa
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))

    # Tworzenie modelu o nazwie zgodnej z Twoim formatem
    # Dynamicznie nadajemy nazwę zmiennej w przestrzeni lokalnej (opcjonalnie, dla porządku)
    model_name = f"model_256_deep{suffix}"
    model = tf.keras.Sequential(layers, name=model_name)

    # 2. Kompilacja i ustawienie specyficznego patience dla Dropout 0.2 (zgodnie z Twoim scenariuszem)
    patience_val = 5 if drop_name == 'drop_02' else 10
    callbacks = [EarlyStopping(monitor="val_pure_loss", patience=patience_val, restore_best_weights=True, verbose=0)]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)]
    )

    # 3. Trening modelu
    print(f"Trenowanie modelu: {model_name} (patience={patience_val})...")
    history = model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0 # ustawione na 0 przy 24 modelach, żeby nie zasypać konsoli tysiącami linii logów
    )

    # Zapisanie do słownika struktur
    histories_comb_all[history_key] = history

    # 1. Generowanie wykresu porównawczego
plotter = tfdocs.plots.HistoryPlotter(metric='pure_loss', smoothing_std=10)
# Ze względu na dużą liczbę linii (24), wykres może być gęsty, ale pokaże ogólne trendy
plotter.plot(histories_comb_all)

# 2. Generowanie tabeli zbiorczej posortowanej od najlepszego wyniku walidacji
records_comb = []
for name, history in histories_comb_all.items():
    best_epoch_idx = history.history['val_pure_loss'].index(min(history.history['val_pure_loss']))

    records_comb.append({
        "Model Name": name,
        "Best Epoch": best_epoch_idx + 1,
        "Train Loss": history.history['pure_loss'][best_epoch_idx],
        "Val Pure Loss": history.history['val_pure_loss'][best_epoch_idx],
        "Train Accuracy": history.history['accuracy'][best_epoch_idx],
        "Val Accuracy": history.history['val_accuracy'][best_epoch_idx]
    })

df_results_comb = pd.DataFrame(records_comb)
# Sortujemy po najniższym Val Pure Loss, aby od razu widzieć zwycięskie kombinacje
df_results_comb = df_results_comb.sort_values(by="Val Pure Loss", ascending=True)

print(df_results_comb.to_markdown(index=False))

histories_2048_comb = {}

# Definiujemy dostępne warianty technik regularyzacyjnych
reg_options = {
    'none': None,
    'l1': tf.keras.regularizers.l1(0.0001),
    'l2': tf.keras.regularizers.l2(0.0001),
    'l1_l2': tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)
}

drop_options = {
    'none': 0.0,
    'drop_02': 0.2,
    'drop_05': 0.5
}

bn_options = {
    'none': False,
    'bn': True
}

# Generujemy wszystkie 24 kombinacje
combinations = list(itertools.product(reg_options.items(), drop_options.items(), bn_options.items()))

print(f"Rozpoczynanie eksperymentu dla {len(combinations)} kombinacji modelu 2048...")

for (reg_name, reg_val), (drop_name, drop_val), (bn_name, bn_val) in combinations:

    tf.keras.backend.clear_session()
    tf.random.set_seed(GLOBAL_SEED)

    # Budujemy unikalny klucz dla słownika historii (np. '2048_deep_l1_drop_02_bn')
    suffix_parts = []
    if reg_name != 'none': suffix_parts.append(reg_name)
    if drop_name != 'none': suffix_parts.append(drop_name)
    if bn_name != 'none': suffix_parts.append(bn_name)

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    history_key = f"2048_deep{suffix}"

    # 1. Definiowanie warstw sieci o architekturze 2048 -> 512 -> 128 -> 32
    layers = [tf.keras.layers.Input(shape=(784,))]
    hidden_units = [2048, 512, 128, 32]

    for units in hidden_units:
        # Gęsta warstwa (wyłączamy bias, jeśli zaraz po niej idzie Batch Normalization)
        layers.append(tf.keras.layers.Dense(
            units,
            use_bias=not bn_val,
            kernel_regularizer=reg_val
        ))

        # Batch Normalization (opcjonalnie, przed aktywacją)
        if bn_val:
            layers.append(tf.keras.layers.BatchNormalization())

        # Aktywacja nieliniowa
        layers.append(tf.keras.layers.Activation("relu"))

        # Dropout (opcjonalnie, po aktywacji)
        if drop_val > 0.0:
            layers.append(tf.keras.layers.Dropout(drop_val))

    # Warstwa wyjściowa
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))

    # Tworzenie instancji modelu o pełnej nazwie
    model_name = f"model_2048_deep{suffix}"
    model = tf.keras.Sequential(layers, name=model_name)

    # 2. Kompilacja — patience ustawione na stałe na 10 dla wszystkich modeli
    callbacks = [EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=0)]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)]
    )

    # 3. Trening modelu
    print(f"Trenowanie modelu: {model_name} (patience=10)...")
    history = model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0  # Wyciszenie logów epok, aby zachować czytelność konsoli
    )

    # Zapis historii do słownika zbiorczego
    histories_2048_comb[history_key] = history

# 1. Wykres zbiorczy historii zmian pure_loss
plotter = tfdocs.plots.HistoryPlotter(metric='pure_loss', smoothing_std=10)
plotter.plot(histories_2048_comb)

# 2. Tworzenie i formatowanie tabeli wynikowej za pomocą pandas
records_2048_comb = []
for name, history in histories_2048_comb.items():
    # Znajdujemy indeks epoki o najniższej stracie walidacyjnej (czyli moment, do którego cofnął nas callback)
    best_epoch_idx = history.history['val_pure_loss'].index(min(history.history['val_pure_loss']))

    records_2048_comb.append({
        "Model Name": name,
        "Best Epoch": best_epoch_idx + 1,
        "Train Loss": history.history['pure_loss'][best_epoch_idx],
        "Val Pure Loss": history.history['val_pure_loss'][best_epoch_idx],
        "Train Accuracy": history.history['accuracy'][best_epoch_idx],
        "Val Accuracy": history.history['val_accuracy'][best_epoch_idx]
    })


df_results_2048_comb = pd.DataFrame(records_2048_comb)

# Sortujemy tabelę automatycznie po najniższym 'Val Pure Loss', wskazując najbardziej efektywne konfiguracje
df_results_2048_comb = df_results_2048_comb.sort_values(by="Val Pure Loss", ascending=True)

# Wyświetlenie pełnej tabeli w formacie Markdown
print(df_results_2048_comb.to_markdown(index=False))

histories_256_comb = {}

# Definiujemy dostępne warianty technik regularyzacyjnych
reg_options = {
    'none': None,
    'l1': tf.keras.regularizers.l1(0.0001),
    'l2': tf.keras.regularizers.l2(0.0001),
    'l1_l2': tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)
}

drop_options = {
    'none': 0.0,
    'drop_02': 0.2,
    'drop_05': 0.5
}

bn_options = {
    'none': False,
    'bn': True
}

# Generujemy wszystkie 24 kombinacje
combinations = list(itertools.product(reg_options.items(), drop_options.items(), bn_options.items()))

print(f"Rozpoczynanie eksperymentu dla {len(combinations)} kombinacji modelu 256...")

for (reg_name, reg_val), (drop_name, drop_val), (bn_name, bn_val) in combinations:

    tf.keras.backend.clear_session()
    tf.random.set_seed(GLOBAL_SEED)

    # Budujemy unikalny klucz dla słownika historii (np. '256_l1_drop_02_bn')
    suffix_parts = []
    if reg_name != 'none': suffix_parts.append(reg_name)
    if drop_name != 'none': suffix_parts.append(drop_name)
    if bn_name != 'none': suffix_parts.append(bn_name)

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    history_key = f"256{suffix}"

    # 1. Definiowanie pojedynczej warstwy ukrytej (256 neuronów)
    layers = [tf.keras.layers.Input(shape=(784,))]

    # Gęsta warstwa (wyłączamy bias, jeśli zaraz po niej idzie Batch Normalization)
    layers.append(tf.keras.layers.Dense(
        256,
        use_bias=not bn_val,
        kernel_regularizer=reg_val
    ))

    # Batch Normalization (opcjonalnie, przed aktywacją)
    if bn_val:
        layers.append(tf.keras.layers.BatchNormalization())

    # Aktywacja nieliniowa
    layers.append(tf.keras.layers.Activation("relu"))

    # Dropout (opcjonalnie, po aktywacji)
    if drop_val > 0.0:
        layers.append(tf.keras.layers.Dropout(drop_val))

    # Warstwa wyjściowa
    layers.append(tf.keras.layers.Dense(10, activation="softmax"))

    # Tworzenie instancji modelu o pełnej nazwie zgodnie z formatem
    model_name = f"model_256{suffix}"
    model = tf.keras.Sequential(layers, name=model_name)

    # 2. Kompilacja — patience ustawione na stałe na 10 dla wszystkich modeli
    callbacks = [EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=0)]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.losses.SparseCategoricalCrossentropy(name="pure_loss", from_logits=False)]
    )

    # 3. Trening modelu
    print(f"Trenowanie modelu: {model_name} (patience=10)...")
    history = model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0  # Wyciszenie logów epok w celu zachowania czytelności konsoli
    )

    # Zapis historii do słownika zbiorczego
    histories_256_comb[history_key] = history


# 1. Wykres zbiorczy historii zmian pure_loss
plotter = tfdocs.plots.HistoryPlotter(metric='pure_loss', smoothing_std=10)
plotter.plot(histories_256_comb)

# 2. Tworzenie i formatowanie tabeli wynikowej za pomocą pandas
records_256_comb = []
for name, history in histories_256_comb.items():
    # Znajdujemy indeks epoki o najniższej stracie walidacyjnej
    best_epoch_idx = history.history['val_pure_loss'].index(min(history.history['val_pure_loss']))

    records_256_comb.append({
        "Model Name": name,
        "Best Epoch": best_epoch_idx + 1,
        "Train Loss": history.history['pure_loss'][best_epoch_idx],
        "Val Pure Loss": history.history['val_pure_loss'][best_epoch_idx],
        "Train Accuracy": history.history['accuracy'][best_epoch_idx],
        "Val Accuracy": history.history['val_accuracy'][best_epoch_idx]
    })


df_results_256_comb = pd.DataFrame(records_256_comb)

# Sortujemy tabelę automatycznie po najniższym 'Val Pure Loss'
df_results_256_comb = df_results_256_comb.sort_values(by="Val Pure Loss", ascending=True)

# Wyświetlenie pełnej tabeli w formacie Markdown
print(df_results_256_comb.to_markdown(index=False))
