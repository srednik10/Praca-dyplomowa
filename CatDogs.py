import os
import keras
import keras.callbacks
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from tensorflow import data as tf_data
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        print(fname)
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print(f"Usunieto: {num_skipped}")

batch_size = 128
image_size = (100, 100)

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size = image_size,
    batch_size = batch_size
)

train_ds = train_ds.cache().prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf_data.AUTOTUNE)

def make_model_small_cnn_flatten(input_shape=(100, 100, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(1, activation=None)(x)

    return keras.Model(inputs, outputs)
model_small_cnn_flatten = make_model_small_cnn_flatten(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_small_cnn_flatten.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_residual = model_small_cnn_flatten.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

# Słownik na historie treningowe
histories_cnn_flatten = {}

# Definiujemy dostępne warianty technik regularyzacyjnych
reg_options = {
    "none": None,
    "l1": keras.regularizers.l1(0.0001),
    "l2": keras.regularizers.l2(0.0001),
    "l1_l2": keras.regularizers.l1_l2(l1=0.0001, l2=0.0001),
}

drop_options = {"none": 0.0, "drop_02": 0.2, "drop_05": 0.5}

bn_options = {"none": False, "bn": True}

# Generujemy 24 kombinacje
combinations = list(
    itertools.product(reg_options.items(), drop_options.items(), bn_options.items())
)

GLOBAL_SEED = 1337
print(
    f"Rozpoczynanie eksperymentu dla {len(combinations)} kombinacji modeli Small CNN Flatten..."
)

for (reg_name, reg_val), (drop_name, drop_val), (bn_name, bn_val) in combinations:

    # Czyszczenie sesji i ustawianie powtarzalnego seedu
    keras.backend.clear_session()
    tf.random.set_seed(GLOBAL_SEED)

    # Budowanie unikalnej nazwy klucza dla historii
    suffix_parts = []
    if reg_name != "none":
        suffix_parts.append(reg_name)
    if drop_name != "none":
        suffix_parts.append(drop_name)
    if bn_name != "none":
        suffix_parts.append(bn_name)

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    history_key = f"small_cnn_flatten{suffix}"
    model_name = f"model_{history_key}"

    # DYNAMICZNA BUDOWA ARCHITEKTURY SMALL CNN FLATTEN
    inputs = keras.Input(shape=(100, 100, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Blok 1
    x = layers.Conv2D(
        16, 3, strides=2, padding="same", kernel_regularizer=reg_val, use_bias=not bn_val
    )(x)
    if bn_val:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if drop_val > 0.0:
        x = layers.Dropout(drop_val)(x)
    x = layers.MaxPooling2D(2)(x)

    # Blok 2
    x = layers.Conv2D(
        32, 3, padding="same", kernel_regularizer=reg_val, use_bias=not bn_val
    )(x)
    if bn_val:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if drop_val > 0.0:
        x = layers.Dropout(drop_val)(x)
    x = layers.MaxPooling2D(2)(x)

    # Blok 3
    x = layers.Conv2D(
        64, 3, padding="same", kernel_regularizer=reg_val, use_bias=not bn_val
    )(x)
    if bn_val:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if drop_val > 0.0:
        x = layers.Dropout(drop_val)(x)

    # Końcówka modelu (Wersja z Flatten)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation=None)(x)

    model = keras.Model(inputs, outputs, name=model_name)

    patience_val = 10
    callbacks = [
        EarlyStopping(
            monitor="val_pure_loss",
            patience=patience_val,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    # Kompilacja zgodnie z Twoimi parametrami (z binary crossentropy z logitów)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True),
        ],
    )

    print(f"Trenowanie: {model_name} (patience={patience_val})...")

    # Trening na pobranych wcześniej i przygotowanych zbiorach train_ds oraz val_ds
    history = model.fit(
        train_ds,
        epochs=150,  # Zgodnie z Twoim zleceniem: maks 150 epok
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,  # Wyłączone logi epok, aby nie zaśmiecać konsoli
    )

    # Zapisanie do słownika zbiorczego
    histories_cnn_flatten[history_key] = history

results_data = []

for key, history in histories_cnn_flatten.items():
    # Pobieramy indeks najlepszej epoki na podstawie najniższego błędu walidacji
    val_pure_loss_history = history.history["val_pure_loss"]
    best_epoch_idx = val_pure_loss_history.index(min(val_pure_loss_history))

    results_data.append(
        {
            "Model Wariant": key,
            "Liczba epok": len(val_pure_loss_history),
            "Best Epoch": best_epoch_idx + 1,
            "Train Loss": history.history["pure_loss"][best_epoch_idx],
            "Val Loss": history.history["val_pure_loss"][best_epoch_idx],
            "Train Acc": history.history["accuracy"][best_epoch_idx],
            "Val Acc": history.history["val_accuracy"][best_epoch_idx],
        }
    )

# Konwersja na DataFrame i wyświetlenie
df_results = pd.DataFrame(results_data)
# Sortowanie od najlepszego wyniku na zbiorze walidacyjnym (najwyższy Accuracy)
df_results = df_results.sort_values(by="Val Acc", ascending=False).reset_index(
    drop=True
)

# Wyświetlenie pełnej tabeli w środowisku (np. Jupyter / Colab)
print("\n--- ZESTAWIENIE WYNIKÓW DLA 24 WARIANTÓW ---")
print(df_results)

def make_model_small_cnn_gap(input_shape=(100, 100, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(1, activation=None)(x)

    return keras.Model(inputs, outputs)
model_small_cnn_gap = make_model_small_cnn_gap(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_small_cnn_gap.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_residual = model_small_cnn_gap.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

# Słownik na historie treningowe wersji GAP
histories_cnn_gap = {}

# Definiujemy dostępne warianty technik regularyzacyjnych
reg_options = {
    "none": None,
    "l1": keras.regularizers.l1(0.0001),
    "l2": keras.regularizers.l2(0.0001),
    "l1_l2": keras.regularizers.l1_l2(l1=0.0001, l2=0.0001),
}

drop_options = {"none": 0.0, "drop_02": 0.2, "drop_05": 0.5}

bn_options = {"none": False, "bn": True}

# Generujemy 24 kombinacje
combinations = list(
    itertools.product(reg_options.items(), drop_options.items(), bn_options.items())
)

GLOBAL_SEED = 1337
print(
    f"Rozpoczynanie eksperymentu dla {len(combinations)} kombinacji modeli Small CNN GAP..."
)

for (reg_name, reg_val), (drop_name, drop_val), (bn_name, bn_val) in combinations:

    # Czyszczenie sesji i ustawianie powtarzalnego seedu
    keras.backend.clear_session()
    tf.random.set_seed(GLOBAL_SEED)

    # Budowanie unikalnej nazwy klucza dla historii
    suffix_parts = []
    if reg_name != "none":
        suffix_parts.append(reg_name)
    if drop_name != "none":
        suffix_parts.append(drop_name)
    if bn_name != "none":
        suffix_parts.append(bn_name)

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    history_key = f"small_cnn_gap{suffix}"
    model_name = f"model_{history_key}"

    # DYNAMICZNA BUDOWA ARCHITEKTURY SMALL CNN GAP
    inputs = keras.Input(shape=(100, 100, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Blok 1
    x = layers.Conv2D(
        16, 3, strides=2, padding="same", kernel_regularizer=reg_val, use_bias=not bn_val
    )(x)
    if bn_val:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if drop_val > 0.0:
        x = layers.Dropout(drop_val)(x)
    x = layers.MaxPooling2D(2)(x)

    # Blok 2
    x = layers.Conv2D(
        32, 3, padding="same", kernel_regularizer=reg_val, use_bias=not bn_val
    )(x)
    if bn_val:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if drop_val > 0.0:
        x = layers.Dropout(drop_val)(x)
    x = layers.MaxPooling2D(2)(x)

    # Blok 3
    x = layers.Conv2D(
        64, 3, padding="same", kernel_regularizer=reg_val, use_bias=not bn_val
    )(x)
    if bn_val:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if drop_val > 0.0:
        x = layers.Dropout(drop_val)(x)

    # --- ZMIANA: Global Average Pooling zamiast Flatten ---
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation=None)(x)

    model = keras.Model(inputs, outputs, name=model_name)

    # Dobór patience: dla drop_02 wynosi 5, dla pozostałych 10
    patience_val = 10
    callbacks = [
        EarlyStopping(
            monitor="val_pure_loss",
            patience=patience_val,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    # Kompilacja (identyczna jak dla wersji Flatten)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True),
        ],
    )

    print(f"Trenowanie: {model_name} (patience={patience_val})...")

    # Trening na tych samych zbiorach danych
    history = model.fit(
        train_ds,
        epochs=150,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,
    )

    # Zapisanie do nowego słownika zbiorczego
    histories_cnn_gap[history_key] = history

results_gap_data = []

for key, history in histories_cnn_gap.items():
    val_pure_loss_history = history.history["val_pure_loss"]
    best_epoch_idx = val_pure_loss_history.index(min(val_pure_loss_history))

    results_gap_data.append(
        {
            "Model Wariant (GAP)": key,
            "Liczba epok": len(val_pure_loss_history),
            "Best Epoch": best_epoch_idx + 1,
            "Train Loss": history.history["pure_loss"][best_epoch_idx],
            "Val Loss": history.history["val_pure_loss"][best_epoch_idx],
            "Train Acc": history.history["accuracy"][best_epoch_idx],
            "Val Acc": history.history["val_accuracy"][best_epoch_idx],
        }
    )

df_gap_results = pd.DataFrame(results_gap_data)
df_gap_results = df_gap_results.sort_values(by="Val Acc", ascending=False).reset_index(
    drop=True
)

print("\n--- ZESTAWIENIE WYNIKÓW DLA 24 WARIANTÓW (SMALL CNN GAP) ---")
print(df_gap_results)

def make_model_big_cnn_flatten(input_shape=(100, 100, 3)):
    model_layers = [
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255)
    ]

    model_layers.append(layers.Conv2D(16, 3, strides=2, padding="same"))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Conv2D(32, 3, padding="same"))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Conv2D(64, 3, padding="same"))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Flatten())
    model_layers.append(layers.Dense(512))
    model_layers.append(layers.Activation("relu"))

    model_layers.append(layers.Dense(1, activation=None))

    return keras.Sequential(model_layers, name="model_vgg_base")

model_big_cnn_flatten = make_model_big_cnn_flatten(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_big_cnn_flatten.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_big_cnn_flatten = model_big_cnn_flatten.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

# Słownik na historie treningowe wersji Big CNN Flatten
histories_big_flatten = {}

# Definiujemy dostępne warianty technik regularyzacyjnych
reg_options = {
    "none": None,
    "l1": regularizers.l1(0.0001),
    "l2": regularizers.l2(0.001),
    "l1_l2": regularizers.l1_l2(l1=0.0001, l2=0.001),
}

drop_options = {"none": 0.0, "drop_02": 0.2, "drop_05": 0.5}

bn_options = {"none": False, "bn": True}

# Generujemy 24 kombinacje
combinations = list(
    itertools.product(reg_options.items(), drop_options.items(), bn_options.items())
)

GLOBAL_SEED = 1337
epochs = 150

print(
    f"Rozpoczynanie eksperymentu dla {len(combinations)} kombinacji modelu Big CNN Flatten..."
)

for (reg_name, reg_val), (drop_name, drop_val), (bn_name, bn_val) in combinations:

    # Czyszczenie sesji i ustawianie powtarzalnego seedu
    keras.backend.clear_session()
    tf.random.set_seed(GLOBAL_SEED)

    # Budowanie unikalnej nazwy klucza dla historii
    suffix_parts = []
    if reg_name != "none":
        suffix_parts.append(reg_name)
    if drop_name != "none":
        suffix_parts.append(drop_name)
    if bn_name != "none":
        suffix_parts.append(bn_name)

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    history_key = f"big_flatten{suffix}"
    model_name = f"model_big_flatten{suffix}"

    # DYNAMICZNA BUDOWA ARCHITEKTURY BIG CNN FLATTEN
    model_layers = [
        layers.Input(shape=(100, 100, 3)),
        layers.Rescaling(1.0 / 255),
    ]

    # Blok 1
    model_layers.append(
        layers.Conv2D(
            16,
            3,
            strides=2,
            padding="same",
            kernel_regularizer=reg_val,
            use_bias=not bn_val,
        )
    )
    if bn_val:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation("relu"))
    if drop_val > 0.0:
        model_layers.append(layers.Dropout(drop_val, seed=GLOBAL_SEED))
    model_layers.append(layers.MaxPooling2D(2))

    # Blok 2
    model_layers.append(
        layers.Conv2D(
            32, 3, padding="same", kernel_regularizer=reg_val, use_bias=not bn_val
        )
    )
    if bn_val:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation("relu"))
    if drop_val > 0.0:
        model_layers.append(layers.Dropout(drop_val, seed=GLOBAL_SEED))
    model_layers.append(layers.MaxPooling2D(2))

    # Blok 3
    model_layers.append(
        layers.Conv2D(
            64, 3, padding="same", kernel_regularizer=reg_val, use_bias=not bn_val
        )
    )
    if bn_val:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation("relu"))
    if drop_val > 0.0:
        model_layers.append(layers.Dropout(drop_val, seed=GLOBAL_SEED))
    model_layers.append(layers.MaxPooling2D(2))

    # Ciężki klasyfikator gęsty z warstwą Flatten
    model_layers.append(layers.Flatten())

    # Potężna warstwa ukryta (512 neuronów)
    model_layers.append(
        layers.Dense(512, kernel_regularizer=reg_val, use_bias=not bn_val)
    )
    if bn_val:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation("relu"))
    if drop_val > 0.0:
        model_layers.append(layers.Dropout(drop_val, seed=GLOBAL_SEED))

    # Warstwa wyjściowa
    model_layers.append(layers.Dense(1, activation=None))

    model = keras.Sequential(model_layers, name=model_name)

    callbacks = [
        EarlyStopping(
            monitor="val_pure_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    # Kompilacja ze standardowym uczeniem 0.001
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            keras.losses.BinaryCrossentropy(from_logits=True, name="pure_loss"),
        ],
    )

    print(f" -> Trenowanie: {model_name}...")

    # Trening na Twoich potokach danych
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,
    )

    # Zapisanie do dedykowanego słownika zbiorczego
    histories_big_flatten[history_key] = history

results_big_flatten_data = []

for key, history in histories_big_flatten.items():
    val_pure_loss_history = history.history["val_pure_loss"]
    best_epoch_idx = np.argmin(val_pure_loss_history)

    # Wyciągamy informacje o konfiguracji na podstawie nazwy klucza
    parts = key.split("_")
    reg_type = parts[2] if "big_flatten" in key and len(parts) > 2 else "none"

    results_big_flatten_data.append(
        {
            "Model Key": key,
            "Liczba epok": len(val_pure_loss_history),
            "Best Epoch": best_epoch_idx + 1,
            "Train Accuracy": history.history["accuracy"][best_epoch_idx],
            "Val Accuracy": history.history["val_accuracy"][best_epoch_idx],
            "Train Pure Loss": history.history["pure_loss"][best_epoch_idx],
            "Val Pure Loss": history.history["val_pure_loss"][best_epoch_idx],
        }
    )

df_big_flatten_results = pd.DataFrame(results_big_flatten_data)
df_big_flatten_results = df_big_flatten_results.sort_values(
    by="Val Accuracy", ascending=False
).reset_index(drop=True)

print(
    "\n"
    + "=" * 60
    + "\nKOŃCOWA TABELA WYNIKÓW (Big CNN Flatten - 24 Warianty):\n"
    + "=" * 60
)
print(df_big_flatten_results.to_string(index=False))

# Zapis do pliku CSV na wypadek konieczności użycia w Excelu lub Wordzie
df_big_flatten_results.to_csv("big_flatten_grid_search_results.csv", index=False)

def make_model_residual(input_shape=(100, 100, 3), l2_val=0.0001):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation=None)(x)

    return keras.Model(inputs, outputs)

model_residual = make_model_residual(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_residual.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_residual = model_residual.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def make_model_big_cnn_flatten_aug(input_shape=(100, 100, 3)):
    model_layers = [
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Rescaling(1.0 / 255)
    ]

    model_layers.append(layers.Conv2D(16, 3, strides=2, padding="same"))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Conv2D(32, 3, padding="same"))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Conv2D(64, 3, padding="same"))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Flatten())
    model_layers.append(layers.Dense(512))
    model_layers.append(layers.Activation("relu"))

    model_layers.append(layers.Dense(1, activation=None))

    return keras.Sequential(model_layers)

model_big_cnn_flatten_aug = make_model_big_cnn_flatten_aug(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_big_cnn_flatten_aug.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_big_cnn_flatten_aug = model_big_cnn_flatten_aug.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

def make_model_big_flatten_regularized_aug(input_shape=(100, 100, 3)):
    reg_val = regularizers.l2(0.001)

    model_layers = [
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Rescaling(1.0 / 255)
    ]

    model_layers.append(layers.Conv2D(16, 3, strides=2, padding="same", kernel_regularizer=reg_val))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.Dropout(0.2))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg_val))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.Dropout(0.2))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg_val))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.Dropout(0.2))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Flatten())
    model_layers.append(layers.Dense(512, kernel_regularizer=reg_val))
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.Dropout(0.2))

    model_layers.append(layers.Dense(1, activation=None))

    return keras.Sequential(model_layers)

model_big_flatten_regularized_aug = make_model_big_flatten_regularized_aug(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_big_flatten_regularized_aug.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_big_flatten_regularized_aug = model_big_flatten_regularized_aug.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

def make_model_big_flatten_regularized_aug_bn(input_shape=(100, 100, 3)):
    reg_val = regularizers.l2(0.001)

    model_layers = [
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Rescaling(1.0 / 255)
    ]

    model_layers.append(layers.Conv2D(16, 3, strides=2, padding="same", kernel_regularizer=reg_val))
    model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.Dropout(0.2))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg_val))
    model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.Dropout(0.2))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg_val))
    model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.Dropout(0.2))
    model_layers.append(layers.MaxPooling2D(2))

    model_layers.append(layers.Flatten())
    model_layers.append(layers.Dense(512, kernel_regularizer=reg_val))
    model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation("relu"))
    model_layers.append(layers.Dropout(0.2))

    model_layers.append(layers.Dense(1, activation=None))

    return keras.Sequential(model_layers)

model_big_flatten_regularized_aug_bn = make_model_big_flatten_regularized_aug_bn(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_big_flatten_regularized_aug_bn.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_big_flatten_regularized_aug_bn = model_big_flatten_regularized_aug_bn.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

def make_model_residual_drop_02(input_shape=(100, 100, 3)):

    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation=None)(x)

    return keras.Model(inputs, outputs)

model_residual_drop_02 = make_model_residual_drop_02(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_residual_drop_02.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_residual_drop_02 = model_residual_drop_02.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

def make_model_residual_drop_05(input_shape=(100, 100, 3)):

    inputs = keras.Input(shape=input_shape)
    x = inputs
    # x = data_augmentation(x)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation=None)(x)

    return keras.Model(inputs, outputs)

model_residual_drop_05 = make_model_residual_drop_05(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_residual_drop_05.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_residual_drop_05 = model_residual_drop_05.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

def make_model_residual_drop_05_bn_aug(input_shape=(100, 100, 3)):

    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = data_augmentation(x)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation=None)(x)

    return keras.Model(inputs, outputs)

model_residual_drop_05_bn_aug = make_model_residual_drop_05_bn_aug(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_residual_drop_05_bn_aug.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_residual_drop_05_bn_aug = model_residual_drop_05_bn_aug.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)

def make_model_residual_solo_aug(input_shape=(100, 100, 3)):

    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = data_augmentation(x)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation=None)(x)

    return keras.Model(inputs, outputs)

model_residual_solo_aug = make_model_residual_solo_aug(input_shape=image_size + (3,))

callbacks = [
    EarlyStopping(monitor="val_pure_loss", patience=10, restore_best_weights=True, verbose=1)
]

model_residual_solo_aug.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.losses.BinaryCrossentropy(name="pure_loss", from_logits=True)]
)

histories_residual_solo_aug = model_residual_solo_aug.fit(
    train_ds,
    epochs=150,
    callbacks=callbacks,
    validation_data=val_ds
)


histories = {}
histories['residual'] = histories_residual
histories['drop_02'] = histories_residual_drop_02
histories['drop_05'] = histories_residual_drop_05
histories['drop_05_bn_aug'] = histories_residual_drop_05_bn_aug
histories['solo_aug'] = histories_residual_solo_aug

plotter = tfdocs.plots.HistoryPlotter(metric = 'pure_loss', smoothing_std=10)
plotter.plot(histories)








