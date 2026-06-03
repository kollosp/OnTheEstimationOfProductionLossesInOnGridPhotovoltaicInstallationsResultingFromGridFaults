from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Input, Flatten, Dropout, Conv1D, BatchNormalization,
    LeakyReLU, GRU, Bidirectional, GlobalAveragePooling1D, Add, 
    Concatenate, MaxPooling1D, TimeDistributed, LayerNormalization,
    PReLU, SpatialDropout1D, MultiHeadAttention, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


class KerasWrapper:
    def __init__(self, model):
        self.model = model
        self.model.compile(
            loss="mean_absolute_error",
            optimizer="adam",
            metrics=["mean_squared_error"],
            steps_per_execution=10,
        )

    def fit(self, *args, **kwargs):
        kwargs["epochs"] = kwargs.get("epochs", 30)
        kwargs["batch_size"] = kwargs.get("batch_size", 256)

        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

class ModelMLP(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential([
            Input(shape=input_shape),
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)
    
    def __str__(self):
        return "MLP"

class ModelLSTM(KerasWrapper):
    def __init__(self, input_shape, output_shape):
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(16, return_sequences=False),
                Dense(16, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(8, activation='relu'),
                Dense(output_shape),
            ]
        )
        super().__init__(model)

    def __str__(self):
        return "LSTM"

class ModelCNN(KerasWrapper):
    def __init__(self, input_shape, output_shape, complexity):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(int(2*complexity), 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(int(complexity), 3, activation='relu', padding='same'),
            Flatten(),
            Dense(int(complexity), activation='relu'),
            Dense(output_shape)
        ])
        super().__init__(model)

    def __str__(self):
        return "CNN"

###