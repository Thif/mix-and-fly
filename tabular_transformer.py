import keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import Metric
import random


from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

@keras.saving.register_keras_serializable()
class MapeThreshold(Metric):
    def __init__(self, threshold=0.5, name='mape_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.mape_mean = self.add_weight(name='mape_mean', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Clip predictions based on the threshold
        y_pred_clipped = tf.where(tf.abs(y_pred) < self.threshold, 0.0, y_pred)
        epsilon = tf.keras.backend.epsilon()

        # Calculate MAPE
        mape = tf.abs(y_true - y_pred_clipped) / tf.maximum(epsilon, tf.abs(y_true))
        
        # Update the mean MAPE
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.mape_mean.dtype)
            mape = tf.multiply(mape, sample_weight)
        
        # Update the total MAPE and count
        self.mape_mean.assign_add(tf.reduce_sum(mape))
        self.count.assign_add(tf.reduce_sum(sample_weight) if sample_weight is not None else tf.cast(tf.shape(y_true)[0], self.count.dtype))

    def result(self):
        return self.mape_mean / self.count

    def reset_states(self):
        # Reset the state of the metric at the start of each epoch
        self.mape_mean.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
class MaxMapeThreshold(Metric):
    def __init__(self, threshold=0.5, name='max_mape_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.mape_max = self.add_weight(name='mape_mean', initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        # Clip predictions based on the threshold
        y_pred_clipped = tf.where(tf.abs(y_pred) < self.threshold, 0.0, y_pred)
        epsilon = tf.keras.backend.epsilon()

        # Calculate MAPE
        mape = tf.abs(y_true - y_pred_clipped) / tf.maximum(epsilon, tf.abs(y_true))
        
        # Update the mean MAPE

        
        # Update the total MAPE and count
        self.mape_max.assign_add(tf.reduce_max(mape))

    def result(self):
        return self.mape_max

    def reset_states(self):
        # Reset the state of the metric at the start of each epoch
        self.mape_max.assign(0.0)


    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Define Positional Encoding layer
@keras.saving.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000, trainable=True, seed_value=42,dtype=tf.float32, **kwargs):
        super(PositionalEncoding, self).__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.d_model = d_model
        self.max_len = max_len

        #for reproducibility
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

        # Precompute the positional encodings
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        self.pos_encoding = tf.constant(pos_encoding, dtype=dtype)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]  # Get the sequence length dynamically
        return inputs + self.pos_encoding[:seq_len, :]

# Define Attention Pooling layer
@keras.saving.register_keras_serializable()
class AttentionPooling(layers.Layer):
    def __init__(self, d_model, trainable=True, dtype=tf.float32, seed_value=42, **kwargs):
        super(AttentionPooling, self).__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.d_model = d_model  # Store the model dimension
        self.attention = None  # Initialize attention weights to None

        # For reproducibility
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

    def build(self, input_shape):
        # Create the attention weights
        self.attention = self.add_weight(
            shape=(input_shape[-1], 1),  # Shape: (d_model, 1)
            initializer='random_normal',
            trainable=True,
            name='attention_weights'
        )

    def call(self, inputs):
        # inputs: (batch_size, sequence_length, d_model)
        # Calculate attention weights
        weights = tf.nn.softmax(tf.matmul(inputs, self.attention), axis=1)  # Shape: (batch_size, sequence_length, 1)
        pooled = tf.reduce_sum(weights * inputs, axis=1)  # Weighted sum across the sequence dimension

        return pooled


# Define the Transformer Regressor model
def build_transformer_regressor(input_dim, d_model, nhead, num_layers, output_dim, size_mult,max_len=5000):


    inputs = layers.Input(shape=(None, input_dim))  # Input shape: (sequence_length, input_dim)

    # Linear projection to d_model
    x = layers.Dense(d_model)(inputs)

    # Add positional encoding
    x = PositionalEncoding(d_model, max_len)(x)

    # Transformer Encoder layers
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=0.1)(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feedforward network
        ffn_output = layers.Dense(d_model * size_mult, activation="relu")(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Attention pooling
    x = AttentionPooling(d_model)(x)

    # Feedforward network for regression
    outputs = layers.Dense(output_dim)(x)

    # Build the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
    
def get_sample_weight(y,alpha=0.0001):

    sample_weight = np.exp(-alpha*y**2)  # Shape: (N,)
    sample_weight/=sum(sample_weight)

    sample_weight = np.expand_dims(sample_weight, axis=-1) 

    return sample_weight


@keras.saving.register_keras_serializable()
def max_ae(y_true, y_pred):
    # Calculate the mean absolute error in the scaled space
    return tf.reduce_max(tf.abs(y_pred - y_true))

@keras.saving.register_keras_serializable()
def custom_mape(y_true, y_pred,t):
    """
    Custom Mean Absolute Percentage Error (MAPE) loss function that minimizes both mean and max MAPE.
    
    Parameters:
    y_true: Tensor of true values.
    y_pred: Tensor of predicted values.
    alpha: Weighting factor to balance mean and max MAPE.
    
    Returns:
    Tensor: Combined loss value.
    """
    y_pred_clipped=tf.where(tf.abs(y_pred)<t,0.0,y_pred)
    epsilon = tf.keras.backend.epsilon()

    # Calculate MAPE
    mape = tf.abs(y_true - y_pred_clipped) / tf.maximum(epsilon, tf.abs(y_true))
    
    # Calculate mean and max MAPE
    mape_mean = tf.reduce_mean(mape)

    
    return mape_mean

@keras.saving.register_keras_serializable()
def custom_max_mape(y_true, y_pred,t):
    """
    Custom Mean Absolute Percentage Error (MAPE) loss function that minimizes both mean and max MAPE.
    
    Parameters:
    y_true: Tensor of true values.
    y_pred: Tensor of predicted values.
    alpha: Weighting factor to balance mean and max MAPE.
    
    Returns:
    Tensor: Combined loss value.
    """
    y_pred_clipped=tf.where(tf.abs(y_pred)<t,0.0,y_pred)
    epsilon = tf.keras.backend.epsilon()

    # Calculate MAPE
    mape = tf.abs(y_true - y_pred_clipped) / tf.maximum(epsilon, tf.abs(y_true))
    
    # Calculate mean and max MAPE
    mape_max = tf.reduce_max(mape)

    
    return mape_max

@keras.saving.register_keras_serializable()
def custom_mape_with_param(t):
    @keras.saving.register_keras_serializable()
    def mape_metric(y_true, y_pred):
        return custom_mape(y_true, y_pred, t)
    return mape_metric

@keras.saving.register_keras_serializable()
def custom_max_mape_with_param(t):
    @keras.saving.register_keras_serializable()
    def max_mape_metric(y_true, y_pred):
        return custom_max_mape(y_true, y_pred, t)
    return max_mape_metric

class TransformerModel():
    def __init__(self,pred_col,seed_value=42):
        # Define hyperparameters
        self.pred_col=pred_col
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)


    def scale_inputs(self,X,fit,random_state=42):
        # Split into training and validation sets (80% train, 20% validation)
        
        if fit:
            self.sc_in=MinMaxScaler()
            X_scaled = self.sc_in.fit_transform(X)
        else:
            X_scaled = self.sc_in.transform(X)

        # Reshape X to the correct shape (2000, 5, 11)
        X_scaled = X_scaled.reshape(len(X_scaled), 11, 5)  # Adjust the shape based on your data

        return X_scaled

    def scale_outputs(self,y,fit,random_state=42):

        if fit:
            self.sc_out=no_op_scaler = FunctionTransformer(func=lambda x: x.values, validate=False)#MinMaxScaler((-1,1))
            y_scaled=self.sc_out.fit_transform(y)
        else:
            y_scaled=self.sc_out.transform(y)

        y_scaled = y_scaled.reshape(len(y_scaled), -1) 

        return y_scaled



    def fit(self,X,y,sample_weight,validation_data,input_dim=5,d_model=64,nhead=4,num_layers=2,num_epochs=1000,batch_size=32,learning_rate=0.001,size_mult=2,cross_validation=True,threshold=0.4):


        output_dim = 1
        
        # Build the model
        self.model = build_transformer_regressor(input_dim, d_model, nhead, num_layers, output_dim,size_mult)
        

        # Compile the model
        self.model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                    loss="mae",
                    metrics=[MapeThreshold(threshold),MaxMapeThreshold(threshold),max_ae])

        # Define callbacks
        if validation_data is not None:
            checkpoint = callbacks.ModelCheckpoint(
            f'./models/best_model_{self.pred_col}.keras',  # Path to save the model
            monitor='val_mape_metric',  # Metric to monitor
            save_best_only=True,  # Save only the best model
            mode='min',  # Use 'min' since lower MAPE is better
            verbose=1  # Verbosity mode
            )
            early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10)

            callbacks_list=[early_stopping, reduce_lr,checkpoint]
        else:
            #use only learning rate schedule to stop when no validation is provided ( other option would be number of epochs )
            reduce_lr = callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10)
            callbacks_list=[reduce_lr]
            
        # Train the model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            sample_weight=sample_weight,  # Use sample weights with the correct shape
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        return None

    def get_pred(self,best_model,X_scaled,threshold):




        c=self.pred_col

        # Get predictions and inverse-transform them to the original scale
        predictions = self.sc_out.inverse_transform(best_model.predict(X_scaled))
        df_res = pd.DataFrame(predictions, columns=["pred_" + c])

        predictions=np.where(np.abs(predictions)<threshold,0,predictions)
        return predictions

    def get_scores(self,best_model,X_scaled,y_scaled,threshold):
        c=self.pred_col
        predictions=self.get_pred(best_model,X_scaled,threshold)

        df_res = pd.DataFrame(predictions, columns=["pred_" + c])
        # Inverse-transform the true values to the original scale
        y_val_original = self.sc_out.inverse_transform(y_scaled)
        df_res = pd.concat([df_res, pd.DataFrame(y_val_original, columns=[c]).reset_index(drop=True)], axis=1)

        # Calculate MAPE for each column
        epsilon = 1e-8  # Small value to avoid division by zero

        
        df_res["mape_" + c] = np.abs(df_res[c] - df_res["pred_" + c]) / np.maximum(epsilon,np.abs(df_res[c]))
        print(df_res["mape_" + c].describe())
        return get_score(df_res["mape_" + c] .quantile(0.95)),get_score(df_res["mape_" + c] .median()),get_score(df_res["mape_" + c] .mean()),get_score(df_res["mape_" + c] .min())
        
    def plot_history(self):
        # Plotting the training and validation loss
        plt.figure(figsize=(12, 6))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot MAE if it was tracked
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mape'], label='Training MAPE')
        plt.plot(history.history['val_mape'], label='Validation MAPE')
        plt.title('Mean Absolute Percentage Error Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.legend()

        plt.tight_layout()
        plt.show()

def get_score(cost):
    return max(10,100-(90*cost)/2.72)

