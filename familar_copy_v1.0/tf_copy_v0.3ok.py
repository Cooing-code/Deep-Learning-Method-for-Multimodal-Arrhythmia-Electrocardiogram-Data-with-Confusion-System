import os
import time
import numpy as np
import scipy
from scipy import signal
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback  # 新增进度条库

# ==================== 全局参数（按论文配置） ====================
# 数据参数
DATA_PATH = "/home/cooing/ECG-Classification/git/ecg-classification-cnn/s2s_mitbih_aami.mat"  # 数据路径错误将导致无法加载训练数据
CLASSES_NAME = ["F", "N", "S", "V", "Q"]          # 类别标签定义错误会直接影响分类结果的准确性
SIGNAL_LENGTH = 280                               # 信号长度需与预处理对齐，过长会引入噪声，过短会丢失关键波形信息
TEST_SIZE = 0.2                                   # 测试集比例过小会导致评估不可靠，过大会减少训练数据量
RANDOM_STATE = 114514                                 # 固定随机种子确保实验可复现性，不同种子可能导致结果波动
TRAIN_ONLY_N = 0

# STFT参数（按论文描述补充）
STFT_PARAMS = {
    'nperseg': 256,       # 窗口长度：值越大频率分辨率越高，但时间分辨率降低（适合捕捉低频特征）
    'noverlap': 128,      # 重叠长度：50%重叠平衡计算效率与时域连续性，过高增加计算量，过低丢失瞬态特征
    'nfft': 512           # FFT点数：值越大频率精度越高，但计算量增加（高频段分析更精细）
}

# Transformer参数（论文Table IV配置）
NUM_ENCODER_LAYERS = 4    # 编码器层数：层数增加提升模型容量，但易过拟合且训练耗时（需更多数据支撑）
NUM_HEADS = 8             # 注意力头数：多头捕捉多样化特征关系，头数过多增加冗余计算（需平衡复杂度）
EMBED_DIM = 128           # 嵌入维度：维度越高特征表达能力越强，但参数量和内存占用显著增加
FF_DIM = 512              # 前馈网络维度：扩大网络宽度增强非线性拟合能力，可能需配合正则化防过拟合
DROPOUT_RATE = 0.1        # 丢弃率：较低值对模型约束小，较高值增强泛化但可能阻碍特征学习
NUM_FUZZY_RULES = 32      # 模糊规则数：规则越多系统越复杂，解释性下降但分类精度可能提升

# 训练参数（论文Section IV.A）
EPOCHS = 1                # 训练轮次：过少导致欠拟合，过多引发过拟合（需配合早停机制）  
BATCH_SIZE = 64           # 批量大小：较大值加速训练但占用显存，较小值增强泛化但收敛慢
PATIENCE = 10             # 早停耐心值：控制验证集性能下降的容忍轮次，避免无效训练
# ==================== Transformer编码器====================
class Transformer2DEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        # 论文使用的多头注意力机制
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim//num_heads,
            dropout=dropout_rate
        )
        # 使用ReLU激活函数（论文公式FFN(x) = max(0, xW1 + b1)W2 + b2）
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),  # ReLU激活
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# ==================== 神经模糊系统（保持不变） ====================
class NeuroFuzzySystem(layers.Layer):
    def __init__(self, num_rules=NUM_FUZZY_RULES):
        super().__init__(name="neuro_fuzzy_system")
        self.num_rules = num_rules
        
        self.mu = self.add_weight(shape=(num_rules, 1), initializer='random_normal', name='mu_centers')
        self.sigma = self.add_weight(shape=(num_rules, 1), initializer='ones', name='mu_sigmas')
        self.rule_weights = self.add_weight(shape=(num_rules, len(CLASSES_NAME)), initializer='random_normal', name='rule_weights')
        
    def call(self, inputs):
        expanded_inputs = tf.expand_dims(inputs, axis=1)
        mu = tf.expand_dims(self.mu, axis=0)
        sigma = tf.expand_dims(self.sigma, axis=0)
        
        membership = tf.exp(-0.5 * tf.square((expanded_inputs - mu) / (sigma + 1e-6)))
        rule_activation = tf.reduce_min(membership, axis=-1)
        
        weighted_outputs = tf.matmul(rule_activation, self.rule_weights)
        total_activation = tf.reduce_sum(rule_activation, axis=-1, keepdims=True) + 1e-6
        defuzzied_output = weighted_outputs / total_activation
        return defuzzied_output

# ==================== 模型构建（添加训练进度可视化） ====================
class TrainingVisualizer(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_progress = None
        self.batch_progress = None
    
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.total_epochs}")
        self.epoch_progress = tf.keras.utils.Progbar(
            self.params['steps'], 
            stateful_metrics=['loss', 'accuracy']
        )
    
    def on_batch_end(self, batch, logs=None):
        self.epoch_progress.update(batch + 1, values=[
            ('loss', logs['loss']),
            ('acc', logs['accuracy'])
        ])
    
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 'N/A')
        val_acc = logs.get('val_accuracy', 'N/A')
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def build_transformer_model(ecg_shape, stft_shape):
    ecg_input = tf.keras.Input(shape=ecg_shape)
    stft_input = tf.keras.Input(shape=stft_shape)
    
    # ECG分支
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(ecg_input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(EMBED_DIM)(x)
    
    # STFT分支
    y = layers.Conv2D(64, (3,3), padding='same', activation='relu')(stft_input)
    y = layers.MaxPooling2D((2,2))(y)
    y = layers.Conv2D(EMBED_DIM, (3,3), padding='same', activation='relu')(y)
    y = layers.Reshape((-1, EMBED_DIM))(y)
    
    for _ in range(NUM_ENCODER_LAYERS):
        y = Transformer2DEncoder(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)(y)
    y = layers.GlobalAveragePooling1D()(y)
    
    merged = layers.concatenate([x, y])
    merged = layers.Dense(256, activation='relu')(merged)
    
    fuzzy_output = NeuroFuzzySystem()(merged)
    outputs = layers.Dense(len(CLASSES_NAME), activation='softmax')(fuzzy_output)
    
    model = tf.keras.Model(inputs=[ecg_input, stft_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==================== 数据处理函数 ====================
def load_and_preprocess_data():
    start_time = time.time()
    print("\n[1/5] 正在加载数据...")
    
    mat_data = spio.loadmat(DATA_PATH)
    samples = mat_data["s2s_mitbih"]
    values = samples[0]["seg_values"]
    labels = samples[0]["seg_labels"]
    
    ecg_signals = []
    annotations = []
    
    print("[2/5] 处理ECG信号...")
    for item in values:
        for itm in item:
            raw_signal = itm[0].flatten()
            normalized = np.interp(raw_signal, (raw_signal.min(), raw_signal.max()), (0, 1))
            ecg_signals.append(normalized)
    
    print("[3/5] 处理标签...")
    for item in labels:
        for label_array in item:
            for label in label_array:
                annotations.append(CLASSES_NAME.index(str(label)))
    
    ecg_signals = np.array(ecg_signals).reshape(-1, SIGNAL_LENGTH)
    annotations = np.array(annotations)
    
    print("[4/5] STFT处理...")
    stft_features = []
    for sig in ecg_signals:
        _, _, Zxx = signal.stft(sig, **STFT_PARAMS)
        stft_features.append(np.abs(Zxx)[..., np.newaxis])
    
    indices = np.arange(len(ecg_signals))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, annotations, test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, stratify=annotations
    )
    
    # 原始ECG信号作为一维输入
    X_ecg_train = ecg_signals[idx_train][..., np.newaxis]
    X_ecg_test = ecg_signals[idx_test][..., np.newaxis]
    
    # STFT频谱作为二维输入
    X_stft_train = np.array(stft_features)[idx_train]
    X_stft_test = np.array(stft_features)[idx_test]

    # 新增：过滤训练数据（当TRAIN_ONLY_N=1时）  <--- 核心修改点
    if TRAIN_ONLY_N == 1:
        n_class_idx = CLASSES_NAME.index("N")
        train_mask = (y_train == n_class_idx)
        X_ecg_train = X_ecg_train[train_mask]
        X_stft_train = X_stft_train[train_mask]
        y_train = y_train[train_mask]
        print(f"\n! 已启用N类专用训练模式，训练样本数: {len(X_ecg_train)}")
    
    print(f"\n数据处理完成! 总耗时: {time.time()-start_time:.1f}秒")
    print(f"训练集样本数: {len(X_ecg_train)}")
    print(f"测试集样本数: {len(X_ecg_test)}")
    
    return (X_ecg_train, X_stft_train), (X_ecg_test, X_stft_test), y_train, y_test
# ==================== 主程序（添加可视化） ====================
if __name__ == "__main__":
    # 加载数据
    (X_ecg_train, X_stft_train), (X_ecg_test, X_stft_test), y_train, y_test = load_and_preprocess_data()
    
    # 构建模型
    model = build_transformer_model(
        ecg_shape=X_ecg_train.shape[1:],
        stft_shape=X_stft_train.shape[1:]
    )
    model.summary()
    
    # 训练配置
    early_stop = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1)
    visualizer = TrainingVisualizer(total_epochs=EPOCHS)  # 自定义可视化回调
    
    # 模型训练（添加进度条）
    print("\n开始训练...")
    history = model.fit(
        [X_ecg_train, X_stft_train], y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, visualizer, TqdmCallback(verbose=0)],  # 添加进度条
        verbose=0  # 禁用默认输出
    )
    
    # 训练曲线可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()
    
    # 模型评估
    print("\n模型评估结果:")
    y_pred = model.predict([X_ecg_test, X_stft_test])
    y_pred = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_test, y_pred, target_names=CLASSES_NAME))
    
    # 混淆矩阵
    plt.figure(figsize=(10,8))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=CLASSES_NAME, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')