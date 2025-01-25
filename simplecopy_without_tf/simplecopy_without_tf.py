import os
import time
import pywt
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

# ==================== 全局可调参数 ====================
# 数据参数
DATA_PATH = "/home/cooing/ECG-Classification/git/ecg-classification-cnn/s2s_mitbih_aami.mat"
CLASSES_NAME = ["F", "N", "S", "V", "Q"]  # 分类标签
SIGNAL_LENGTH = 280                       # 每个ECG信号的长度
TEST_SIZE = 0.1                           # 测试集比例
RANDOM_STATE = 42                         # 随机种子

# 小波变换参数
WAVELET_TYPE = 'db2'                      # 使用的小波类型

# STFT参数
STFT_PARAMS = {
    'nperseg': 64,                        # 窗口长度
    'noverlap': 48,                       # 重叠采样数(必须小于nperseg)
    'nfft': 128                           # FFT点数
}

# 模型结构参数
CONV1D_FILTERS = 64                       # 一维卷积核数量
CONV2D_FILTERS = 32                       # 二维卷积核数量
LSTM_UNITS = 32                           # LSTM单元数
DENSE_UNITS = 64                          # 全连接层单元数

# 训练参数
EPOCHS = 100                               # 最大训练轮次
BATCH_SIZE = 64                           # 批量大小
VAL_SPLIT = 0.2                           # 验证集比例
LEARNING_RATE = 0.001                     # 学习率
PATIENCE = 10                             # 早停耐心值

# 进度条参数
PROGRESS_BAR_LENGTH = 30                  # 进度条长度（字符数）

# ==================== 训练监控类 ====================
class TrainingMonitor(Callback):
    def __init__(self, test_data, class_names, total_epochs):
        super().__init__()
        self.test_data = test_data
        self.class_names = class_names
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
        # 初始化历史记录
        self.history = {
            'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': []
        }
        # 初始化可视化画布
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10))
        plt.tight_layout()

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        print(f"\n训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        # 记录当前epoch指标
        self._record_metrics(logs)
        # 计算时间相关指标
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        avg_time = np.mean(self.epoch_times)
        remaining = avg_time * (self.total_epochs - epoch - 1)
        
        # 显示进度信息
        self._show_progress(epoch, epoch_time, remaining)
        # 更新可视化图表
        self._update_plots(epoch)
        plt.pause(0.1)

    def _record_metrics(self, logs):
        """记录训练指标到历史记录"""
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['accuracy'].append(logs.get('accuracy', 0))
        self.history['val_accuracy'].append(logs.get('val_accuracy', 0))

    def _show_progress(self, epoch, epoch_time, remaining):
        """显示训练进度信息"""
        progress = (epoch + 1) / self.total_epochs
        filled = int(PROGRESS_BAR_LENGTH * progress)
        bar = '[' + '=' * filled + '>' + ' ' * (PROGRESS_BAR_LENGTH - filled - 1) + ']'
        
        print(f"Epoch {epoch+1:03d}/{self.total_epochs} {bar} {progress*100:.1f}%")
        print(f" - 耗时: {epoch_time:.1f}s/epoch | 剩余预估: {remaining//60:.0f}m {remaining%60:.0f}s")
        print(f" - 训练损失: {self.history['loss'][-1]:.4f} | 验证损失: {self.history['val_loss'][-1]:.4f}")
        print(f" - 训练准确率: {self.history['accuracy'][-1]*100:.2f}% | 验证准确率: {self.history['val_accuracy'][-1]*100:.2f}%")
        print("-"*60)

    def _update_plots(self, epoch):
        """更新可视化图表"""
        # 更新损失曲线
        self.axs[0,0].clear()
        self.axs[0,0].plot(range(epoch+1), self.history['loss'], 
                         marker='o', color='tab:blue', label='Train')
        self.axs[0,0].plot(range(epoch+1), self.history['val_loss'], 
                         marker='o', color='tab:orange', linestyle='--', label='Val')
        self.axs[0,0].set_title('Loss Curve')
        self.axs[0,0].legend()
        
        # 更新准确率曲线
        self.axs[0,1].clear()
        self.axs[0,1].plot(range(epoch+1), self.history['accuracy'], 
                         marker='o', color='tab:blue', label='Train')
        self.axs[0,1].plot(range(epoch+1), self.history['val_accuracy'], 
                         marker='o', color='tab:orange', linestyle='--', label='Val')
        self.axs[0,1].set_ylim(0, 1)
        self.axs[0,1].set_title('Accuracy Curve')
        self.axs[0,1].legend()
        
        # 每3个epoch更新样本展示
        if epoch % 3 == 0:
            self._show_sample_prediction()

    def _show_sample_prediction(self):
        """展示随机样本预测结果"""
        sample_idx = np.random.randint(0, len(self.test_data[0]))
        prediction = self.model.predict(
            [self.test_data[0][sample_idx:sample_idx+1], 
             self.test_data[1][sample_idx:sample_idx+1]],
            verbose=0
        )
        pred_class = np.argmax(prediction)
        true_class = np.argmax(self.test_data[2][sample_idx]) if len(self.test_data) > 2 else -1
        
        # 绘制波形图
        self.axs[1,0].clear()
        self.axs[1,0].plot(self.test_data[0][sample_idx].squeeze())
        title = 'ECG Waveform\n'
        if true_class != -1:
            title += f'True: {self.class_names[true_class]} | '
        title += f'Pred: {self.class_names[pred_class]}'
        self.axs[1,0].set_title(title)
        
        # 绘制频谱图
        self.axs[1,1].clear()
        self.axs[1,1].imshow(self.test_data[1][sample_idx].squeeze().T, 
                           aspect='auto', 
                           origin='lower',
                           cmap='viridis')
        self.axs[1,1].set_title('STFT Spectrogram')
        
# ==================== 数据处理函数 ====================
def load_and_preprocess_data():
    start_time = time.time()
    print("\n[1/5] 正在加载数据...")
    
    # 加载MAT文件
    mat_data = spio.loadmat(DATA_PATH)
    samples = mat_data["s2s_mitbih"]
    values = samples[0]["seg_values"]
    labels = samples[0]["seg_labels"]
    
    # 计算总样本数
    num_samples = sum(item.shape[0] for item in values)
    
    # 初始化存储
    ecg_signals = []
    annotations = []
    
    print("[2/5] 处理ECG信号...")
    data_counter = 0
    for item in values:
        for itm in item:
            if data_counter >= num_samples:
                break
            # 提取并标准化信号
            raw_signal = itm[0].flatten()
            normalized = np.interp(raw_signal, (raw_signal.min(), raw_signal.max()), (0, 1))
            ecg_signals.append(normalized)
            data_counter += 1
    
    print("[3/5] 处理标签...")
    label_counter = 0
    for item in labels:
        for label_array in item:
            for label in label_array:
                if label_counter >= num_samples:
                    break
                annotations.append(CLASSES_NAME.index(str(label)))
                label_counter += 1
    
    # 转换为numpy数组
    ecg_signals = np.array(ecg_signals).reshape(-1, SIGNAL_LENGTH)
    annotations = np.array(annotations)
    
    print("[4/5] 小波变换处理...")
    wavelet_coeffs, _ = pywt.dwt(ecg_signals, WAVELET_TYPE)
    
    print("[5/5] STFT处理...")
    stft_features = []
    for sig in ecg_signals:
        _, _, Zxx = signal.stft(sig, **STFT_PARAMS)
        stft_features.append(np.abs(Zxx))
    
    # 数据划分
    indices = np.arange(len(ecg_signals))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, annotations, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=annotations
    )
    
    # 构建最终数据集
    X_wave_train = wavelet_coeffs[idx_train][..., np.newaxis]
    X_wave_test = wavelet_coeffs[idx_test][..., np.newaxis]
    X_stft_train = np.array(stft_features)[idx_train][..., np.newaxis]
    X_stft_test = np.array(stft_features)[idx_test][..., np.newaxis]
    
    print(f"\n数据处理完成! 总耗时: {time.time()-start_time:.1f}秒")
    print(f"训练集样本数: {len(X_wave_train)}")
    print(f"测试集样本数: {len(X_wave_test)}")
    
    return (X_wave_train, X_stft_train), (X_wave_test, X_stft_test), y_train, y_test

# ==================== 模型构建函数 ====================
def build_dual_model(wave_shape, stft_shape):
    # 小波分支
    wave_input = tf.keras.Input(shape=wave_shape)
    x = tf.keras.layers.Conv1D(CONV1D_FILTERS, 3, activation='relu')(wave_input)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS))(x)
    
    # STFT分支
    stft_input = tf.keras.Input(shape=stft_shape)
    y = tf.keras.layers.Conv2D(CONV2D_FILTERS, (3,3), activation='relu')(stft_input)
    y = tf.keras.layers.MaxPooling2D((2,2))(y)
    y = tf.keras.layers.Reshape((y.shape[1], -1))(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS))(y)
    
    # 特征融合
    merged = tf.keras.layers.concatenate([x, y])
    dense = tf.keras.layers.Dense(DENSE_UNITS, activation='relu')(merged)
    outputs = tf.keras.layers.Dense(len(CLASSES_NAME), activation='softmax')(dense)
    
    model = tf.keras.Model(inputs=[wave_input, stft_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 数据预处理
    (X_wave_train, X_stft_train), (X_wave_test, X_stft_test), y_train, y_test = load_and_preprocess_data()
    
    # 模型构建
    model = build_dual_model(
        wave_shape=X_wave_train.shape[1:],
        stft_shape=X_stft_train.shape[1:]
    )
    model.summary()
    
    # 训练配置
    monitor = TrainingMonitor(
        test_data=(X_wave_test, X_stft_test, tf.keras.utils.to_categorical(y_test)),
        class_names=CLASSES_NAME,
        total_epochs=EPOCHS
    )
    
    # 模型训练
    print("\n开始模型训练...")
    history = model.fit(
        [X_wave_train, X_stft_train], y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[monitor, EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
        verbose=0
    )
    
    # 模型评估
    print("\n模型评估结果:")
    y_pred = model.predict([X_wave_test, X_stft_test])
    y_pred = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_test, y_pred, target_names=CLASSES_NAME))
    
    # 保存混淆矩阵
    plt.figure(figsize=(10,8))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=CLASSES_NAME, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()