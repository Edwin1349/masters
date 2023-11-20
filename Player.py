import os
import CombinedModel
import numpy as np
from PyQt5.QtCore import QUrl, Qt, QBuffer, QIODevice
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSlider
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import tempfile
import soundfile as sf
import librosa

HOP_SIZE = 256
WINDOW_SIZE = 1024
SAMPLING_RATE = 22050

class AudioPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.media_players = list()
        self.sliders = list()
        self.labels = list()
        self.mute_buttons = list()

        self.init_ui()
        self.controlled_manually = False
        self.duration = None

    def init_ui(self):
        self.setWindowTitle("Audio Player")
        self.setGeometry(100, 100, 400, 200)

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.add_button = QPushButton("Add File")

        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.stop_button.clicked.connect(self.stop)
        self.add_button.clicked.connect(self.add_file)

        audio_layouts = list()

        for i in range(4):
            progress_label = QLabel("00:00 / 00:00", self)
            progress_label.setProperty("channel_id", i)
            self.labels.append(progress_label)

            progress_slider = QSlider(Qt.Horizontal, self)
            progress_slider.setGeometry(50, 70, 320, 30)
            progress_slider.sliderPressed.connect(self.slider_pressed)
            progress_slider.sliderMoved.connect(self.set_position)
            progress_slider.sliderReleased.connect(self.slider_released)
            progress_slider.setProperty("channel_id", i)
            self.sliders.append(progress_slider)

            mute_button = QPushButton("Mute " + CombinedModel.Channel(i + 1).name)
            mute_button.setProperty("channel_id", i)
            mute_button.clicked.connect(self.mute)
            self.mute_buttons.append(mute_button)

            audio_layout = QHBoxLayout()
            audio_layout.addWidget(progress_slider)
            audio_layout.addWidget(progress_label)
            audio_layout.addWidget(mute_button)
            audio_layouts.append(audio_layout)

            media_player = QMediaPlayer()
            media_player.positionChanged.connect(self.update_position)
            media_player.setProperty("channel_id", i)
            self.media_players.append(media_player)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.add_button)

        layout = QVBoxLayout()
        for i in range(4):
            layout.addLayout(audio_layouts[i])
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def slider_pressed(self):
        self.controlled_manually = True
        self.pause()

    def slider_released(self):
        self.play()
        self.controlled_manually = False

    def set_position(self, position):
        if self.controlled_manually:
            sender = self.sender()
            position_ms = int((position * self.duration) / 99)
            for audio in self.media_players:
                audio.setPosition(position_ms)
                for slider in self.sliders:
                    if slider is not sender:
                        slider.setValue(position)

    def update_duration(self):
        minutes = self.duration // 60000
        seconds = (self.duration // 1000) % 60
        for label in self.labels:
            label.setText(f"00:00 / {minutes:02}:{seconds:02}")

    def update_position(self, position):
        if not self.controlled_manually:
            sender = self.sender()
            minutes = position // 60000
            seconds = (position // 1000) % 60
            self.sliders[sender.property("channel_id")].setValue(int(position * 99 / self.duration))
            self.labels[sender.property("channel_id")].setText(f"{minutes:02}:{seconds:02} / {self.duration // 60000:02}:{(self.duration // 1000) % 60:02}")

    def mute(self):
        sender = self.sender()
        if self.media_players[sender.property("channel_id")].volume() > 0:
            self.media_players[sender.property("channel_id")].setVolume(0)
            sender.setText('Muted')
        elif self.media_players[sender.property("channel_id")].volume() == 0:
            self.media_players[sender.property("channel_id")].setVolume(100)
            sender.setText('Mute ' + CombinedModel.Channel(sender.property("channel_id") + 1).name)

    def pause(self):
        for audio in self.media_players:
            audio.pause()

    def stop(self):
        for audio in self.media_players:
            audio.stop()

    def play(self):
        for audio in self.media_players:
            audio.play()

    def add_file(self):
        self.stop()

        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", os.path.expanduser("~"),
                                                   "Audio Files (*.mp4 *.mp3 *.wav)")

        if file_path:
            # Read the audio file using librosa with mono=False to get multi-channel data
            combined_model = CombinedModel.create_combineModel()
            dataloader = CombinedModel.music_to_dataloader(file_path)
            mixture, result = CombinedModel.predict(combined_model, dataloader)
            print("Separation finished")
            audio_data = np.array_split(result, 4, axis=1)
            audio_data = list(
                map(lambda channel: librosa.istft(mixture.T.astype(np.complex64) * channel.T.astype(np.complex64),
                                                  hop_length=HOP_SIZE, n_fft=WINDOW_SIZE), audio_data))
            self.duration = int(len(audio_data[0]) * 1000 / SAMPLING_RATE)

            self.update_duration()
            # audio_data, sample_rate = librosa.load(file_path, mono=False)
            temp_dir = tempfile.mkdtemp()
            temp_files = []
            for i in range(len(audio_data)):
                # convert arrat to bytes
                temp_file_path = os.path.join(temp_dir, f"channel_{i}.wav")
                print(temp_file_path)
                sf.write(temp_file_path, audio_data[i], SAMPLING_RATE)
                temp_files.append(temp_file_path)

            media_contents = [QMediaContent(QUrl.fromLocalFile(file_path)) for file_path in temp_files]
            for i in range(len(audio_data)):
                self.media_players[i].setMedia(media_contents[i])

if __name__ == "__main__":
    app = QApplication([])
    player = AudioPlayer()
    player.show()
    app.exec_()
