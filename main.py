
import cv2
import logging
import sys
import time
import numpy as np
from modules import utils
from pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import gc

# cap = cv2.VideoCapture(0)

class WebCam(Pipeline):
    cap = ""
    def __init__(self, flag, video_file_path, name):
        global cap
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        super().__init__()
        #ret = self.translator_manager.load_knn_database()
        if not True:
            logging.error("KNN Sample is missing. Please record some samples before starting play mode.")
            self.notebook.select(0)
        self.flag  = flag #手話単語の判別とデータベースに登録　Trueなら判別, Falseならデータベース登録
        self.name = name
        self.word_memory = []
        self.frame_count = 0
        self.word = ""
        self.resumeList = []
        self.N = 30 #手話単語認識のフレームの閾値
        self.splitFrame = []
        self.video_loop()
       

    def record_btn_cb(self):        
        vid_res = {
            "pose_frames": np.stack(self.pose_history),
            "face_frames": np.stack(self.face_history),
            "lh_frames": np.stack(self.lh_history),
            "rh_frames": np.stack(self.rh_history),
            "n_frames": len(self.pose_history)
        }
       
        feats = self.translator_manager.get_feats(vid_res)
        if self.flag: #手話単語認識
            res_txt, resume = self.translator_manager.run_knn(feats)
            if self.word != res_txt:
                self.word = res_txt
            self.word_memory.append([self.frame_count, res_txt])
            self.resumeList.append([self.frame_count, resume])
            self.delete_front()

        else: #手話単語の特徴量登録
            self.knn_records.append(feats)
            self.save_btn_cb()


    def save_btn_cb(self):  #手話単語の特徴量をDBに保存
        gloss_name = self.name

        if (gloss_name == ""):
            logging.error("Empty gloss name.")
            return
        if (len(self.knn_records) < 0):
            logging.error("No knn record found.")
            return

        self.translator_manager.save_knn_database(gloss_name, self.knn_records)

        logging.info("database saved.")
        # clear.
        self.knn_records = []

        


    def video_loop(self):
        while True:
            self.frame_count += 1
            ret, frame = cap.read()
            if not ret:
                print("動画が終了しました")
                if self.flag == True: #手話単語認識
                    self.split_word()
                    print("手話単語認識が完了しました。")
                else: #手話単語DB登録
                    self.record_btn_cb()
                    print("データベースに登録が完了しました。")
                gc.collect()
                break
            try:
                #frame = utils.crop_utils.crop_square(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.update(frame_rgb)
                cv2.imshow("Camera", frame_rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if self.frame_count >= self.N and self.flag==True: #閾値以上の時に手話単語認識を行う
                    self.record_btn_cb()

            except AttributeError:
                print("エラーが起きました")
                break

    def plot_word_counts(self, resume_list):
    # Counting occurrences of each word
        word_counts = Counter()
        for _, words in resume_list:
            word_counts.update(words)

        # Extracting counts for specific words
        words_of_interest = ['like', 'lunch', 'morning', "evening", "food", "greeting", "banana", "I"]
        word_counts = {word: [] for word in words_of_interest}

        # Frame numbers
        frames = [item[0] for item in resume_list]

        # Counting occurrences
        for frame in resume_list:
            frame_words = frame[1]
            for word in words_of_interest:
                word_counts[word].append(np.sum(frame_words == word))
        # Plotting
        plt.figure(figsize=(10, 6))

        for word, counts in word_counts.items():
            plt.plot(frames, counts, label=word)
        
        for s_f in self.splitFrame:
            plt.vlines(s_f, 0, 5.5, color='red', linewidth=1.5, linestyle="dashed")
       
        plt.xlabel('Frame Number')
        plt.ylabel('Count')
        plt.title('Word Counts in Frames')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def split_word(self): #単語の切り目を探す
        word = self.word_memory[0][1]
        word_count = 0
        start_frame = 0
        for i, memory in enumerate(self.word_memory):
            if word == memory[1]:
                word_count += 1
                if i+1 == len(self.word_memory) and  word_count >= 8:#最終フレームの話
                    self.splitFrame.append(start_frame)
                    self.splitFrame.append(memory[0])
                    print("単語の切り目", start_frame, memory[0], word) 
            else:
                if word_count >= 8 and word != "None":
                    self.splitFrame.append(start_frame)
                    self.splitFrame.append(memory[0])
                    print("単語の切り目", start_frame, memory[0], word)

                word_count = 0
                start_frame = memory[0]
                word = memory[1]
        self.plot_word_counts(self.resumeList)


# if __name__ == "__main__":
#     app = WebCam(True, './test/おはよう/04.mp4', "like")
