import cv2
import logging
import sys
import time
import numpy as np
from gui import DemoGUI
from modules import utils
from pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# cap = cv2.VideoCapture(0)
cap = ""
class WebCam(DemoGUI, Pipeline):

    def __init__(self, flag, video_file_path, name):
        global cap
        cap = cv2.VideoCapture(video_file_path)
        super().__init__()
        self.flag  = flag #手話単語の判別とデータベースに登録　Trueなら判別, Falseならデータベース登録
        self.name = name
        self.word_memory = []
        self.frame_count = 0
        self.video_loop()
        self.record_btn_cb()
        self.word = ""
        self.resumeList = []
        
        
    
    def show_frame(self, frame_rgb):
        self.frame_rgb_canvas = frame_rgb
        self.update_canvas()

    def tab_btn_cb(self, event):
        super().tab_btn_cb(event)
        # check database before change from record mode to play mode.
        self.is_play_mode = True
        if self.is_play_mode:
            ret = self.translator_manager.load_knn_database()
            if not ret:
                logging.error("KNN Sample is missing. Please record some samples before starting play mode.")
                self.notebook.select(0)

    def record_btn_cb(self):
        super().record_btn_cb()
        
        
        if self.is_recording:
            self.reset_pipeline()
            self.is_recording = True
            return
        if len(self.pose_history) < 3:
            logging.warnning("Video too short.")
            self.reset_pipeline()
            return
        vid_res = {
            "pose_frames": np.stack(self.pose_history),
            "face_frames": np.stack(self.face_history),
            "lh_frames": np.stack(self.lh_history),
            "rh_frames": np.stack(self.rh_history),
            "n_frames": len(self.pose_history)
        }
               
        feats = self.translator_manager.get_feats(vid_res)
        # #リアルタイム用
        self.delete_front()

        self.is_play_mode = self.flag
        # Play mode: run translator.
        if self.is_play_mode:
            res_txt, resume = self.translator_manager.run_knn(feats)
            self.console_box.delete('1.0', 'end')
            self.console_box.insert('end', f"Nearest class: {res_txt}\n")
            if self.word != res_txt:
                self.word = res_txt
            self.word_memory.append([self.frame_count, res_txt])
            self.resumeList.append([self.frame_count, resume])
                

        # KNN-Record mode: save feats.
        else:    
            # raise Exception
            self.knn_records.append(feats)
            self.num_records_text.set(f"num records: {len(self.knn_records)}")
            self.save_btn_cb()

        self.is_recording = True
        # self.close_all()
    def save_btn_cb(self):
        super().save_btn_cb()

        # Read texbox entry, use as folder name.
        # gloss_name = self.name_box.get()
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
        self.num_records_text.set("num records: " + str(len(self.knn_records)))
        self.name_box.delete(0, 'end')
        


    def video_loop(self):
        self.frame_count += 1
        ret, frame = cap.read()
        
        if not ret:
            logging.error("Camera frame not available.")
            self.record_btn_cb()
            self.close_all()
        try:
            frame = utils.crop_utils.crop_square(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t1 = time.time()

            self.update(frame_rgb)

            t2 = time.time() - t1
            cv2.putText(frame_rgb, "{:.0f} ms".format(t2 * 1000), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)
            self.show_frame(frame_rgb)
            cv2.imshow("Camera", frame_rgb)

            # #リアルタイム用
            if self.frame_count >= 30:
                self.record_btn_cb()

        except AttributeError:
            self.root.destroy()
            pass
        else:
            self.root.after(1, self.video_loop)

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
        print(word_counts.items())
        # Plotting
        plt.figure(figsize=(10, 6))

        for word, counts in word_counts.items():
            plt.plot(frames, counts, label=word)

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
        print(len(self.word_memory))
        for i, memory in enumerate(self.word_memory):
            if word == memory[1]:
                word_count += 1
                if i+1 == len(self.word_memory) and  word_count >= 8:
                    print("単語の切り目", start_frame, memory[0], word)
            else:
                if word_count >= 8 and word != "None":
                    print("単語の切り目", start_frame, memory[0], word)
                word_count = 0
                start_frame = memory[0]
                word = memory[1]
        # self.plot_word_counts(self.resumeList)
        print(self.resumeList[0])

    def close_all(self):
        print("完了")
        self.split_word()
        cap.release()
        cv2.destroyAllWindows()
        #sys.exit()

if __name__ == "__main__":
    app = WebCam(True, './test/おはよう/04.mp4', "a")
    app.root.mainloop()
