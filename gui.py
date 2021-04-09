from tkinter import *
from tkinter.ttk import Progressbar, Style
import lib


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.master.title("Kpop reddit analyzer")
        self.pack(fill=BOTH, expand=1)

        self.set_layout()
        # self.load_statistics()
        self.processor = lib.Processor()

    def printer(self, x):
        (sent, label) = x
        label = label[0]
        self.LABEL_OUTPUT.configure(state="normal")
        self.LABEL_OUTPUT.delete("1.0", END)
        if label == -1:
            self.LABEL_OUTPUT.configure(bg="brown1")
            self.LABEL_OUTPUT.insert(END, '-1 ', "center")
        elif label == 0:
            self.LABEL_OUTPUT.configure(bg="khaki1")
            self.LABEL_OUTPUT.insert(END, '0', "center")
        elif label == 1:
            self.LABEL_OUTPUT.configure(bg="pale green")
            self.LABEL_OUTPUT.insert(END, '1', "center")
        self.LABEL_OUTPUT.configure(state="disable")
        self.PROGRESS_BAR.stop()

    def classify(self):
        self.PROGRESS_BAR.start()
        l = self.SENTENCE_INPUT.get("1.0", 'end-1c')
        self.processor.classify(l, self.printer)

    def load_statistics(self):
        f = open("test data/results.txt", "r")
        for x in f:
            self.STATISTICS_TEXT.insert(END, x, "center")
        f.close()

    def set_layout(self):
        self.configure(bg="honeydew3")
        self.DIV_TOP = Frame(self, padx=5, pady=5, bg="honeydew3")
        self.DIV_TEXT = Frame(self, bg="honeydew3")

        self.TEXT_SCROLLBAR = Scrollbar(self.DIV_TEXT)
        self.SENTENCE_INPUT = Text(self.DIV_TEXT, height=6, width=40, bg="honeydew2", font=("Courier", 12))

        self.TOP_CENTER_DIV = Frame(self.DIV_TOP, bg="honeydew3")
        self.B_SUBMIT = Button(self.TOP_CENTER_DIV, width=5, text="Submit", command=self.classify, bg="honeydew2",
                               font=("Courier", 12), padx=5)
        self.PROGRESS_BAR = Progressbar(self.TOP_CENTER_DIV, orient=HORIZONTAL, mode='indeterminate', length=50)
        self.LABEL_OUTPUT = Text(self, width=5, height=1, state="disabled", bg="honeydew2", font=("Courier", 32))

        self.SENTENCE_INPUT.insert(END, "Input goes here.")
        self.LABEL_OUTPUT.tag_configure("center", justify='center')
        self.LABEL_OUTPUT.insert(END, "", "center")

        self.DIV_CENTER = Frame(self, padx=5, bg="honeydew3")
        self.B_STATISTICS = Label(self.DIV_CENTER, text="Statistics", bg="honeydew2", font=("Courier", 12), padx=5)
        self.DIV_STATISTICS_TEXT = Frame(self.DIV_CENTER, bg="honeydew3")
        self.STATISTICS_TEXT = Text(self.DIV_STATISTICS_TEXT, height=16, width=60, bg="honeydew2", font=("Courier", 10))
        self.STATISTICS_SCROLLBAR = Scrollbar(self.DIV_STATISTICS_TEXT, bg="honeydew3")

        Frame(self, height=10, bg="honeydew3").pack(side=TOP)
        self.DIV_TEXT.pack(side=TOP, anchor=CENTER)
        self.SENTENCE_INPUT.pack(side=LEFT, fill=Y)
        self.TEXT_SCROLLBAR.pack(side=LEFT, fill=Y)
        self.DIV_TOP.pack(side=TOP)
        Frame(self.DIV_TOP, width=10, bg="honeydew3").pack(side=LEFT)
        self.TOP_CENTER_DIV.pack(side=LEFT)

        self.B_SUBMIT.pack(side=TOP)
        Frame(self.TOP_CENTER_DIV, height=5, bg="honeydew3").pack(side=TOP)
        self.PROGRESS_BAR.pack(side=TOP)
        Frame(self.DIV_TOP, width=10, bg="honeydew3").pack(side=LEFT)
        self.LABEL_OUTPUT.pack(side=TOP)

        Frame(self.DIV_CENTER, height=20, bg="honeydew3").pack(side=TOP)
        self.DIV_CENTER.pack(side=TOP, fill=Y)
        self.B_STATISTICS.pack(side=TOP)
        Frame(self.DIV_CENTER, height=10, bg="honeydew3").pack(side=TOP)
        self.DIV_STATISTICS_TEXT.pack(side=TOP)
        self.STATISTICS_TEXT.pack(side=LEFT, fill=Y)
        self.STATISTICS_SCROLLBAR.pack(side=RIGHT, fill=Y)

        self.SENTENCE_INPUT.config(yscrollcommand=self.TEXT_SCROLLBAR.set)
        self.TEXT_SCROLLBAR.config(command=self.SENTENCE_INPUT.yview)

        self.STATISTICS_TEXT.config(yscrollcommand=self.STATISTICS_SCROLLBAR.set)
        self.STATISTICS_SCROLLBAR.config(command=self.STATISTICS_TEXT.yview)


if __name__ == "__main__":
    root = Tk()
    root.style = Style()
    root.style.theme_use("clam")
    root.geometry("515x540")
    app = Window(root)
    root.mainloop()
