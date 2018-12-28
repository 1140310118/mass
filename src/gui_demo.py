"""
@author: zhangyice
@time: 2018/12/28

python3
"""

from tkinter import *
from tkinter import messagebox

class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.resizable(False, False)
        self.root.title("sin函数绘制")

    def add_label(self):
        Label(self.root, text='A=').place(x=20,y=40)

    def add_entry(self):
        self.stringVar = StringVar()
        Entry(
            self.root, 
            textvariable = self.stringVar,
            width=10,
        ).place(x=45,y=40)

    def _show_entry_value(self):
        A_value = self.stringVar.get()
        if A_value == '':
            message = "A的值为空"
        else:
            message = "A=%s"%A_value
        messagebox.showinfo(title="提示框",message=message)

    def add_button(self):
        Button(
            self.root, 
            text="显示文本框内的值",
            command=self._show_entry_value,
        ).place(x=80,y=100)

    def mainloop(self):
        self.root.mainloop()


if __name__ == '__main__':
    gui = GUI()
    gui.add_label()
    gui.add_entry()
    gui.add_button()
    gui.mainloop()
