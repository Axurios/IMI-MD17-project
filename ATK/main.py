"""
This module is the main module of the program,
and can be executed to run the needed part of the project
"""

from tkinter import Tk, Label
import Descriptor

# Entry point of the program
if __name__ == "__main__":
    print("ok")
    Descriptor.display()
    fenetre = Tk()

    label = Label(fenetre, text="Hello World")
    label.pack()

    fenetre.mainloop()
