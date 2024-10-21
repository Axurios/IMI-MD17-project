"""
This module is the main module of the program,
and can be executed to run the needed part of the project
"""

from tkinter import Tk, Label

# Entry point of the program
if __name__ == "__main__":
    # create several players
    print("ok")
    fenetre = Tk()

    label = Label(fenetre, text="Hello World")
    label.pack()

    fenetre.mainloop()
