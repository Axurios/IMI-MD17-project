"""
This module is the main module of the program,
and can be executed to run the needed part of the project
"""
import os
import pandas as pd
from tkinter import Tk, Label, Button, StringVar, IntVar, Scale, HORIZONTAL, Text # noqa:
from helper import check_requirements, select_file
# import Descriptor
# selected_file_path = None  # Variable to store the selected file path


def change_name(easier, file):
    fileName = file.get()
    if fileName:
        easierName = os.path.basename(fileName)
        # might have to change it for when not basename compatible
        easier.set(easierName)  # Update the label to show just the filename
    else:
        file_name_var.set("No file selected")


def display(var):
    # Construct the full path from the current directory and the filename
    file_path = var.get()
    # print("Selected file name:", file_name)  # Debugging: Print the file name
    try:
        # Check if the file exists before loading
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")

        # Load the .csv file
        df = pd.read_csv(file_path)
        # Get the head of the dataframe
        data_head = df.head().to_string(index=False)
        # Update the text widget with the data head
        data_display.delete(1.0, "end")  # Clear previous content
        data_display.insert("end", data_head)
    except Exception as e:
        # If there is an error, display it in the text widget
        data_display.delete(1.0, "end")
        data_display.insert("end", f"Error loading file: {e}")


if __name__ == "__main__":
    # check if packages of "requirements.txt" installed # noqa:
    check_requirements()

    # Initialize main window
    fenetre = Tk()
    fenetre.title("ATK title")

    # Label to display the selected file name
    file_name_var = StringVar()
    file_name_var.set("No file selected")  # Default text
    easier_name = StringVar()
    easier_name.set("No file selected")
    file_name_label = Label(fenetre, textvariable=easier_name)
    file_name_label.pack()
    # Button to trigger file selection, using lambda to capture file_name_var
    button = Button(fenetre, text="Select CSV File", command=lambda: select_file(file_name_var)) # noqa:
    button.pack()

    # Trace the variable to call test() when it changes
    file_name_var.trace_add("write", lambda *args: change_name(easier_name, file_name_var)) # noqa:
    file_name_var.trace_add("write", lambda *args: display(file_name_var)) # noqa:

    # Scale to select a percentage (1 to 100), default at 50%
    # proportion of training data turned into boltzman ?
    percentage_var = IntVar(value=50)
    percentage_scale = Scale(fenetre, from_=1, to=100, orient=HORIZONTAL,
        variable=percentage_var, label="Select Percentage", length=100) # noqa:
    percentage_scale.pack()

    # Text widget to display the head of the CSV data
    data_display = Text(fenetre, height=10, width=50)
    data_display.pack()

    # Main loop
    fenetre.mainloop()
