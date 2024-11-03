import os
import pandas as pd
from tkinter import Tk, Label, Button, StringVar, IntVar, Scale, HORIZONTAL, Text  # noqa
from helper import select_file  # noqa


class GUI:
    def __init__(self):
        self.fenetre = Tk()
        self.fenetre.title("ATK title")

        # Variables to hold file information
        self.file_name_var = StringVar(value="No file selected")
        self.easier_name = StringVar(value="No file selected")

        # Build the GUI components
        self.build()
        self.percentBar()
        # Start the main loop
        self.fenetre.mainloop()

    def build(self):
        # Label to display the selected file name
        file_name_label = Label(self.fenetre, textvariable=self.easier_name)
        file_name_label.pack()

        # Button to trigger file selection
        button = Button(self.fenetre, text="Select CSV File", command=self.select_file) # noqa:
        button.pack()

        # Trace the variable to call methods when it changes
        self.file_name_var.trace_add("write", lambda *args: self.change_name())
        self.file_name_var.trace_add("write", lambda *args: self.display())

        # Text widget to display the head of the CSV data
        self.data_display = Text(self.fenetre, height=7, width=50)
        self.data_display.pack()

    def percentBar(self):
        # Scale to select a percentage (1 to 100)
        self.percentage_var = IntVar(value=50)
        percentage_scale = Scale(
            self.fenetre, from_=1, to=100, orient=HORIZONTAL,
            variable=self.percentage_var, label="Select Percentage", length=100
        )
        percentage_scale.pack()

    def select_file(self):
        # Function to select the file and update the file name variable
        selected_file = select_file(self.file_name_var)
        if selected_file:
            self.file_name_var.set(selected_file)

    def change_name(self):
        file_name = self.file_name_var.get()
        if file_name:
            easier_name = os.path.basename(file_name)
            self.easier_name.set(easier_name)  # Update filename
        else:
            self.easier_name.set("No file selected")

    def display(self):
        # Construct the full path from the current directory and the filename
        file_path = self.file_name_var.get()
        try:
            # Check if the file exists before loading
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"No such file: '{file_path}'")

            # Load the .csv file
            df = pd.read_csv(file_path)
            # Get the head of the dataframe
            data_head = df.head().to_string(index=False)
            # Update the text widget with the data head
            self.data_display.delete(1.0, "end")  # Clear previous content
            self.data_display.insert("end", data_head)
        except Exception as e:
            # If there is an error, display it in the text widget
            self.data_display.delete(1.0, "end")
            self.data_display.insert("end", f"Error loading file: {e}")
