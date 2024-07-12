from tkinter import Tk, Frame, Canvas, Button, Label, StringVar
from PIL import Image, ImageDraw
import handwritten_digits_recognition


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("28x28 Drawing Canvas")
        
        self.string_var1 = StringVar()
        self.string_var1.set("")
        
        self.frame1 = Frame(root)
        self.frame1.pack()

        self.frame2 = Frame(root)
        self.frame2.pack()
        
        self.frame3 = Frame(root, height=100)
        self.frame3.pack()

        self.label = Label(self.frame3, textvariable=self.string_var1)
        self.label.pack()
        
        self.array=[[256]*28 for i in range(28)]
        print(self.array)

        # Set the size of each "pixel" on the canvas
        self.pixel_size = 20  # Adjust pixel size for better drawing precision

        # Create the canvas widget with an enlarged size
        self.canvas = Canvas(self.frame2, width=28*self.pixel_size, height=28*self.pixel_size, bg="white")
        self.canvas.pack()

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # Initialize the 28x28 grid with all white "pixels"
        self.pixels = [[self.canvas.create_rectangle(j*self.pixel_size, i*self.pixel_size,
                                                     (j+1)*self.pixel_size, (i+1)*self.pixel_size,
                                                     outline="lightgrey", fill="white")
                        for j in range(28)] for i in range(28)]

        # Create and pack the save button in frame1
        self.save_button = Button(self.frame1, text="Guess", command=self.guess)
        self.save_button.pack(side="left", padx=5, pady=5)

        # Create and pack another button (e.g., Clear) in frame1
        self.clear_button = Button(self.frame1, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side="right", padx=5, pady=5)

    def paint(self, event):
        # Calculate the row and column index of the rectangle to fill
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size

        # Ensure the row and col are within the bounds of the canvas
        if 0 <= row < 28 and 0 <= col < 28:
            # Fill the rectangle with black color to simulate drawing
            self.canvas.itemconfig(self.pixels[row][col], fill="black")
            self.canvas.itemconfig(self.pixels[row][col+1],fill="black")
            self.canvas.itemconfig(self.pixels[row][col-1],fill="black")
            self.canvas.itemconfig(self.pixels[row+1][col], fill="black")
            self.canvas.itemconfig(self.pixels[row+1][col+1],fill="black")
            self.canvas.itemconfig(self.pixels[row+1][col-1],fill="black")
            self.canvas.itemconfig(self.pixels[row-1][col], fill="black")
            self.canvas.itemconfig(self.pixels[row-1][col+1],fill="black")
            self.canvas.itemconfig(self.pixels[row-1][col-1],fill="black")
        



            self.array[row][col]=0
            self.array[row][col+1]=0
            self.array[row][col-1]=0
            self.array[row+1][col]=0
            self.array[row+1][col+1]=0
            self.array[row+1][col-1]=0
            self.array[row-1][col]=0
            self.array[row-1][col+1]=0
            self.array[row-1][col-1]=0            
          
            

    def guess(self):
        image = Image.new("L", (28, 28))  # "L" mode for grayscale, 255 for white background
        pixel_data = [pixel for row in self.array for pixel in row]

        image.putdata(pixel_data)

        draw = ImageDraw.Draw(image)

        
        print(self.array)
        image.save("image.png")  

        # Perform recognition
        digit, probability = handwritten_digits_recognition.guesser('image.png')

        self.string_var1.set(f"The digit is {digit}")
        self.label.configure(text=self.string_var1.get())

    def clear_canvas(self):
        # Clear the canvas by setting all pixels to white
        for row in range(28):
            for col in range(28):
                self.canvas.itemconfig(self.pixels[row][col], fill="white")
        
        self.array=[[256]*28 for i in range(28)]


if __name__ == "__main__":
    root = Tk()
    app = DrawingApp(root)
    root.mainloop()
